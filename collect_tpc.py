"""Build maps/tpc.json: pinned per-query records and runtime lines for TestExecutor's TPC tests.

Usage: python collect_tpc.py SQLGLOT_ROOT OUT_JSON

Each test runs in its own child process with its pool inlined. Lines of sqlglot
that execute while it runs are recorded, except inside the steps
tpc_fingerprint digests (parsing, pickling the parsed query, transpiling,
finding tables, building the schema, optimizing, rendering the optimized SQL
and planning). Python code generation depends on rows reaching subqueries, so
it is recorded rather than digested. An edit that reaches none of the
recorded lines changes the test only through those digests. Beside the map
it writes the digest of the state sqlglot builds at import (import_state),
taken under several hash seeds.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import types
import json
import os
import pickle
import subprocess
import sys
import unittest
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path

import import_state
from import_state import import_everything

HERE = Path(__file__).resolve().parent
TESTS = {"tpch": "test_execute_tpch", "tpcds": "test_execute_tpcds"}
_RECORDING = [True]


@contextlib.contextmanager
def _paused() -> Iterator[None]:
    previous = _RECORDING[0]
    _RECORDING[0] = False
    try:
        yield
    finally:
        _RECORDING[0] = previous


def _quiet(function: Callable) -> Callable:
    def wrapper(*args: object, **kwargs: object) -> object:
        with _paused():
            return function(*args, **kwargs)

    return wrapper


class _Pool:
    """In-process pool. The submitted query crosses pickle unrecorded; the rest is recorded."""

    def __init__(self, *_args: object, initializer: Callable | None = None, initargs: tuple = (), **_kw: object):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def __enter__(self) -> _Pool:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def submit(self, function: Callable, /, *args: object, **kwargs: object) -> Future:
        future: Future = Future()
        with _paused():
            args, kwargs = pickle.loads(pickle.dumps((args, kwargs)))
        try:
            future.set_result(pickle.loads(pickle.dumps(function(*args, **kwargs))))
        except Exception as error:  # noqa: BLE001 - surfaced through the future as in a real pool
            future.set_exception(error)
        return future

    def shutdown(self, *_args: object, **_kwargs: object) -> None:
        return None


def _quiet_log_sql(caller: object) -> None:
    """Pause for the SQL that execute() builds for its debug log; tpc_fingerprint digests it."""
    from sqlglot import exp

    owner = next(cls for cls in exp.Select.__mro__ if "sql" in vars(cls))
    original = owner.sql

    def sql(self: object, *args: object, **kwargs: object) -> object:
        if sys._getframe(1).f_code is caller:
            with _paused():
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    owner.sql = sql


def _import_everything(_root: Path) -> None:
    import_everything()


def _run_one(root: Path, suite: str) -> dict[str, dict[str, list[int]]]:
    """Run one TPC test with recording on; return the sqlglot lines it ran and the function lines run at import.

    What functions compute at import (metaclasses, tries, helpers called from
    class bodies) is kept in class and module attributes the test reads later,
    so those function bodies count as reached too. Module- and class-level
    statements run at import are left to the name check instead.
    """
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    package = str(root / "sqlglot") + os.sep
    lines: dict[str, set[int]] = {}
    imported: dict[str, set[int]] = {}
    importing = [True]
    monitoring = sys.monitoring
    tool = monitoring.COVERAGE_ID

    def on_line(code: types.CodeType, line: int) -> object:
        filename = code.co_filename
        if not filename.startswith(package):
            return monitoring.DISABLE
        path = os.path.relpath(filename, root).replace(os.sep, "/")
        if importing[0]:
            if code.co_flags & inspect.CO_OPTIMIZED:
                imported.setdefault(path, set()).add(line)
            return monitoring.DISABLE
        if not _RECORDING[0]:
            return None
        lines.setdefault(path, set()).add(line)
        return monitoring.DISABLE

    monitoring.use_tool_id(tool, "collect_tpc")
    monitoring.register_callback(tool, monitoring.events.LINE, on_line)
    monitoring.set_events(tool, monitoring.events.LINE)
    _import_everything(root)
    import sqlglot.executor as executor_module
    import tests.test_executor as module

    importing[0] = False
    monitoring.restart_events()
    module.ProcessPoolExecutor = _Pool
    for name in ("parse_one", "transpile", "find_tables"):
        setattr(module, name, _quiet(getattr(module, name)))
    for name in ("optimize", "Plan", "ensure_schema"):
        setattr(executor_module, name, _quiet(getattr(executor_module, name)))
    _quiet_log_sql(executor_module.execute.__code__)

    module.TestExecutor.setUpClass()
    result = unittest.TestResult()
    module.TestExecutor(TESTS[suite]).run(result)
    monitoring.set_events(tool, 0)
    if not result.wasSuccessful():
        raise SystemExit(f"{suite} failed on the pinned commit: {result.errors + result.failures}")
    return {
        "after": {path: sorted(found) for path, found in sorted(lines.items())},
        "imported": {path: sorted(found) for path, found in sorted(imported.items())},
    }


def _word(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.alias):
        return (node.asname or node.name).split(".")
    if isinstance(node, ast.keyword) and node.arg:
        return [node.arg]
    return []


def _words(node: ast.AST) -> set[str]:
    return {word for child in ast.walk(node) for word in _word(child)}


def _names(root: Path, runtime: dict[str, list[int]], after: set[str]) -> list[str]:
    """Names that, when rebound, can change what the runtime lines compute.

    Every name, attribute and string constant written on a runtime line; every
    name read by code that runs while classes are created, which derives
    attributes those lines may read under other names; and, in the modules
    `after` that ran after import, every name read by a module- or class-level
    statement that binds one of these, since its value reaches them.
    """
    trees = {
        path.relative_to(root).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted((root / "sqlglot").rglob("*.py"))
    }
    found: set[str] = set()
    for path, lines in runtime.items():
        wanted = set(lines)
        for node in ast.walk(trees[path]):
            if hasattr(node, "lineno") and wanted.intersection(range(node.lineno, (node.end_lineno or node.lineno) + 1)):
                found.update(_word(node))
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                creates = any(isinstance(base, ast.Name) and base.id == "type" for base in node.bases)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (creates or item.name in _CREATION):
                        found |= _words(item)
    statements = [
        (_bound(statement), _words(statement))
        for path, tree in trees.items()
        if path in after
        for statement in _outside_functions(tree)
    ]
    changed = True
    while changed:
        changed = False
        for bound, read in statements:
            if bound & found and not read <= found:
                found |= read
                changed = True
    return sorted(found)


_CREATION = {"__new__", "__init_subclass__", "__set_name__", "__class_getitem__"}


def _outside_functions(tree: ast.Module) -> list[ast.stmt]:
    """Module- and class-level statements, with a class standing for its header only."""
    found: list[ast.stmt] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        statement = pending.pop()
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.ClassDef):
            header = ast.ClassDef(
                name=statement.name,
                bases=statement.bases,
                keywords=statement.keywords,
                body=[],
                decorator_list=statement.decorator_list,
                type_params=[],
            )
            found.append(header)
            pending.extend(statement.body)
            continue
        nested = [child for child in ast.iter_child_nodes(statement) if isinstance(child, ast.stmt)]
        if nested:
            pending.extend(nested)
        found.append(statement)
    return found


def _bound(node: ast.stmt) -> set[str]:
    """Names a statement binds, as tpc_proof checks them."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    found: set[str] = set()
    for each in ast.walk(node):
        if isinstance(each, ast.alias):
            found.update((each.asname or each.name).split("."))
        elif isinstance(each, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(each.ctx, (ast.Store, ast.Del)):
            target: ast.AST = each
            while isinstance(target, ast.Subscript):
                target = target.value
            if isinstance(target, ast.Name):
                found.add(target.id)
            elif isinstance(target, ast.Attribute):
                found.add(target.attr)
    return found


def _record(root: Path, suite: str) -> list[dict[str, str] | None]:
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(HERE))
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    import tpc_fingerprint

    return tpc_fingerprint.record(suite)


def _stable(first: list, second: list, suite: str) -> list[dict[str, str | None] | None]:
    """The two runs' entries, with a result that differs between them left unknown; other differences abort."""
    merged: list[dict[str, str | None] | None] = []
    for one, two in zip(first, second, strict=True):
        if one is None or two is None:
            if one != two:
                raise SystemExit(f"{suite} executes different queries between runs")
            merged.append(None)
            continue
        if (one["transpile"], one["plan"]) != (two["transpile"], two["plan"]):
            raise SystemExit(f"{suite} plans differ between runs")
        stages = [first if first == second else None for first, second in zip(one["stages"], two["stages"], strict=True)]
        merged.append({**one, "result": one["result"] if one["result"] == two["result"] else None, "stages": stages})
    return merged


def _child(kind: str, root: Path, suite: str, seed: str = "random") -> object:
    command = [sys.executable, "-B", __file__, "--child", kind, str(root), suite]
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True, env=environment).stdout)


def _child_main(kind: str, root: Path, suite: str) -> object:
    if kind == "runtime":
        return _run_one(root, suite)
    if kind == "lines":
        sys.path.insert(1, str(HERE))
        import tpc_queries

        return tpc_queries.record_lines(root, suite)
    if kind == "state":
        sys.path.insert(0, str(root))
        os.chdir(root)
        os.environ["SKIP_INTEGRATION"] = "0"
        return import_state.record()
    return _record(root, suite)


SEEDS = ("1", "2", "3")
# A probe's parent: the selector's modules loaded, then the warm import, then everything else.
_SELECTOR_STATE = (
    "import json, os, sys; root = sys.argv[1]; sys.path[:0] = [root, {here!r}]; os.chdir(root); "
    "os.environ['SKIP_INTEGRATION'] = '0'; import selector, import_state; from warm_probe import WARM_MODULES; "
    "print(json.dumps(import_state.record(WARM_MODULES)))"
)


def _selector_state(root: Path, seed: str) -> object:
    command = [sys.executable, "-B", "-c", _SELECTOR_STATE.format(here=str(HERE)), str(root)]
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True, env=environment).stdout)


def main() -> None:
    if sys.argv[1] == "--child":
        sys.stdout.write(json.dumps(_child_main(sys.argv[2], Path(sys.argv[3]), sys.argv[4])))
        return
    import tpc_queries

    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    built = {}
    for suite in TESTS:
        digests = _stable(_child("record", root, suite), _child("record", root, suite), suite)
        ran = _child("runtime", root, suite)
        runtime = {
            path: sorted(set(ran["after"].get(path, [])) | set(ran["imported"].get(path, [])))
            for path in sorted(set(ran["after"]) | set(ran["imported"]))
        }
        queries = tpc_queries.merge([_child("lines", root, suite, seed) for seed in SEEDS])
        queries["imported"] = ran["imported"]
        built[suite] = {
            "digests": digests,
            "runtime": runtime,
            "names": _names(root, runtime, set(ran["after"])),
            "queries": queries,
        }
        unknown = sum(1 for entry in digests if entry is not None and entry["result"] is None)
        print(suite, len(digests), "unstable results", unknown, sum(map(len, runtime.values())), file=sys.stderr)
    runs = [_child("state", root, "", seed) for seed in SEEDS] + [_selector_state(root, seed) for seed in SEEDS[:2]]
    state = import_state.stable(runs)  # type: ignore[arg-type]
    print("import state: unordered", len(state["unordered"]), "skipped", state["skip"], file=sys.stderr)  # type: ignore[arg-type]
    out.write_text(json.dumps(built, separators=(",", ":")), encoding="utf-8")
    import_state.state_path(out).write_text(json.dumps(state, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
