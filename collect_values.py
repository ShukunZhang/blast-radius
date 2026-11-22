"""Record what every single-operator comparison in sqlglot saw while the whole suite ran.

For each comparison site (file:line:col of the `Compare` node in the pinned source) the map keeps:

- `ops`: the operators that returned the same result as the original at every evaluation;
- `values`: when one side is a constant, every value the other side took (scalars by value, objects
  that compare by identity as "o"); "many" once there are more than VALUE_CAP distinct values;
- `raised`: the comparison raised at least once.

A patch that only swaps a site's operator for one in `ops`, or only changes its constant so that every
recorded value compares the same, leaves every evaluation and so every test unchanged. Sites the suite
never evaluated are absent. The suite runs twice under different hash seeds; the map keeps the union of
values and the intersection of `ops`, and is empty when either run did not match the expected failures.

    python collect_values.py SQLGLOT_CHECKOUT OUT.json [EXPECTED_FAILURES.json]
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import json
import operator
import os
import subprocess
import sys
from pathlib import Path

VALUE_CAP = 128
OPS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtE": operator.le,
    "Gt": operator.gt,
    "GtE": operator.ge,
    "Is": operator.is_,
    "IsNot": operator.is_not,
    "In": lambda left, right: left in right,
    "NotIn": lambda left, right: left not in right,
}
NAMES = tuple(OPS)
CODE = {name: index for index, name in enumerate(NAMES)}
FUNCS = tuple(OPS[name] for name in NAMES)
ALL = (1 << len(NAMES)) - 1
_ORDER = (CODE["Eq"], CODE["NotEq"], CODE["Lt"], CODE["LtE"], CODE["Gt"], CODE["GtE"])
_IDENTITY = (CODE["Eq"], CODE["NotEq"], CODE["Is"], CODE["IsNot"])
# Adding the empty value turns a subclass instance into the plain builtin without calling its __str__.
_SCALARS = ((int, 0), (float, 0.0), (str, ""))
_COMPARISONS = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")
_EXPRESSION: type | None = None


def constant_display(node: ast.AST) -> bool:
    """A literal, or a tuple/list/set display of literals."""
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(isinstance(e, ast.Constant) for e in node.elts)


def site_key(relative: str, node: ast.Compare) -> str:
    return f"{relative}:{node.lineno}:{node.col_offset}"


def _plain(value: object) -> object | None:
    """The builtin scalar a value compares as, or None when its type changes comparison."""
    kind = type(value)
    if value is None or kind is bool:
        return value
    for base, empty in _SCALARS:
        if isinstance(value, base):
            if all(getattr(kind, name) is getattr(base, name) for name in _COMPARISONS):
                return value if kind is base else base.__add__(value, empty)
            return None
    if kind is tuple:
        items = [_plain(item) for item in value]
        return tuple(items) if all(item is not None or orig is None for item, orig in zip(items, value)) else None
    return None


def _identity_only(value: object) -> bool:
    kind = type(value)
    return all(getattr(kind, name) is getattr(object, name) for name in _COMPARISONS)


def _expression(value: object) -> bool:
    """A sqlglot expression: `==` is `self is other or (type(self) is type(other) and ...)`, so against a
    literal it is identity and runs nothing else."""
    global _EXPRESSION
    if _EXPRESSION is None:
        _EXPRESSION = getattr(sys.modules.get("sqlglot.expressions.core"), "Expression", None)
        if _EXPRESSION is None:
            return False
    kind = type(value)
    return kind.__eq__ is _EXPRESSION.__eq__ and kind.__ne__ is _EXPRESSION.__ne__


def canonical(value: object) -> list | str:
    """A value compared against a literal: its plain scalar, "o" when that comparison is identity, else "x"."""
    plain = _plain(value)
    if plain is not None or value is None:
        return ["v", plain]
    return "o" if _identity_only(value) or _expression(value) else "x"


class Sites:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.const: list[int] = []
        self.ops: list[int] = []
        self.values: list[set[str] | None] = []
        self.raised: list[bool] = []
        self.seen: list[bool] = []
        self.probing = False

    def add(self, key: str, const_side: int) -> int:
        self.keys.append(key)
        self.const.append(const_side)
        self.ops.append(ALL)
        self.values.append(set())
        self.raised.append(False)
        self.seen.append(False)
        return len(self.keys) - 1

    def compare(self, site: int, left: object, right: object, code: int) -> object:
        if self.probing:
            # Comparisons inside an alternative operator's evaluation never happen in the real run.
            return FUNCS[code](left, right)
        self.seen[site] = True
        try:
            result = FUNCS[code](left, right)
        except BaseException:
            self.raised[site] = True
            raise
        if self.ops[site]:
            self.probing = True
            try:
                self.ops[site] &= _agreeing(left, right, code, result)
            finally:
                self.probing = False
        side = self.const[site]
        values = self.values[site]
        if side and values is not None:
            values.add(json.dumps(canonical(right if side == 1 else left)))
            if len(values) > VALUE_CAP:
                self.values[site] = None
        return result

    def dump(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for index, key in enumerate(self.keys):
            if not self.seen[index]:
                continue
            values = self.values[index]
            out[key] = {
                "ops": [name for bit, name in enumerate(NAMES) if self.ops[index] >> bit & 1],
                "values": "many" if values is None else sorted(values) if self.const[index] else None,
                "raised": self.raised[index],
            }
        return out


def _agreeing(left: object, right: object, code: int, result: object) -> int:
    """Bitmask of operators that return `result` here without running any user-defined method."""
    mask = 1 << code
    safe_left = _plain(left) is not None or left is None
    safe_right = _plain(right) is not None or right is None
    if safe_left and safe_right:
        candidates = _ORDER + _IDENTITY
    elif (safe_left or _identity_only(left)) and (safe_right or _identity_only(right)):
        candidates = _IDENTITY
    elif (safe_left and _expression(right)) or (safe_right and _expression(left)):
        candidates = _IDENTITY
    else:
        candidates = (CODE["Is"], CODE["IsNot"])
    for other in candidates:
        if other == code:
            continue
        try:
            value = FUNCS[other](left, right)
        except Exception:  # noqa: BLE001 - an operator that raises here does not agree
            continue
        if type(value) is type(result) and value == result:
            mask |= 1 << other
    return mask


SITES = Sites()


class _Instrument(ast.NodeTransformer):
    def __init__(self, relative: str) -> None:
        self.relative = relative

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]).__name__ not in CODE:
            return node
        right = node.comparators[0]
        side = 2 if constant_display(right) and not constant_display(node.left) else 0
        side = 1 if constant_display(node.left) and not constant_display(right) else side
        site = SITES.add(site_key(self.relative, node), side)
        call = ast.Call(
            ast.Name("_br_compare", ast.Load()),
            [ast.Constant(site), node.left, right, ast.Constant(CODE[type(node.ops[0]).__name__])],
            [],
        )
        return ast.copy_location(call, node)


class _Loader(importlib.machinery.SourceFileLoader):
    root = Path()

    def get_code(self, fullname: str):  # type: ignore[no-untyped-def]
        source = self.get_data(self.path)
        relative = Path(self.path).resolve().relative_to(self.root).as_posix()
        tree = _Instrument(relative).visit(ast.parse(source))
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)

    def exec_module(self, module) -> None:  # type: ignore[no-untyped-def]
        module._br_compare = SITES.compare
        super().exec_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        prefix = str(_Loader.root / "sqlglot")
        if spec is not None and isinstance(spec.origin, str) and spec.origin.startswith(prefix):
            spec.loader = _Loader(fullname, spec.origin)
            return spec
        return None


def _record(repo: Path, out: Path) -> None:
    import pytest

    import concurrent.futures
    import concurrent.futures.process

    from inline_pool import InlinePool

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    _Loader.root = repo
    concurrent.futures.ProcessPoolExecutor = InlinePool  # type: ignore[misc]
    concurrent.futures.process.ProcessPoolExecutor = InlinePool  # type: ignore[misc]
    sys.meta_path.insert(0, _Finder())
    failed: list[str] = []

    class Plugin:
        def pytest_runtest_logreport(self, report) -> None:  # type: ignore[no-untyped-def]
            if report.failed:
                failed.append(report.nodeid)

    pytest.main(
        [
            "tests",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "--ignore=tests/test_docs.py",
            "--ignore=tests/leakcheck.py",
            "--ignore=tests/gen_fixtures.py",
        ],
        plugins=[Plugin()],
    )
    out.write_text(json.dumps({"failed": sorted(set(failed)), "sites": SITES.dump()}))


def _merge(runs: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for run in runs:
        for key, entry in run["sites"].items():
            have = merged.get(key)
            if have is None:
                merged[key] = dict(entry)
                continue
            have["ops"] = [name for name in have["ops"] if name in entry["ops"]]
            have["raised"] = have["raised"] or entry["raised"]
            if have["values"] is None or entry["values"] is None:
                have["values"] = have["values"] if entry["values"] is None else entry["values"]
            elif "many" in (have["values"], entry["values"]):
                have["values"] = "many"
            else:
                union = sorted(set(have["values"]) | set(entry["values"]))
                have["values"] = union if len(union) <= VALUE_CAP else "many"
    return merged


def main() -> None:
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if os.environ.get("BR_VALUES_CHILD"):
        _record(repo, out)
        return
    expected = set(json.loads(Path(sys.argv[3]).read_text())) if len(sys.argv) > 3 else None
    runs = []
    for seed in ("1", "2"):
        part = out.with_suffix(f".seed{seed}.json")
        env = dict(os.environ, BR_VALUES_CHILD="1", PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", __file__, str(repo), str(part)], env=env, check=False)
        runs.append(json.loads(part.read_text()))
        part.unlink()
    ok = all(expected is None or set(run["failed"]) == expected for run in runs)
    for run in runs:
        print(f"[values] failed {len(run['failed'])}: {run['failed'][:10]}", file=sys.stderr)
    sites = _merge(runs) if ok else {}
    out.write_text(json.dumps({"sites": sites}, separators=(",", ":"), sort_keys=True))
    print(f"[values] wrote {out} sites={len(sites)} ok={ok}", file=sys.stderr)


if __name__ == "__main__":
    main()
