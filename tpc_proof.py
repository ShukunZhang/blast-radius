"""Drop test_execute_tpch/tpcds when a patch provably leaves them passing.

Both tests compare DuckDB with sqlglot's Python executor on fixture queries.
maps/tpc.json holds, from the pinned commit where they pass, digests per
executed query of its DuckDB SQL, its plan and its result (tpc_fingerprint),
the sqlglot lines run after planning or by functions at import, and the names
those lines read. When the patch edits no such function and binds no such
name, a query whose plan digests as pinned keeps its pinned result; any other
executed query is run in the probe and its result compared.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

from diffutil import DiffApplyError, FileDiff, added_lines, apply_hunks, parse_patch, removed_lines
from probe_small import run_patched

TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds",
}
# Measured CPU of a probe (starting the child and importing sqlglot included): checking digests when no
# edit reaches the code run after planning, and executing every query when one does.
PROBE_CPU_S = {"tpch": 0.35, "tpcds": 4.4}
REPLAY_CPU_S = {"tpch": 2.0, "tpcds": 11.2}
# Patches that reach the code run after planning break these tests about half the time, and a probe that
# finds the break still costs a few seconds, so dropping must save this much more to be worth a replay.
REPLAY_RISK_S = 5.0
PROBE_LIMIT_S = 60.0
_GUARDED_TESTS = {"tests/test_executor.py", "tests/helpers.py", "tests/__init__.py", "tests/conftest.py"}
_INERT_SUFFIXES = (".md", ".rst", ".txt")
BLOCKED, REACHED, UNREACHED = "blocked", "reached", "unreached"


class TpcProof:
    def __init__(self, path: Path, root: Path) -> None:
        self.root = root
        self.suites: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._trees: dict[str, ast.Module] = {}

    def drop_passing(
        self, patch_text: str, selected: set[int], tests: dict[int, dict], cost: Callable[[set[int]], float]
    ) -> set[int]:
        """Selected without the TPC tests proven to pass, when proving costs less than running them."""
        candidates = {
            number: TESTS[tests[number]["id"]] for number in selected if tests[number]["id"] in TESTS
        }
        candidates = {number: suite for number, suite in candidates.items() if suite in self.suites}
        if not candidates or cost(selected) - cost(selected - set(candidates)) <= min(PROBE_CPU_S.values()):
            return selected
        files = parse_patch(patch_text)
        reach = {suite: self._reach(files, suite) for suite in set(candidates.values())}
        chosen = _most_saving(candidates, reach, selected, cost)
        if not chosen:
            return selected
        expected = {suite: (self.suites[suite]["digests"], reach[suite] == UNREACHED) for suite in chosen.values()}
        proven = run_patched(self.root, patch_text, lambda: _passing(expected), PROBE_LIMIT_S)
        if not isinstance(proven, list):
            return selected
        return selected - {number for number, suite in chosen.items() if suite in proven}

    def _reach(self, files: list[FileDiff], suite: str) -> str:
        """BLOCKED when the probe cannot stand in for the test, else whether an edit reaches code it runs after planning."""
        runtime = {path: set(lines) for path, lines in self.suites[suite]["runtime"].items()}
        names = set(self.suites[suite]["names"])
        reach = UNREACHED
        for file_diff in files:
            path = file_diff.path or ""
            if file_diff.binary or (file_diff.old_path is not None and file_diff.new_path != file_diff.old_path):
                return BLOCKED
            if not path.startswith("sqlglot/"):
                if path in _GUARDED_TESTS or path.startswith("tests/fixtures/optimizer/tpc-"):
                    return BLOCKED
                if not (path.startswith("tests/") or path.endswith(_INERT_SUFFIXES)):
                    return BLOCKED
                continue
            if file_diff.old_path is None:
                continue
            if not path.endswith(".py") or path.startswith("sqlglot/executor/"):
                reach = REACHED
            elif reach == UNREACHED and not self._file_unreached(file_diff, runtime, names):
                reach = REACHED
        return reach

    def _file_unreached(self, file_diff: FileDiff, runtime: dict[str, set[int]], names: set[str]) -> bool:
        path = file_diff.old_path or ""
        try:
            old_text = (self.root / path).read_text(encoding="utf-8")
            new_tree = ast.parse(apply_hunks(old_text, file_diff.hunks))
        except (OSError, SyntaxError, DiffApplyError):
            return False
        old_tree = self._tree(path, old_text)
        ran = runtime.get(path, set())
        old_functions = _functions(old_tree)
        for line, _text in removed_lines(file_diff.hunks):
            if not _edit_unreached(_chain(old_tree, line), line, old_functions, ran, names):
                return False
        for line, _text in added_lines(file_diff.hunks):
            if not _edit_unreached(_chain(new_tree, line), line, old_functions, ran, names):
                return False
        return True

    def _tree(self, path: str, text: str) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(text)
        return self._trees[path]


def _most_saving(
    candidates: dict[int, str], reach: dict[str, str], selected: set[int], cost: Callable[[set[int]], float]
) -> dict[int, str]:
    """The provable TPC tests whose dropping saves the most beyond the probe's CPU; empty when none pays."""
    provable = [number for number, suite in candidates.items() if reach[suite] != BLOCKED]
    best: dict[int, str] = {}
    best_gain = 0.0
    full = cost(selected)
    for size in range(1, len(provable) + 1):
        for numbers in combinations(provable, size):
            suites = [candidates[number] for number in numbers]
            probe = sum((PROBE_CPU_S if reach[suite] == UNREACHED else REPLAY_CPU_S)[suite] for suite in suites)
            if any(reach[suite] == REACHED for suite in suites):
                probe += REPLAY_RISK_S
            gain = full - cost(selected - set(numbers)) - probe
            if gain > best_gain:
                best, best_gain = {number: candidates[number] for number in numbers}, gain
    return best


def _passing(expected: dict[str, tuple[list, bool]]) -> list[str]:
    import tpc_fingerprint

    return [suite for suite, (entries, unreached) in expected.items() if tpc_fingerprint.replay(suite, entries, unreached)]


def _start(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _chain(tree: ast.Module, line: int) -> list[ast.stmt]:
    """Statements enclosing line, outermost first."""
    chain: list[ast.stmt] = []
    current: ast.AST = tree
    while True:
        inner = next(
            (
                child
                for child in ast.iter_child_nodes(current)
                if isinstance(child, ast.stmt) and _start(child) <= line <= (child.end_lineno or child.lineno)
            ),
            None,
        )
        if inner is None:
            return chain
        chain.append(inner)
        current = inner


def _qualname(chain: list[ast.stmt]) -> str:
    return ".".join(node.name for node in chain if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))


def _functions(tree: ast.Module) -> dict[str, list[tuple[int, int]]]:
    """Pinned function spans by qualified name."""
    spans: dict[str, list[tuple[int, int]]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    spans.setdefault(name, []).append((_start(child), child.end_lineno or child.lineno))
                visit(child, name + ".")
            elif isinstance(child, ast.stmt):
                visit(child, prefix)

    visit(tree, "")
    return spans


def _bound(node: ast.stmt) -> set[str]:
    """Names a statement binds in its scope, by attribute or subscript base too."""
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


def _edit_unreached(
    chain: list[ast.stmt], line: int, old_functions: dict[str, list[tuple[int, int]]], ran: set[int], names: set[str]
) -> bool:
    """Whether an edit on line, inside chain, stays out of what runs outside the digested steps."""
    function_indexes = [index for index, node in enumerate(chain) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    existing = [index for index in function_indexes if _qualname(chain[: index + 1]) in old_functions]
    for index in existing:
        for start, end in old_functions[_qualname(chain[: index + 1])]:
            if ran.intersection(range(start, end + 1)):
                return False
    added = [index for index in function_indexes if index not in existing]
    if added:
        if existing and existing[0] < added[0]:
            return True
        statement: ast.stmt = chain[added[0]]
    elif function_indexes:
        function = chain[function_indexes[-1]]
        if line >= function.body[0].lineno or len(function_indexes) > 1:
            return True
        statement = function
    elif chain:
        statement = chain[-1]
        if ran.intersection(range(_start(statement), (statement.end_lineno or statement.lineno) + 1)):
            return False
    else:
        return True
    if isinstance(statement, ast.Pass) or (
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
    ):
        return True
    bound = _bound(statement)
    return bool(bound) and not any(name in names or name.startswith("__") for name in bound)
