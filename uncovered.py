"""Select tests the line-coverage map cannot see, using maps/reads.json.

Some tests read sqlglot files as data or import sqlglot in a child process, so
no executed line points at them. `collect_reads.py` records, on the pinned
commit, which files each such test opened and whether it starts a child that
imports sqlglot.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from diffutil import DiffApplyError, FileDiff, apply_hunks

# These modules run code when a child process executes `import sqlglot`; any
# syntax change to them can change what that import loads.
_ENTRY_MODULES = {"sqlglot/__init__.py", "sqlglot/dialects/__init__.py"}


class UncoveredTests:
    def __init__(self, path: Path, numbers: dict[str, int], root: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self.root = root
        self.eager: set[str] = set(payload.get("eager", ()))
        self.readers: dict[str, set[int]] = {}
        self.dir_readers: dict[str, set[int]] = {}
        self.spawners: set[int] = set()
        for nodeid, found in payload.get("tests", {}).items():
            number = numbers.get(nodeid)
            if number is None:
                continue
            reads = found.get("reads", [])
            for relative in reads:
                self.readers.setdefault(relative, set()).add(number)
            # A test that opened several files of one folder enumerates it, so a
            # file added there is read too.
            folders = Counter(str(PurePosixPath(relative).parent) for relative in reads)
            for folder, count in folders.items():
                if count > 1:
                    self.dir_readers.setdefault(folder, set()).add(number)
            if found.get("spawns_sqlglot"):
                self.spawners.add(number)

    def select(self, file_diff: FileDiff) -> set[int]:
        paths = {path for path in (file_diff.old_path, file_diff.new_path) if path}
        selected: set[int] = set()
        trees: tuple[ast.AST | None, ast.AST | None] | None = None
        for path in paths:
            readers = set(self.readers.get(path, ()))
            if file_diff.old_path is None or file_diff.new_path is None:
                readers |= self.dir_readers.get(str(PurePosixPath(path).parent), set())
            if not readers:
                continue
            trees = trees or self._trees(file_diff)
            if _content_changed(path, trees):
                selected |= readers
        if self.spawners and paths & self.eager:
            trees = trees or self._trees(file_diff)
            if paths & _ENTRY_MODULES and _content_changed(file_diff.path or "", trees):
                selected |= self.spawners
            elif _runs_at_import(trees):
                selected |= self.spawners
        return selected

    def _trees(self, file_diff: FileDiff) -> tuple[ast.AST | None, ast.AST | None]:
        """Old and patched syntax trees. None stands for a missing or unparsable side."""
        if not (file_diff.path or "").endswith(".py"):
            return None, None
        original = ""
        if file_diff.old_path is not None:
            source = self.root / file_diff.old_path
            original = source.read_text(encoding="utf-8") if source.is_file() else ""
        try:
            patched = apply_hunks(original, file_diff.hunks) if file_diff.new_path is not None else ""
        except DiffApplyError:
            return None, None
        return _parse(original), _parse(patched)


def _parse(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _content_changed(path: str, trees: tuple[ast.AST | None, ast.AST | None]) -> bool:
    """Comment-only edits leave a .py file's syntax tree unchanged; anything else counts."""
    old, new = trees
    if not path.endswith(".py") or old is None or new is None:
        return True
    return ast.dump(old) != ast.dump(new)


def _runs_at_import(trees: tuple[ast.AST | None, ast.AST | None]) -> bool:
    """Whether import-time statements that can run code differ between the trees.

    Data-only statements cannot import or load anything; function bodies do not
    run at import. A missing tree means the edit could not be read, so it counts.
    """
    old, new = trees
    if old is None or new is None:
        return True
    return _executing_units(old) != _executing_units(new)


def _executing_units(tree: ast.AST) -> Counter[str]:
    units: Counter[str] = Counter()
    _collect(getattr(tree, "body", []), units)
    return units


def _collect(statements: list[ast.stmt], units: Counter[str]) -> None:
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            header = [statement.decorator_list, statement.args.defaults, statement.args.kw_defaults]
            if statement.decorator_list or _executes(header):
                units[f"def {statement.name} {_dump(header)}"] += 1
        elif isinstance(statement, ast.ClassDef):
            # Creating a class runs its metaclass, which can register it.
            header = [statement.bases, statement.keywords, statement.decorator_list]
            units[f"class {statement.name} {_dump(header)}"] += 1
            _collect(statement.body, units)
        elif isinstance(statement, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for field in ("test", "iter", "items"):
                part = getattr(statement, field, None)
                if part is not None and _executes(part):
                    units[f"{type(statement).__name__} {_dump(part)}"] += 1
            for field in ("body", "orelse", "finalbody"):
                _collect(getattr(statement, field, []), units)
            for handler in getattr(statement, "handlers", []):
                _collect(handler.body, units)
        elif _executes(statement):
            units[_dump(statement)] += 1


def _executes(node: object) -> bool:
    nodes = node if isinstance(node, list) else [node]
    for item in nodes:
        for child in _walk(item):
            if isinstance(child, (ast.Import, ast.ImportFrom, ast.Call)):
                return True
            if isinstance(child, ast.Name) and child.id == "dialects":
                return True
            if isinstance(child, ast.Attribute) and child.attr == "dialects":
                return True
    return False


def _walk(node: object) -> list[ast.AST]:
    """Nodes evaluated when `node` runs; lambda bodies wait until they are called."""
    if isinstance(node, list):
        return [child for item in node for child in _walk(item)]
    if not isinstance(node, ast.AST):
        return []
    found: list[ast.AST] = [node]
    for field, value in ast.iter_fields(node):
        if isinstance(node, ast.Lambda) and field == "body":
            continue
        found.extend(_walk(value))
    return found


def _dump(node: object) -> str:
    if isinstance(node, list):
        return "[" + ",".join(_dump(item) for item in node) + "]"
    if isinstance(node, ast.AST):
        return ast.dump(node, include_attributes=False)
    return repr(node)
