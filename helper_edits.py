"""Follow an edit to shared code in a test module into the test files that use it.

`tests/dialects/test_dialect.py` defines `Validator`, which every dialect test
file subclasses, and `tests/helpers.py` loads fixtures for several suites. An
edit there changes tests in other files, not only the file that was edited.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from analyze import _has_identifier


def changed_helper_names(old_tree: ast.AST, new_tree: ast.AST) -> set[str] | None:
    """Top-level names whose definition changed, ignoring test-method bodies.

    None when a changed top-level statement binds no name we can follow.
    """
    old = _top_level(old_tree)
    new = _top_level(new_tree)
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if any(name.startswith("<") for name in changed):
        return None
    return changed


def dependent_files(
    root: Path, edited: str, candidates: set[str], names: set[str] | None
) -> set[str]:
    """Test files among `candidates` that use a changed name, directly or by inheritance.

    A file that uses a changed name passes the change on through every class it
    defines, so names grow until no further file matches. None selects them all.
    """
    pending = set(candidates) - {edited}
    if names is None:
        return pending
    reach = set(names)
    texts = {relative: _read(root / relative) for relative in pending}
    selected: set[str] = set()
    grew = True
    while grew:
        grew = False
        for relative in sorted(pending - selected):
            text = texts[relative]
            if any(_has_identifier(text, name) for name in reach):
                selected.add(relative)
                reach |= _defined_names(text)
                grew = True
    return selected


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _defined_names(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body  # type: ignore[attr-defined]
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level(tree: ast.AST) -> dict[str, str]:
    """Name -> definition. A statement that binds no followable name keys itself as `<...>`."""
    found: dict[str, str] = {}
    for index, node in enumerate(tree.body):  # type: ignore[attr-defined]
        dump = ast.dump(node, include_attributes=False)
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            dump = ast.dump(_without_test_bodies(node), include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if len(names) != len(targets):
                names = []
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        elif index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            names = ["__doc__"]
        for name in names or [f"<{dump}>"]:
            found[name] = dump
    return found


def _without_test_bodies(node: ast.AST) -> ast.AST:
    clone = copy.deepcopy(node)
    for child in ast.walk(clone):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
            child.body = [ast.Pass()]
    return clone
