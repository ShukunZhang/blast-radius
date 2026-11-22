"""Which fixture cases of one test can read a class attribute an edit rebinds; for case_proof.

class_users proves, from the lines that load an attribute's name, which tests
read it, and gives up (None) when a read is dynamic, happens while classes are
created, goes through a memoized table, or the value is copied into another
class. Given one test's per-case lines instead of per-test coverage, the same
proof yields the cases that read it. A line the test runs outside any case,
at import or with cold caches stands for every case. As class_users does per
test, an attribute of a dialect component class is read only by the cases
that used that class or a subclass (collect_cases records them per case).
An expression class's constant arg_types is read only by the cases
case_args recorded reading it.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from analyze import assigned_names, module_name, statement_at
from class_reads import constant_truth
from class_users import ClassUsers, class_key

Mentions = Callable[[str], "list[tuple[str, int]] | None"]


def keys_of(mask: int) -> set[int]:
    return {key for key in range(mask.bit_length()) if (mask >> key) & 1}


class _CaseLines:
    """One file's lines of one test: line -> the keys of the cases that ran it."""

    def __init__(self, masks: dict[str, str], shared: set[int], every: set[int]) -> None:
        self._masks, self._shared, self._every = masks, shared, every

    def get(self, line: int, default: set[int] | None = None) -> set[int]:
        if line in self._shared:
            return self._every
        mask = self._masks.get(str(line))
        if mask is None:
            return set() if default is None else default
        return keys_of(int(mask, 16))


class CaseCoverage:
    """One test's per-case lines shaped like the selector's coverage: path -> line -> case keys."""

    # Every sqlglot function line the test ran in-process is recorded, so a method missing from it never ran.
    complete = True

    def __init__(self, entry: dict[str, Any], shared: dict[str, set[int]]) -> None:
        self._entry, self._shared = entry, shared
        self._every = set(range(len(entry["case_s"])))

    def get(self, path: str, default: Any = None) -> Any:
        masks = self._entry["lines"].get(path)
        shared = self._shared.get(path)
        if masks is None and shared is None:
            return {} if default is None else default
        return _CaseLines(masks or {}, shared or set(), self._every)

    def users_of(self, key: str) -> set[int]:
        """Keys of the cases that used the dialect component class `key` or a subclass."""
        return keys_of(int(self._entry.get("classes", {}).get(key, "0"), 16))


def attribute_cases(
    users: ClassUsers, mentions: Mentions, test_id: str, coverage: CaseCoverage, path: str, tree: Any, line: int, original: bool
) -> set[int] | None:
    """Keys of the cases that can read what the class-body statement at `line` binds; None for every case."""
    key = class_key(path, tree, line)
    if key is None:
        return None
    statement = statement_at(tree, line)
    names = assigned_names(statement)
    if len(names) != 1 or test_id in (users.test_words or {}).get(names[0], ()):
        return None
    truthy = constant_truth(statement) if original else None
    proof = users._readers(names[0], key, coverage, mentions, truthy, False, False)  # type: ignore[arg-type]
    if proof is None:
        return None
    readers, copiers, _via_dispatch, _derived = proof
    if copiers:
        return None
    if key in users.users or key in users.mro:
        return readers & coverage.users_of(key)
    return readers


def arg_types_owner(path: str, trees: dict[str, ast.Module], side: str, line: int) -> str | None:
    """`module:Class` when `line` of trees[side] lies in the one constant `arg_types = {...}` of a top-level
    class body, which the same class assigns in both trees with no new unpacked source; None otherwise."""
    name = _class_at(trees[side], line)
    if name is None:
        return None
    old, new = _arg_types_of(trees["old"], name), _arg_types_of(trees["new"], name)
    if old is None or new is None or not _in(trees[side], name, line):
        return None
    return f"{module_name(path)}:{name}" if _unpacked(new) <= _unpacked(old) else None


def _class_at(tree: ast.Module, line: int) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.lineno <= line <= (node.end_lineno or node.lineno):
            return node.name
    return None


def _assignments(tree: ast.Module, name: str) -> list[ast.Assign | ast.AnnAssign]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [statement for statement in node.body if assigned_names(statement) == ["arg_types"]]  # type: ignore[misc]
    return []


def _arg_types_of(tree: ast.Module, name: str) -> ast.Dict | None:
    found = _assignments(tree, name)
    if len(found) != 1 or not isinstance(found[0].value, ast.Dict):
        return None
    value = found[0].value
    for key, item in zip(value.keys, value.values):
        if key is None:
            if not (isinstance(item, ast.Attribute) and item.attr == "arg_types" and isinstance(item.value, ast.Name)):
                return None
        elif not (isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(item, ast.Constant)):
            return None
    return value


def _in(tree: ast.Module, name: str, line: int) -> bool:
    return any(statement.lineno <= line <= (statement.end_lineno or statement.lineno) for statement in _assignments(tree, name))


def _unpacked(value: ast.Dict) -> set[str]:
    return {item.value.id for key, item in zip(value.keys, value.values) if key is None}  # type: ignore[attr-defined]
