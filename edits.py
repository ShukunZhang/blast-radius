"""Classify a Python edit from the two syntax trees.

A file is skipped when the trees match, or the only change is `is` versus `==`. A data edit changes a value and
leaves control flow alone, so only tests that executed the edited line can
observe it. Any other edit keeps the wider coverage set.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from analyze import statement_at


# Identity and equality agree for the comparisons these swaps replace.
_IDENTITY = {
    (ast.Is, ast.Eq),
    (ast.Eq, ast.Is),
    (ast.IsNot, ast.NotEq),
    (ast.NotEq, ast.IsNot),
}


def identity_swap(old: object, new: object) -> bool:
    """Trees match, allowing only `is`/`==` and `is not`/`!=`."""
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(identity_swap(left, right) for left, right in zip(old, new))
    if isinstance(old, ast.cmpop) and isinstance(new, ast.cmpop):
        return type(old) is type(new) or (type(old), type(new)) in _IDENTITY
    if type(old) is not type(new):
        return False
    if not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return old == new
    return all(identity_swap(getattr(old, name), getattr(new, name)) for name in old._fields)


def data_only(old: object, new: object) -> bool:
    """True when the only differences are constants that are not conditions.

    A test that never executed that constant in the old file still cannot
    observe it, because the edit does not change which branch runs.
    """
    return _data(old, new, conditional=False)


def _data(old: object, new: object, conditional: bool) -> bool:
    if type(old) is not type(new):
        return False
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(_data(left, right, conditional) for left, right in zip(old, new))
    if not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return old == new
    if isinstance(old, ast.Constant):
        return old.value == new.value or not conditional
    for name in old._fields:
        left, right = getattr(old, name), getattr(new, name)
        child = conditional or _condition_field(old, name)
        if isinstance(left, (ast.AST, list)):
            if not _data(left, right, child):
                return False
        elif left != right:
            return False
    return True


def component_sinks(
    root: Path,
    name: str,
    mentions: Callable[[str], list[tuple[str, int]] | None],
) -> list[tuple[str, int]] | None:
    """Lines that index a value which flowed out of `name`.

    Membership and truthiness checks do not read a constant stored inside the
    value. None means the flow was too wide to follow.
    """
    found = mentions(name)
    if not found:
        return None
    trees: dict[str, ast.AST | None] = {}

    def parsed(relative: str) -> ast.AST | None:
        if relative not in trees:
            try:
                trees[relative] = ast.parse((root / relative).read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                trees[relative] = None
        return trees[relative]

    aliases: set[str] = set()
    for relative, lineno in found:
        tree = parsed(relative)
        statement = statement_at(tree, lineno) if tree is not None else None
        if statement is not None:
            aliases |= _value_aliases(statement, name)
    aliases.discard(name)
    if not aliases:
        return None
    sinks: list[tuple[str, int]] = []
    for alias in aliases:
        alias_lines = mentions(alias)
        if not alias_lines:
            return None
        for relative, lineno in alias_lines:
            tree = parsed(relative)
            if tree is not None and _reads_component(tree, lineno, alias):
                sinks.append((relative, lineno))
    return sinks or None


def _contains_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


def _value_aliases(statement: ast.AST, name: str) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and _contains_name(keyword.value, name):
                    aliases.add(keyword.arg)
        elif isinstance(node, ast.Assign) and _contains_name(node.value, name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)
                elif isinstance(target, ast.Attribute):
                    aliases.add(target.attr)
    return aliases


def _reads_component(tree: ast.AST, lineno: int, alias: str) -> bool:
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        if isinstance(node, ast.Subscript) and _expr_has_alias(node.value, alias):
            return True
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"get", "items", "values"} and _expr_has_alias(node.func.value, alias):
            return True
    return False


def _expr_has_alias(node: ast.AST, alias: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == alias
    if isinstance(node, ast.Attribute):
        return node.attr == alias or _expr_has_alias(node.value, alias)
    return False


def _condition_field(node: ast.AST, name: str) -> bool:
    if name == "test" and isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp)):
        return True
    return name == "ifs" and isinstance(node, ast.comprehension)
