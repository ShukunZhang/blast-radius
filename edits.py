"""Classify a Python edit from the two syntax trees.

A file is skipped when the trees match, or the only change is `is` versus `==`. A data edit changes a value and
leaves control flow alone, so only tests that executed the edited line can
observe it. Any other edit keeps the wider coverage set.
"""

from __future__ import annotations

import ast


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


def _condition_field(node: ast.AST, name: str) -> bool:
    if name == "test" and isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp)):
        return True
    return name == "ifs" and isinstance(node, ast.comprehension)
