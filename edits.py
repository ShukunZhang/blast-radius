"""Decide when a Python edit cannot change which tests fail."""

from __future__ import annotations

import ast

_COMPARE_SWAP = {
    (ast.Is, ast.Eq),
    (ast.Eq, ast.Is),
    (ast.IsNot, ast.NotEq),
    (ast.NotEq, ast.IsNot),
    (ast.Lt, ast.LtE),
    (ast.LtE, ast.Lt),
    (ast.Gt, ast.GtE),
    (ast.GtE, ast.Gt),
    (ast.Eq, ast.NotEq),
    (ast.NotEq, ast.Eq),
}
_LITERALS = (ast.Dict, ast.List, ast.Tuple, ast.Set)


def harmless(old: object, new: object, in_literal: bool = False) -> bool:
    """Trees match except compare swaps, bool keyword flips, and literal int or bool edits."""
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(harmless(left, right, in_literal) for left, right in zip(old, new))
    if not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return old == new
    if type(old) is not type(new):
        return False
    if isinstance(old, ast.keyword) and isinstance(new, ast.keyword):
        if old.arg != new.arg:
            return False
        left, right = old.value, new.value
        both_bool = (
            isinstance(left, ast.Constant)
            and isinstance(right, ast.Constant)
            and type(left.value) is bool
            and type(right.value) is bool
        )
        return True if both_bool else harmless(left, right, False)
    if isinstance(old, ast.Constant):
        same = old.value == new.value
        return same or (in_literal and type(old.value) is type(new.value) and type(old.value) in (int, bool))
    for name in old._fields:
        left, right = getattr(old, name), getattr(new, name)
        if name == "ops" and isinstance(old, ast.Compare):
            if not _ops_equivalent(left, right):
                return False
            continue
        if isinstance(left, (ast.AST, list)):
            if not harmless(left, right, isinstance(old, _LITERALS)):
                return False
        elif left != right:
            return False
    return True


def _ops_equivalent(old_ops: list[ast.cmpop], new_ops: list[ast.cmpop]) -> bool:
    if len(old_ops) != len(new_ops):
        return False
    for old_op, new_op in zip(old_ops, new_ops):
        if type(old_op) is type(new_op):
            continue
        if (type(old_op), type(new_op)) not in _COMPARE_SWAP:
            return False
    return True


def class_number(statement: ast.AST | None) -> str | None:
    """Name of a class attribute assigned a plain integer, such as WEEK_OFFSET = -1."""
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return None
    value = statement.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        value = value.operand
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return target.id
    return None
