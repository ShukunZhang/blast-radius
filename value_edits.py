"""Comparison edits that answer every recorded evaluation the same way.

`collect_values.py` ran the whole suite on the pinned tree and kept, for each single-operator comparison,
the operators that agreed with the original at every evaluation and, when one side is a literal, every
value the other side took. A patch whose only changes are such comparisons, each swapped to an agreeing
operator or given a literal that every recorded value compares the same way against, leaves every
evaluation unchanged. The patched run then repeats the pinned run step for step, so no test can fail
from it. A comparison the suite never evaluated, one that raised, or one whose values are not all
recorded keeps the normal selection.
"""

from __future__ import annotations

import ast
import json
import operator
import os
from pathlib import Path

VALUES_MAP = Path(os.environ.get("VALUES_MAP", Path(__file__).resolve().parent / "maps" / "values.json"))
_OPS = {
    ast.Eq: ("Eq", operator.eq),
    ast.NotEq: ("NotEq", operator.ne),
    ast.Lt: ("Lt", operator.lt),
    ast.LtE: ("LtE", operator.le),
    ast.Gt: ("Gt", operator.gt),
    ast.GtE: ("GtE", operator.ge),
    ast.Is: ("Is", operator.is_),
    ast.IsNot: ("IsNot", operator.is_not),
    ast.In: ("In", lambda left, right: left in right),
    ast.NotIn: ("NotIn", lambda left, right: left not in right),
}
_SINGLETONS = (None, True, False)


class ValueProfile:
    def __init__(self, path: Path = VALUES_MAP) -> None:
        self.path = path
        self._sites: dict[str, dict] | None = None

    @property
    def sites(self) -> dict[str, dict]:
        if self._sites is None:
            try:
                self._sites = json.loads(self.path.read_text(encoding="utf-8")).get("sites", {})
            except (OSError, ValueError):
                self._sites = {}
        return self._sites

    def neutral(self, relative: str, old_tree: ast.AST, new_tree: ast.AST) -> bool:
        """Every difference between the trees is a comparison edit no recorded evaluation can see."""
        pairs: list[tuple[ast.Compare, ast.Compare]] = []
        if not _walk(old_tree, new_tree, pairs) or not pairs or not self.sites:
            return False
        return all(self._unseen(relative, old, new) for old, new in pairs)

    def _unseen(self, relative: str, old: ast.Compare, new: ast.Compare) -> bool:
        entry = self.sites.get(f"{relative}:{old.lineno}:{old.col_offset}")
        if entry is None or entry["raised"]:
            return False
        old_name, old_op = _OPS[type(old.ops[0])]
        new_name, new_op = _OPS[type(new.ops[0])]
        side = _literal_side(old)
        if side == 0 or _dump(_side(old, side)) == _dump(_side(new, side)):
            return new_name in entry["ops"]
        values = entry["values"]
        if not isinstance(values, list):
            return False
        old_literal = ast.literal_eval(_side(old, side))
        new_literal = ast.literal_eval(_side(new, side))
        for name, literal in ((old_name, old_literal), (new_name, new_literal)):
            if name in ("Is", "IsNot") and not any(literal is item for item in _SINGLETONS):
                return False
        for raw in values:
            value = _value(json.loads(raw))
            if value is _OPAQUE:
                return False
            try:
                before = _apply(old_op, value, old_literal, side)
                after = _apply(new_op, value, new_literal, side)
            except Exception:  # noqa: BLE001 - a comparison that raises here is not provably unchanged
                return False
            if type(before) is not type(after) or before != after:
                return False
        return True


_OPAQUE = object()


def _value(canonical: object) -> object:
    """The recorded value, a fresh identity-only object for "o", or _OPAQUE."""
    if canonical == "o":
        return object()
    if isinstance(canonical, list) and len(canonical) == 2 and canonical[0] == "v":
        return _tuples(canonical[1])
    return _OPAQUE


def _tuples(value: object) -> object:
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _apply(function, value: object, literal: object, side: int) -> object:  # type: ignore[no-untyped-def]
    return function(value, literal) if side == 2 else function(literal, value)


def _literal_display(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(isinstance(e, ast.Constant) for e in node.elts)


def _literal_side(node: ast.Compare) -> int:
    """2 when only the right side is a literal, 1 when only the left is, else 0."""
    left, right = _literal_display(node.left), _literal_display(node.comparators[0])
    return 2 if right and not left else 1 if left and not right else 0


def _side(node: ast.Compare, side: int) -> ast.AST:
    return node.comparators[0] if side == 2 else node.left


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _walk(old: object, new: object, pairs: list[tuple[ast.Compare, ast.Compare]]) -> bool:
    """Whether the trees differ only in comparison edits; collects those comparisons."""
    if type(old) is not type(new):
        return False
    if isinstance(old, list):
        return len(old) == len(new) and all(_walk(a, b, pairs) for a, b in zip(old, new))  # type: ignore[arg-type]
    if not isinstance(old, ast.AST):
        return old == new
    if isinstance(old, ast.Compare) and _dump(old) != _dump(new):  # type: ignore[arg-type]
        found: list[tuple[ast.Compare, ast.Compare]] = []
        if all(_walk(getattr(old, f), getattr(new, f), found) for f in old._fields):
            pairs.extend(found)
            return True
        if _compare_edit(old, new):  # type: ignore[arg-type]
            pairs.append((old, new))  # type: ignore[arg-type]
            return True
        return False
    return all(_walk(getattr(old, field), getattr(new, field), pairs) for field in old._fields)


def _compare_edit(old: ast.Compare, new: ast.Compare) -> bool:
    """One operator, the same non-literal side, and either the operator or the literal changed."""
    if len(old.ops) != 1 or len(new.ops) != 1:
        return False
    if type(old.ops[0]) not in _OPS or type(new.ops[0]) not in _OPS:
        return False
    side = _literal_side(old)
    if side == 0:
        return _dump(old.left) == _dump(new.left) and _dump(old.comparators[0]) == _dump(new.comparators[0])
    if _literal_side(new) != side:
        return False
    other = 1 if side == 2 else 2
    return _dump(_side(old, other)) == _dump(_side(new, other))
