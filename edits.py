"""Classify a Python edit from the two syntax trees.

A file is skipped only when the only change is `<` versus `<=` on
`dialect.version` compared with a tuple that cannot be equal to it.
`is` versus `==` keeps the tests that executed that line. A data edit keeps
tests that load the value. Any other edit keeps the wider coverage set.
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
# `<` and `<=` differ only when the two sides are equal. `==` and `!=` are not here:
# those flip every value, and a test that executes the line can fail.
_BOUNDARY = {
    (ast.Lt, ast.LtE),
    (ast.LtE, ast.Lt),
    (ast.Gt, ast.GtE),
    (ast.GtE, ast.Gt),
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


def boundary_unreachable(old: object, new: object) -> bool:
    """True when every change is `<`/`<=` or `>`/`>=` on `dialect.version`.

    `Dialect.__init__` is the only write of that attribute, and it always stores
    a 3-tuple. A tuple of another length is never equal to it, so the two
    operators take the same branch for every version the constructor can build.
    A same-length tuple, a bare integer, or `==`/`!=` can still change a test.
    """
    compares: list[ast.Compare] = []
    if not _boundary(old, new, compares):
        return False
    return bool(compares) and all(_dialect_version_mismatch(node) for node in compares)


def _boundary(old: object, new: object, found: list[ast.Compare]) -> bool:
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(_boundary(left, right, found) for left, right in zip(old, new))
    if isinstance(old, ast.cmpop) and isinstance(new, ast.cmpop):
        return type(old) is type(new) or (type(old), type(new)) in _BOUNDARY
    if type(old) is not type(new):
        return False
    if not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return old == new
    if isinstance(old, ast.Compare) and any(type(left) is not type(right) for left, right in zip(old.ops, new.ops)):
        found.append(old)
    return all(_boundary(getattr(old, name), getattr(new, name), found) for name in old._fields)


def _dialect_version_mismatch(node: ast.Compare) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    sides = (node.left, node.comparators[0])
    version = next((side for side in sides if _is_dialect_version(side)), None)
    literal = next((side for side in sides if isinstance(side, ast.Tuple)), None)
    if version is None or literal is None:
        return False
    if not literal.elts or any(not isinstance(elt, ast.Constant) or type(elt.value) is not int for elt in literal.elts):
        return False
    return len(literal.elts) != 3


def _is_dialect_version(node: ast.AST) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != "version":
        return False
    owner = node.value
    if isinstance(owner, ast.Name):
        return owner.id == "dialect"
    return isinstance(owner, ast.Attribute) and owner.attr == "dialect"


def enum_names(root: Path) -> set[str]:
    """Class names that extend Enum, including mixins such as `class E(str, Enum)`."""
    classes: list[tuple[str, list[str]]] = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            classes.append((node.name, bases))
    known = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "AutoName"}
    changed = True
    while changed:
        changed = False
        for name, bases in classes:
            if name not in known and any(base in known for base in bases):
                known.add(name)
                changed = True
    return known


def neutral_identity(old: object, new: object, enums: set[str]) -> bool:
    """`is`/`==` swaps that stay equivalent for every value this tree can hold.

    `self.attr is None` matches `== None`: none of the pinned classes treat None
    as equal to an instance. `self.attr is EnumClass.MEMBER` matches `==` because
    that attribute is built as a member, and members compare by identity.
    A bare name or a non-enum class is left alone: a later commit can store a
    string or an object with its own `__eq__` there.
    """
    found: list[bool] = []
    if not _neutral(old, new, found, enums):
        return False
    return bool(found) and all(found)


def _neutral(old: object, new: object, found: list[bool], enums: set[str]) -> bool:
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(
            _neutral(left, right, found, enums) for left, right in zip(old, new)
        )
    if isinstance(old, ast.cmpop) and isinstance(new, ast.cmpop):
        return type(old) is type(new) or (type(old), type(new)) in _IDENTITY
    if type(old) is not type(new):
        return False
    if not isinstance(old, ast.AST) or not isinstance(new, ast.AST):
        return old == new
    if isinstance(old, ast.Compare) and any(
        type(left) is not type(right) for left, right in zip(old.ops, new.ops)
    ):
        found.append(_none_or_enum_member(old, enums))
    return all(_neutral(getattr(old, name), getattr(new, name), found, enums) for name in old._fields)


def _none_or_enum_member(node: ast.Compare, enums: set[str]) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
        return False
    sides = (node.left, node.comparators[0])
    if any(isinstance(side, ast.Constant) and side.value is None for side in sides):
        other = next(side for side in sides if not (isinstance(side, ast.Constant) and side.value is None))
        return isinstance(other, ast.Attribute)
    member = next((side for side in sides if _enum_member(side, enums)), None)
    if member is None:
        return False
    other = sides[0] if member is sides[1] else sides[1]
    return isinstance(other, ast.Attribute)


def _enum_member(node: ast.AST, enums: set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in enums
        and node.attr.isupper()
    )


def changed_dict_key(tree: ast.AST, line: int) -> str | None:
    """String key of the dict entry that contains `line`, when the key is specific."""
    best: str | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        end = node.end_lineno or node.lineno
        if not node.lineno <= line <= end:
            continue
        for key, value in zip(node.keys, node.values):
            if key is None or value is None:
                continue
            value_end = value.end_lineno or value.lineno
            if not value.lineno <= line <= value_end:
                continue
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if len(key.value) < 8:
                continue
            span = end - node.lineno
            if span < best_span:
                best = key.value
                best_span = span
    return best


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


def loaded_mentions(
    name: str,
    mentions: Callable[[str], list[tuple[str, int]] | None],
    coverage: dict[str, dict[int, set[int]]],
    root: Path,
    path: str,
    tree: ast.AST,
    line: int,
) -> set[int] | None:
    """Tests that load `name`, or that execute the changed assignment.

    A test that only runs another class's `NAME = ...` does not read this
    value. A line we cannot classify as a store is kept: the old rule missed
    commits by dropping dialect helpers that load the name. None means the
    mention list itself is missing.
    """
    found = mentions(name)
    if not found:
        return None
    statement = statement_at(tree, line)
    start = statement.lineno if statement is not None else line
    end = statement.end_lineno if statement is not None and statement.end_lineno else line
    trees: dict[str, ast.AST | None] = {}
    readers: set[int] = set()
    traced = False
    for relative, lineno in found:
        if relative not in trees:
            try:
                trees[relative] = ast.parse((root / relative).read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                trees[relative] = None
        parsed = trees[relative]
        in_change = relative == path and start <= lineno <= end
        if parsed is not None and _name_use(parsed, lineno, name) == "store" and not in_change:
            continue
        owners = coverage.get(relative, {}).get(lineno, set())
        if owners:
            traced = True
        readers |= owners
    return readers if traced else None


def _name_use(tree: ast.AST, lineno: int, name: str) -> str | None:
    """`load`, `store`, or None when this line does not name `name` in the AST."""
    loaded = False
    stored = False
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Store):
                stored = True
            else:
                loaded = True
        elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(getattr(node, "ctx", None), ast.Load):
            loaded = True
    if loaded:
        return "load"
    if stored:
        return "store"
    return None


def proven_readers(
    name: str,
    class_name: str | None,
    mentions: Callable[[str], list[tuple[str, int]] | None],
    coverage: dict[str, dict[int, set[int]]],
    root: Path,
    hot: int,
) -> set[int] | None:
    """Readers of the changed value. None when the read sites are not proven."""
    if mentions(name) is not None:
        sinks = component_sinks(root, name, mentions)
        if not sinks:
            return None
        readers: set[int] = set()
        traced = False
        for relative, lineno in sinks:
            owners = coverage.get(relative, {}).get(lineno, set())
            if owners:
                traced = True
            readers |= owners
        return readers if traced else None
    if class_name is None or mentions(class_name) is None:
        return None
    readers = set()
    for relative, lineno in mentions(class_name) or []:
        owners = coverage.get(relative, {}).get(lineno, set())
        if len(owners) > hot:
            return None
        readers |= owners
    return readers or None


def _condition_field(node: ast.AST, name: str) -> bool:
    if name == "test" and isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp)):
        return True
    return name == "ifs" and isinstance(node, ast.comprehension)
