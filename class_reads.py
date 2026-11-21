"""AST checks on the places that read a class attribute, used by `class_users`.

Each answers whether a read can be traced to the tests that executed a line,
or whether it reaches tests some other way (class creation, a module-level
cache, dynamic lookup), in which case the caller falls back to class users.
"""

from __future__ import annotations

import ast

from analyze import functions_and_classes, statement_at, tightest

Coverage = dict[str, dict[int, set[int]]]
_LINES: dict[ast.AST, dict[int, list[ast.AST]]] = {}
_SCOPES: dict[ast.AST, tuple[list[ast.AST], list[ast.AST]]] = {}
_CACHES: dict[ast.AST, tuple[dict[str, list[ast.AST]], set[ast.AST]]] = {}


def nodes_at(tree: ast.AST, lineno: int) -> list[ast.AST]:
    if tree not in _LINES:
        index: dict[int, list[ast.AST]] = {}
        for node in ast.walk(tree):
            line = getattr(node, "lineno", None)
            if line is not None:
                index.setdefault(line, []).append(node)
        _LINES[tree] = index
    return _LINES[tree].get(lineno, [])


def scopes(tree: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """`functions_and_classes`, computed once per tree."""
    if tree not in _SCOPES:
        _SCOPES[tree] = functions_and_classes(tree)
    return _SCOPES[tree]


def statement_on(tree: ast.AST, lineno: int) -> ast.AST | None:
    """Innermost statement holding a node on `lineno`."""
    parents = _parents(tree)
    for node in nodes_at(tree, lineno):
        while node is not None and not isinstance(node, ast.stmt):
            node = parents.get(node)
        if node is not None:
            return node
    return statement_at(tree, lineno)


def name_use(tree: ast.AST, lineno: int, name: str) -> str | None:
    """`load`, `store`, or None when this line does not name `name` (as `edits._name_use`)."""
    loaded = stored = False
    for node in nodes_at(tree, lineno):
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Store):
                stored = True
            else:
                loaded = True
        elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(getattr(node, "ctx", None), ast.Load):
            loaded = True
    return "load" if loaded else "store" if stored else None
# Code here runs while classes are created, so what it reads reaches tests
# through derived tables rather than through lines that name the attribute.
_CLASS_CREATION = {"__new__", "__init_subclass__", "__set_name__", "__class_getitem__"}


def at_class_creation(tree: ast.AST, function: ast.AST) -> bool:
    if function.name in _CLASS_CREATION:  # type: ignore[attr-defined]
        return True
    _functions, classes = scopes(tree)
    owner = tightest(classes, function.lineno)  # type: ignore[attr-defined]
    if owner is None:
        return False
    return any(isinstance(base, ast.Name) and base.id == "type" for base in owner.bases)  # type: ignore[attr-defined]


def dynamic_access(tree: ast.AST, lineno: int, name: str) -> bool:
    """`getattr(obj, "NAME")` and friends, which read the attribute without naming it in code."""
    for node in nodes_at(tree, lineno):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if called in {"getattr", "hasattr", "setattr", "delattr", "attrgetter"} and any(
            isinstance(arg, ast.Constant) and arg.value == name for arg in node.args
        ):
            return True
    return False


def in_lambda_body(tree: ast.AST, lineno: int) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            body = node.body
            if body.lineno <= lineno <= (body.end_lineno or body.lineno):
                return True
    return False


def in_place(statement: ast.AST | None, name: str) -> bool:
    """A statement that only mutates `x.NAME` itself, leaving later reads under the same name."""
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        func = statement.value.func
        return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute) and func.value.attr == name
    if isinstance(statement, (ast.Assign, ast.AugAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            base = target.value if isinstance(target, ast.Subscript) else target
            if not (isinstance(base, ast.Attribute) and base.attr == name):
                return False
        return True
    return False


def copied_owners(tree: ast.AST, lineno: int, name: str) -> list[str | None]:
    """For each load of `name` on `lineno`, the class name it is read from (`X` in `pkg.X.NAME`).

    None for a bare `NAME` or an owner that is not a plain name.
    """
    owners: list[str | None] = []
    for node in nodes_at(tree, lineno):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            owners.append(None)
        elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
            owner = node.value
            owners.append(owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None))
    return owners


_KEY_CALLS = {"new_trie", "set", "frozenset", "sorted", "len", "list", "tuple"}


def keys_only(tree: ast.AST, lineno: int, name: str) -> bool:
    """Whether every load of `name` on `lineno` reads only the keys of the table.

    Iterating it, testing membership, or passing it to `new_trie`/`set`/`len`
    and the like derive nothing from values. A plain alias (`x = cls.NAME`)
    qualifies when every use of the alias in that scope does.
    """
    parents = _parents(tree)
    loads = [
        node
        for node in nodes_at(tree, lineno)
        if isinstance(getattr(node, "ctx", None), ast.Load)
        and ((isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name))
    ]
    if not loads:
        return False
    for node in loads:
        parent = parents.get(node)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            scope = _scope(parents, parent)
            alias = parent.targets[0].id
            uses = [
                use
                for use in ast.walk(scope)
                if isinstance(use, ast.Name) and use.id == alias and use is not parent.targets[0]
            ]
            if not all(isinstance(use.ctx, ast.Load) and _key_context(use, parents) for use in uses):
                return False
        elif not _key_context(node, parents):
            return False
    return True


def _key_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, (ast.comprehension, ast.For)):
        return parent.iter is node
    if isinstance(parent, ast.Compare):
        return any(
            comparator is node and isinstance(op, (ast.In, ast.NotIn))
            for op, comparator in zip(parent.ops, parent.comparators)
        )
    if isinstance(parent, ast.Call) and node in parent.args:
        func = parent.func
        return (func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)) in _KEY_CALLS
    return False


def _scope(parents: dict[ast.AST, ast.AST], node: ast.AST) -> ast.AST:
    while node in parents and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        node = parents[node]
    return node


_PARENTS: dict[ast.AST, dict[ast.AST, ast.AST]] = {}


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    if tree not in _PARENTS:
        _PARENTS[tree] = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    return _PARENTS[tree]


def constant_truth(statement: ast.AST | None) -> bool | None:
    value = getattr(statement, "value", None)
    if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Constant):
        return bool(value.value)
    return None


def dict_entry_key(statement: ast.AST | None, line: int) -> str | None:
    """Key of the class-level dict literal entry that spans `line`, named as `table_tracking.label`
    records it: the string, the class name (`exp.ArrayAgg`), or the enum member (`TokenType.SELECT`).
    """
    value = getattr(statement, "value", None)
    if not isinstance(value, ast.Dict):
        return None
    for key, item in zip(value.keys, value.values):
        if key is not None and key.lineno <= line <= (item.end_lineno or item.lineno):
            return key_label(key)
    return None


def key_label(key: ast.AST) -> str | None:
    if isinstance(key, ast.Constant):
        return key.value if isinstance(key.value, str) else repr(key.value)
    if isinstance(key, ast.Attribute):
        return key.attr
    return key.id if isinstance(key, ast.Name) else None


def changed_dict_keys(old_tree: ast.AST, new_tree: ast.AST) -> set[str] | None:
    """Old and new labels of the dict keys an edit renamed; None when entries were added,
    removed, or a renamed key has no label.
    """
    old = [node for node in ast.walk(old_tree) if isinstance(node, ast.Dict)]
    new = [node for node in ast.walk(new_tree) if isinstance(node, ast.Dict)]
    if len(old) != len(new):
        return None
    changed: set[str] = set()
    for before, after in zip(old, new):
        if len(before.keys) != len(after.keys):
            return None
        for was, now in zip(before.keys, after.keys):
            if (was is None) != (now is None):
                return None
            if was is None or now is None or ast.dump(was) == ast.dump(now):
                continue
            labels = {key_label(was), key_label(now)}
            if None in labels:
                return None
            changed |= {label for label in labels if label is not None}
    return changed


def chain_tail(tree: ast.AST, lineno: int, name: str, truthy: bool) -> int | None:
    """First line of the last operand after the only `x.NAME` on `lineno`, when that read sits in
    an `and` chain (truthy value) or an `or` chain (falsy value) and the operand starts on a later line.

    Such a read only lets the chain go on, so changing it matters only where the last operand ran.
    """
    reads = [node for node in nodes_at(tree, lineno) if isinstance(node, ast.Attribute) and node.attr == name]
    if len(reads) != 1:
        return None
    node = _parents(tree).get(reads[0])
    wanted = ast.And if truthy else ast.Or
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, wanted) or node.values[-1] is reads[0]:
        return None
    before = node.values[-2]
    tail = node.values[-1]
    if tail.lineno > (before.end_lineno or before.lineno):
        return tail.lineno
    return None


def memoized(tree: ast.AST, function: ast.AST) -> bool:
    """Whether what `function` computes may be kept in a module-level cache and reused by later tests."""
    name = function.name  # type: ignore[attr-defined]
    if any("cache" in ast.unparse(decorator).lower() for decorator in function.decorator_list):  # type: ignore[attr-defined]
        return True
    callers, storing = _cache_writers(tree)
    return function in storing or any(caller in storing for caller in callers.get(name, ()))


def _cache_writers(tree: ast.AST) -> tuple[dict[str, list[ast.AST]], set[ast.AST]]:
    """Functions by the names they call, and the functions that store into a module-level name."""
    if tree not in _CACHES:
        module_names = {
            target.id
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        callers: dict[str, list[ast.AST]] = {}
        storing: set[ast.AST] = set()
        for function in scopes(tree)[0]:
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    if isinstance(called, str):
                        callers.setdefault(called, []).append(function)
            if _stores_into(function, module_names):
                storing.add(function)
        _CACHES[tree] = (callers, storing)
    return _CACHES[tree]


def _stores_into(function: ast.AST, module_names: set[str]) -> bool:
    """`CACHE[key] = ...` or `CACHE.setdefault(...)`/`.update(...)` on a module-level name."""
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id in module_names:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if node.func.attr in {"setdefault", "update"} and isinstance(owner, ast.Name) and owner.id in module_names:
                return True
    return False


def ever_executed(coverage: Coverage, relative: str, function: ast.AST) -> bool:
    """Whether some test ran a line of `function`; a never-run function may run only at import."""
    by_line = coverage.get(relative, {})
    start = function.lineno  # type: ignore[attr-defined]
    end = function.end_lineno or start  # type: ignore[attr-defined]
    return any(by_line.get(lineno) for lineno in range(start + 1, end + 1))
