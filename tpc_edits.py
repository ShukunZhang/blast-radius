"""Where a patch's edits to one sqlglot file can take effect, for tpc_proof.

An edit inside the body of a function that exists on the pinned commit takes
effect only where an old statement around it runs: a modified or removed
statement, or for an insertion the smallest old statement holding both of
its neighbours, widened to the enclosing block when that statement compiles
to no code. Those statements' line spans are returned. Any other edit
(module or class level, a signature, a new function) is judged by the names
it binds: PLAN_ALL when none is read by the code a TPC test runs after
planning, else FULL.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from diffutil import Hunk

PLAN_ALL, FULL = "plan", "full"
Function = ast.FunctionDef | ast.AsyncFunctionDef
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Reach:
    spans: list[tuple[int, int]] = field(default_factory=list)
    level: str | None = None
    needs_probe: bool = False
    # The edited lines judged by name, each with the tree it is in: "old" (pinned) or "new" (patched).
    outside: list[tuple[str, int]] = field(default_factory=list)

    def escalate(self, harmless: bool) -> None:
        self.level = FULL if not harmless or self.level == FULL else PLAN_ALL


def file_reach(old_tree: ast.Module, new_tree: ast.Module, hunks: list[Hunk], names: set[str]) -> Reach:
    reach = Reach()
    old_functions = _functions(old_tree)
    for removed, added, before, after in _blocks(hunks):
        anchor = _anchor(old_tree, removed or [before, after], insertion=not removed)
        for line in removed:
            inside = _in_body(_chain(old_tree, line), line, old_functions)
            if inside is not None:
                reach.spans.append(_span(_widened(_chain(old_tree, line), inside[1])))
            elif _outside_body(reach, _chain(old_tree, line), line, old_functions, names):
                reach.outside.append(("old", line))
        for line in added:
            chain = _chain(new_tree, line)
            inside = _in_body(chain, line, old_functions)
            if inside is None:
                if _outside_body(reach, chain, line, old_functions, names):
                    reach.outside.append(("new", line))
            elif anchor is not None and anchor[2] == inside[0] and chain[-1].col_offset >= anchor[1]:
                reach.spans.append(anchor[0])
            else:
                reach.spans.extend(_body_span(function) for function in old_functions[inside[0]])
    return reach


def _blocks(hunks: list[Hunk]) -> list[tuple[list[int], list[int], int, int]]:
    """Runs of changed code lines: old lines removed, new lines added, and the old lines just before and after."""
    blocks: list[tuple[list[int], list[int], int, int]] = []
    for hunk in hunks:
        old, new = hunk.old_start, hunk.new_start
        removed: list[int] = []
        added: list[int] = []
        start = old
        for kind, text in [*hunk.lines, (" ", "")]:
            if kind == " ":
                if removed or added:
                    blocks.append((removed, added, start - 1, old))
                removed, added = [], []
                old, new = old + 1, new + 1
                start = old
            elif kind == "-":
                if _code(text):
                    removed.append(old)
                old += 1
            elif kind == "+":
                if _code(text):
                    added.append(new)
                new += 1
    return [block for block in blocks if block[0] or block[1]]


def _code(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("#")


def _start(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _span(node: ast.stmt) -> tuple[int, int]:
    return (_start(node), node.end_lineno or node.lineno)


def _body_span(function: Function) -> tuple[int, int]:
    return (function.body[0].lineno, function.end_lineno or function.lineno)


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
    return ".".join(node.name for node in chain if isinstance(node, (*_FUNCTIONS, ast.ClassDef)))


def _functions(tree: ast.Module) -> dict[str, list[Function]]:
    """Pinned functions by qualified name."""
    found: dict[str, list[Function]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if isinstance(child, _FUNCTIONS):
                    found.setdefault(name, []).append(child)
                visit(child, name + ".")
            elif isinstance(child, ast.stmt):
                visit(child, prefix)

    visit(tree, "")
    return found


def _in_body(chain: list[ast.stmt], line: int, old_functions: dict[str, list[Function]]) -> tuple[str, int] | None:
    """The innermost pinned function whose body holds line, and its index in chain; None outside any."""
    for index in range(len(chain) - 1, -1, -1):
        node = chain[index]
        if isinstance(node, _FUNCTIONS) and _qualname(chain[: index + 1]) in old_functions:
            if index == len(chain) - 1 or line < node.body[0].lineno:
                return None
            return (_qualname(chain[: index + 1]), index)
    return None


def _no_code(node: ast.stmt) -> bool:
    """Statements that compile to nothing, so never report running."""
    if isinstance(node, (ast.Pass, ast.Global, ast.Nonlocal)):
        return True
    if isinstance(node, ast.AnnAssign) and node.value is None:
        return True
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def _widened(chain: list[ast.stmt], function_index: int) -> ast.stmt:
    """The innermost statement of chain below the function, or its block when it compiles to nothing."""
    index = len(chain) - 1
    while index > function_index + 1 and _no_code(chain[index]):
        index -= 1
    if _no_code(chain[index]):
        function = chain[function_index]
        return ast.If(test=ast.Constant(True), body=function.body, orelse=[], lineno=function.body[0].lineno, end_lineno=function.end_lineno)
    return chain[index]


def _anchor(old_tree: ast.Module, lines: list[int], insertion: bool) -> tuple[tuple[int, int], int, str] | None:
    """For a block inside one pinned function body: the old span that must run for it to matter, the least
    indentation new statements may have to stay inside that span, and the function's qualified name."""
    chains = [_chain(old_tree, line) for line in lines]
    common: list[ast.stmt] = []
    for nodes in zip(*chains):
        if any(node is not nodes[0] for node in nodes):
            break
        common.append(nodes[0])
    functions = [index for index, node in enumerate(common) if isinstance(node, _FUNCTIONS)]
    if not functions:
        return None
    index = functions[-1]
    function = common[index]
    name = _qualname(common[: index + 1])
    if index == len(common) - 1:
        if insertion and min(lines) < function.body[0].lineno - 1:
            return None
        return (_body_span(function), function.body[0].col_offset, name)
    statement = _widened(common, index)
    return (_span(statement), getattr(statement, "col_offset", function.body[0].col_offset), name)


def _outside_body(reach: Reach, chain: list[ast.stmt], line: int, old_functions: dict[str, list[Function]], names: set[str]) -> bool:
    """Judge an edit outside pinned function bodies by the names its statement binds; True when it escalated."""
    reach.needs_probe = True
    function_indexes = [index for index, node in enumerate(chain) if isinstance(node, _FUNCTIONS)]
    existing = [index for index in function_indexes if _qualname(chain[: index + 1]) in old_functions]
    added = [index for index in function_indexes if index not in existing]
    if added:
        statement: ast.stmt = chain[added[0]]
    elif function_indexes:
        statement = chain[function_indexes[-1]]
        if line >= statement.lineno and not any(decorator.lineno <= line <= (decorator.end_lineno or 0) for decorator in statement.decorator_list):
            reach.spans.extend(_span(function) for function in old_functions[_qualname(chain[: function_indexes[-1] + 1])])
            return False
    elif chain:
        statement = chain[-1]
    else:
        return False
    if _no_code(statement) and not isinstance(statement, ast.AnnAssign):
        return False
    bound = _bound(statement)
    reach.escalate(bool(bound) and not any(name in names or name.startswith("__") for name in bound))
    return True


def _bound(node: ast.stmt) -> set[str]:
    """Names a statement binds in its scope, by attribute or subscript base too."""
    if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
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
