"""Static facts about the pinned sqlglot tree: fixtures, imports, and scopes.

Built from source at runtime. Nothing here is derived from sample answers.
"""

from __future__ import annotations

import ast
from pathlib import Path


def python_code(line: str) -> str:
    """Return a line with comments removed, preserving string contents."""
    output: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'} and line[index : index + 3] == char * 3:
            quote = char * 3
            output.append(quote)
            index += 3
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "#":
            break
        output.append(char)
        index += 1
    return "".join(output).strip()


def module_name(relative: str) -> str:
    path = relative[:-3] if relative.endswith(".py") else relative
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def _node_start(node: ast.AST) -> int:
    start = getattr(node, "lineno", 0) or 0
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        start = min(start, min(item.lineno for item in decorators))
    return start


def _node_end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0) or 0


def tightest(nodes: list[ast.AST], line: int) -> ast.AST | None:
    best: ast.AST | None = None
    best_span = 10**9
    for node in nodes:
        start = _node_start(node)
        end = _node_end(node)
        if start <= line <= end:
            span = end - start
            if span < best_span:
                best = node
                best_span = span
    return best


def functions_and_classes(tree: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    functions: list[ast.AST] = []
    classes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
    return functions, classes


def statement_at(tree: ast.AST, line: int) -> ast.stmt | None:
    best: ast.stmt | None = None
    best_span = 10**9
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "lineno"):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= line <= end and end - node.lineno < best_span:
            best = node
            best_span = end - node.lineno
    return best


def assigned_names(statement: ast.AST | None) -> list[str]:
    if statement is None:
        return []
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, ast.AnnAssign) and statement.target is not None:
        targets = [statement.target]
    elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [statement.name]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def iter_python_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*.py")
        if ".git" not in path.parts and "sqlglot-integration-tests" not in path.parts
    ]
    return files


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _fixture_path(call_name: str, argument: str) -> str | None:
    argument = argument.strip()
    if call_name == "check_file":
        relative = argument if argument.endswith(".sql") else f"{argument}.sql"
        return f"tests/fixtures/optimizer/{relative}"
    if call_name in {"load_sql_fixtures", "load_sql_fixture_pairs"}:
        return f"tests/fixtures/{argument.lstrip('/')}"
    return None


def _names_bound_to_fixture(function: ast.AST, fixture: str) -> set[str]:
    """Names this function assigns from a load of `fixture`."""
    bound: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if fixture not in _fixture_paths_in(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
            elif isinstance(target, ast.Attribute):
                bound.add(target.attr)
    return bound


def _fixture_paths_in(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _call_name(child)
        if name is None or not child.args:
            continue
        argument = _literal(child.args[0])
        if argument is None:
            continue
        path = _fixture_path(name, argument)
        if path is not None:
            found.add(path)
    return found


def _references_name(function: ast.AST, name: str) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _nodeid(relative: str, class_name: str | None, function_name: str) -> str:
    if class_name:
        return f"{relative}::{class_name}::{function_name}"
    return f"{relative}::{function_name}"


def build_fixture_index(root: Path) -> dict[str, set[str]]:
    """Map a fixture path to pytest node ids that load it."""
    index: dict[str, set[str]] = {}
    for path in iter_python_files(root / "tests"):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        functions: list[tuple[str | None, ast.AST]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((None, node))
        for class_node in classes:
            for node in class_node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append((class_node.name, node))

        direct: dict[str, list[ast.AST]] = {}
        for class_name, function in functions:
            function_name = function.name  # type: ignore[attr-defined]
            node_id = _nodeid(relative, class_name, function_name)
            for fixture in _fixture_paths_in(function):
                direct.setdefault(fixture, []).append(function)
                if function_name.startswith("test"):
                    index.setdefault(fixture, set()).add(node_id)

        test_functions = [
            (class_name, function)
            for class_name, function in functions
            if function.name.startswith("test")  # type: ignore[attr-defined]
        ]
        for fixture, owners in direct.items():
            bound: set[str] = set()
            for function in owners:
                bound |= _names_bound_to_fixture(function, fixture)
            if not bound:
                continue
            for class_name, function in test_functions:
                if any(_references_name(function, name) for name in bound):
                    index.setdefault(fixture, set()).add(
                        _nodeid(relative, class_name, function.name)  # type: ignore[attr-defined]
                    )
    return index


def build_import_graph(root: Path) -> dict[str, set[str]]:
    """Module name -> modules it imports, one edge per import statement."""
    graph: dict[str, set[str]] = {}
    for path in iter_python_files(root):
        relative = path.relative_to(root).as_posix()
        name = module_name(relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            graph[name] = set()
            continue
        graph[name] = _imports_of(tree, name)
    return graph


def _imports_of(tree: ast.AST, module: str) -> set[str]:
    found: set[str] = set()
    package = module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(package, node.module, node.level or 0)
            if base:
                found.add(base)
    return found


def _resolve_from(package: str, module: str | None, level: int) -> str | None:
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return module
    prefix = parts[: len(parts) - (level - 1)]
    if module:
        prefix.append(module)
    return ".".join(prefix) if prefix else None


def tests_reaching(graph: dict[str, set[str]], test_modules: dict[str, list[str]], target: str) -> set[str]:
    """Pytest node ids whose module transitively imports `target`."""
    reverse: dict[str, set[str]] = {}
    for source, deps in graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(source)
    # Also treat a parent package import as reaching submodules only through recorded edges.
    seen = {target}
    stack = [target]
    while stack:
        current = stack.pop()
        for source in reverse.get(current, ()):
            if source not in seen:
                seen.add(source)
                stack.append(source)
    selected: set[str] = set()
    for module, node_ids in test_modules.items():
        if module in seen:
            selected.update(node_ids)
    return selected


def lines_mentioning(root: Path, name: str, limit: int = 400) -> list[tuple[str, int]] | None:
    """Lines in sqlglot/ whose text contains `name` as an identifier.

    Returns None when the name is too common to be a useful signal.
    """
    needle = name
    found: list[tuple[str, int]] = []
    for path in iter_python_files(root / "sqlglot"):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _has_identifier(line, needle):
                found.append((relative, lineno))
                if len(found) > limit:
                    return None
    return found


def _has_identifier(line: str, name: str) -> bool:
    start = 0
    while True:
        index = line.find(name, start)
        if index < 0:
            return False
        before = line[index - 1] if index else ""
        after = line[index + len(name) : index + len(name) + 1]
        if (not before.isalnum() and before != "_") and (not after.isalnum() and after != "_"):
            return True
        start = index + len(name)
    return False
