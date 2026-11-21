"""Choose the sqlglot tests a patch can break, using a coverage map of the pinned commit.

A miss costs far more than an extra test, so a changed line pulls in every test
that executes it. Class and module assignments also pull in tests that execute
a reference to the assigned name, because those lines run at import time and
are not attributed to the tests that later read them.
"""

from __future__ import annotations

import ast
import json
import os
import struct
import sys
from pathlib import Path

from analyze import (
    assigned_names,
    build_fixture_index,
    build_import_graph,
    functions_and_classes,
    lines_mentioning,
    module_name,
    python_code,
    statement_at,
    tests_reaching,
    tightest,
)
from diffutil import DiffApplyError, FileDiff, added_lines, apply_hunks, parse_patch, removed_lines
from edits import component_sinks, data_only, identity_swap
from pack import pack

DATA = Path(os.environ.get("DATA_DIR", "/data"))
ROOT = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
COVERAGE_MAP = Path(os.environ.get("COVERAGE_MAP", "/app/maps/coverage.bin"))
# A reference line executed by this many tests is a shared helper, not a signal
# that those tests depend on the particular class attribute that changed.
HOT_REFERENCE = 40


def load_coverage(path: Path, numbers: dict[str, int]) -> dict[str, dict[int, set[int]]]:
    """File -> line -> test numbers. Coverage indices are translated to tests.json."""
    blob = path.read_bytes()
    if blob[:4] != b"BRC1":
        raise SystemExit(f"bad coverage map: {path}")
    offset = 4
    (count,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    nodeids: list[str] = []
    for _ in range(count):
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        nodeids.append(blob[offset : offset + length].decode("utf-8"))
        offset += length
    (file_count,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    lines: dict[str, dict[int, set[int]]] = {}
    for _ in range(file_count):
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        relative = blob[offset : offset + length].decode("utf-8")
        offset += length
        (line_count,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        by_line: dict[int, set[int]] = {}
        for _line in range(line_count):
            lineno, owners = struct.unpack_from("<IH", blob, offset)
            offset += 6
            if owners:
                raw = struct.unpack_from(f"<{owners}H", blob, offset)
                offset += 2 * owners
            else:
                raw = ()
            translated = {numbers[nodeids[index]] for index in raw if nodeids[index] in numbers}
            if translated:
                by_line[lineno] = translated
        if by_line:
            lines[relative] = by_line
    return lines


def tests_on_lines(coverage: dict[str, dict[int, set[int]]], relative: str, start: int, end: int) -> set[int]:
    by_line = coverage.get(relative, {})
    selected: set[int] = set()
    for lineno in range(start, end + 1):
        selected.update(by_line.get(lineno, ()))
    return selected


def tests_touching(coverage: dict[str, dict[int, set[int]]], relative: str) -> set[int]:
    selected: set[int] = set()
    for owners in coverage.get(relative, {}).values():
        selected.update(owners)
    return selected


def _start(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1) or 1
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        start = min(start, min(item.lineno for item in decorators))
    return start


def _end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or _start(node)


def _signature(function: ast.AST, line: int) -> bool:
    body = getattr(function, "body", None) or []
    body_start = body[0].lineno if body else _start(function) + 1
    return _start(function) <= line < body_start


class Selector:
    def __init__(self, root: Path, coverage: dict[str, dict[int, set[int]]], catalog: list[dict]) -> None:
        self.root = root
        self.coverage = coverage
        self.by_node = {test["id"]: test["number"] for test in catalog}
        self.by_file: dict[str, list[int]] = {}
        self.test_modules: dict[str, list[str]] = {}
        for test in catalog:
            self.by_file.setdefault(test["file"], []).append(test["number"])
            self.test_modules.setdefault(module_name(test["file"]), []).append(test["id"])
        self.fixtures = build_fixture_index(root)
        self.imports = build_import_graph(root)
        self._mentions: dict[str, list[tuple[str, int]] | None] = {}

    def mentions(self, name: str) -> list[tuple[str, int]] | None:
        if name not in self._mentions:
            self._mentions[name] = lines_mentioning(self.root, name)
        return self._mentions[name]

    def numbers_for(self, node_ids: set[str]) -> set[int]:
        return {self.by_node[node_id] for node_id in node_ids if node_id in self.by_node}

    def importers_of(self, relative: str) -> set[int]:
        return self.numbers_for(tests_reaching(self.imports, self.test_modules, module_name(relative)))

    def choose(self, patch_text: str) -> set[int]:
        selected: set[int] = set()
        for file_diff in parse_patch(patch_text):
            selected |= self._file(file_diff)
        return selected

    def _file(self, file_diff: FileDiff) -> set[int]:
        path = file_diff.path
        if path is None or file_diff.binary:
            return set()
        if path.startswith("tests/") and path.endswith(".py"):
            return self._test_python(file_diff, path)
        if path.startswith("tests/"):
            return self._fixture(path)
        if path.endswith(".py") and (self.root / path).is_file():
            return self._library_python(file_diff, path)
        if not (self.root / path).exists() and path.endswith(".py"):
            return self.importers_of(path) | tests_touching(self.coverage, path)
        return set()

    def _fixture(self, path: str) -> set[int]:
        found = self.numbers_for(self.fixtures.get(path, set()))
        if found:
            return found
        stem = Path(path).stem
        if len(stem) < 6:
            return set()
        matched: set[int] = set()
        for relative, numbers in self.by_file.items():
            source = self.root / relative
            if source.is_file() and stem in source.read_text(encoding="utf-8"):
                matched.update(numbers)
        return matched

    def _test_python(self, file_diff: FileDiff, path: str) -> set[int]:
        original = (self.root / path).read_text(encoding="utf-8") if (self.root / path).is_file() else ""
        try:
            patched = apply_hunks(original, file_diff.hunks) if file_diff.old_path else _added_text(file_diff)
            new_tree = ast.parse(patched)
        except (DiffApplyError, SyntaxError):
            return set(self.by_file.get(path, ()))
        if not original:
            return set(self.by_file.get(path, ()))
        old_tree = ast.parse(original)
        if ast.dump(old_tree) == ast.dump(new_tree):
            return set()
        changed = _changed_test_names(old_tree, new_tree)
        if changed is None:
            return set(self.by_file.get(path, ()))
        selected: set[int] = set()
        for class_name, function_name in changed:
            node_id = f"{path}::{class_name}::{function_name}" if class_name else f"{path}::{function_name}"
            if node_id in self.by_node:
                selected.add(self.by_node[node_id])
        return selected or set(self.by_file.get(path, ()))

    def _library_python(self, file_diff: FileDiff, path: str) -> set[int]:
        original_path = self.root / (file_diff.old_path or path)
        if file_diff.old_path is None:
            return self.importers_of(path)
        if file_diff.new_path is None:
            return self.importers_of(file_diff.old_path) | tests_touching(self.coverage, file_diff.old_path)
        original = original_path.read_text(encoding="utf-8")
        try:
            patched = apply_hunks(original, file_diff.hunks)
        except DiffApplyError:
            return self.importers_of(path) | tests_touching(self.coverage, path)
        try:
            new_tree = ast.parse(patched)
        except SyntaxError:
            return self.importers_of(path) | tests_touching(self.coverage, path)
        old_tree = ast.parse(original)
        if identity_swap(old_tree, new_tree):
            return set()
        # A value change does not send new tests into the line. Tests that
        # never executed it cannot observe it. Control-flow edits stay wide.
        narrow = data_only(old_tree, new_tree)
        selected: set[int] = set()
        removed = [(line, text) for line, text in removed_lines(file_diff.hunks) if python_code(text)]
        added = [(line, text) for line, text in added_lines(file_diff.hunks) if python_code(text)]
        for line, _text in removed:
            selected |= self._old_line(path, old_tree, line, narrow)
        if added and not removed:
            selected |= self._insertion(path, old_tree, new_tree, [line for line, _text in added])
        return selected

    def _old_line(self, path: str, tree: ast.AST, line: int, narrow: bool) -> set[int]:
        functions, classes = functions_and_classes(tree)
        # Calling a lambda executes its body. Tests that only imported the class
        # ran the header and did not see a value change inside the body.
        if narrow:
            lambda_node = tightest([node for node in ast.walk(tree) if isinstance(node, ast.Lambda)], line)
            if lambda_node is not None:
                start = lambda_node.lineno + 1 if line > lambda_node.lineno else line
                selected = tests_on_lines(self.coverage, path, start, _end(lambda_node))
                if selected:
                    return selected
        function = tightest(functions, line)
        if function is not None and _signature(function, line):
            selected = tests_on_lines(self.coverage, path, _start(function), _end(function))
            return selected or tests_touching(self.coverage, path)
        if function is not None:
            statement = statement_at(tree, line)
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                return self.importers_of(path) | tests_touching(self.coverage, path)
            if narrow:
                selected = self.coverage.get(path, {}).get(line, set())
                if selected:
                    return set(selected)
            start = statement.lineno if statement is not None else line
            end = statement.end_lineno if statement is not None and statement.end_lineno else line
            selected = tests_on_lines(self.coverage, path, start, end)
            if not selected:
                return tests_on_lines(self.coverage, path, line, _end(function))
            return selected
        return self._scope_name(path, tree, line, tightest(classes, line), narrow)

    def _observed_value(self, name: str, class_node: ast.AST | None) -> set[int]:
        if self.mentions(name) is not None:
            sinks = component_sinks(self.root, name, self.mentions)
            readers: set[int] = set()
            for relative, lineno in sinks or []:
                readers |= self.coverage.get(relative, {}).get(lineno, set())
            return readers
        if class_node is None:
            return set()
        class_name = class_node.name  # type: ignore[attr-defined]
        mentions = self.mentions(class_name)
        if not mentions:
            return set()
        readers = set()
        for relative, lineno in mentions:
            owners = self.coverage.get(relative, {}).get(lineno, set())
            if len(owners) > HOT_REFERENCE:
                return set()
            readers |= owners
        return readers

    def _scope_name(
        self, path: str, tree: ast.AST, line: int, class_node: ast.AST | None, narrow: bool
    ) -> set[int]:
        statement = statement_at(tree, line)
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            return self.importers_of(path) | tests_touching(self.coverage, path)
        names = assigned_names(statement)
        # Tests that never execute a line naming this assignment cannot observe
        # its new value. A name on too many lines falls through to the file.
        if len(names) == 1 and self.mentions(names[0]) is not None:
            readers: set[int] = set()
            shared = False
            for relative, lineno in self.mentions(names[0]) or []:
                owners = self.coverage.get(relative, {}).get(lineno, set())
                if len(owners) > HOT_REFERENCE:
                    shared = True
                    break
                readers |= owners
            if readers and not shared:
                return readers
        if narrow and len(names) == 1:
            observed = self._observed_value(names[0], class_node)
            if observed:
                return observed
        useful = [name for name in names if self.mentions(name) is not None]
        if class_node is not None and not useful:
            class_name = class_node.name  # type: ignore[attr-defined]
            if self.mentions(class_name) is not None:
                useful.append(class_name)
        selected: set[int] = tests_touching(self.coverage, path)
        if class_node is not None:
            selected |= tests_on_lines(self.coverage, path, _start(class_node), _end(class_node))
        for name in useful:
            for relative, lineno in self.mentions(name) or []:
                owners = self.coverage.get(relative, {}).get(lineno, set())
                if 0 < len(owners) <= HOT_REFERENCE:
                    selected |= owners
        return selected

    def _insertion(self, path: str, old_tree: ast.AST, new_tree: ast.AST, lines: list[int]) -> set[int]:
        selected: set[int] = set()
        old_functions, old_classes = functions_and_classes(old_tree)
        new_functions, new_classes = functions_and_classes(new_tree)
        for line in lines:
            function = tightest(new_functions, line)
            if function is None:
                selected |= self._scope_name(path, new_tree, line, tightest(new_classes, line), False)
                continue
            class_node = tightest(new_classes, line)
            class_name = class_node.name if class_node is not None else None  # type: ignore[attr-defined]
            match = _same_function(old_functions, old_classes, function.name, class_name)  # type: ignore[attr-defined]
            if match is None:
                selected |= tests_touching(self.coverage, path)
            else:
                selected |= tests_on_lines(self.coverage, path, _start(match), _end(match))
        return selected


def _same_function(
    functions: list[ast.AST], classes: list[ast.AST], name: str, class_name: str | None
) -> ast.AST | None:
    for function in functions:
        if function.name != name:  # type: ignore[attr-defined]
            continue
        parent = tightest(classes, function.lineno)  # type: ignore[attr-defined]
        parent_name = parent.name if parent is not None else None  # type: ignore[attr-defined]
        if parent_name == class_name:
            return function
    return None


def _changed_test_names(old_tree: ast.AST, new_tree: ast.AST) -> set[tuple[str | None, str]] | None:
    """Changed test methods, or None when a helper or class body changed too."""
    old_map = _test_dumps(old_tree)
    new_map = _test_dumps(new_tree)
    if _dump_without_tests(old_tree) != _dump_without_tests(new_tree):
        return None
    changed = {key for key in old_map.keys() | new_map.keys() if old_map.get(key) != new_map.get(key)}
    return changed


def _test_dumps(tree: ast.AST) -> dict[tuple[str | None, str], str]:
    found: dict[tuple[str | None, str], str] = {}
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    found[(node.name, item.name)] = ast.dump(item, include_attributes=False)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[(None, node.name)] = ast.dump(node, include_attributes=False)
    return found


def _dump_without_tests(tree: ast.AST) -> str:
    clone = ast.parse(ast.unparse(tree)) if hasattr(ast, "unparse") else tree
    # Compare everything except test-method bodies by blanking those bodies.
    for node in ast.walk(clone):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            node.body = [ast.Pass()]
    return ast.dump(clone, include_attributes=False)


def _added_text(file_diff: FileDiff) -> str:
    return "\n".join(text for _line, text in added_lines(file_diff.hunks)) + "\n"


def main() -> None:
    catalog = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    numbers = {test["id"]: test["number"] for test in catalog}
    seconds = {test["number"]: float(test["seconds"]) for test in catalog}
    coverage = load_coverage(COVERAGE_MAP, numbers)
    selector = Selector(ROOT, coverage, catalog)
    commits = sorted((DATA / "commits").glob("*.patch"))
    for patch in commits:
        selected = selector.choose(patch.read_text(encoding="utf-8"))
        print(json.dumps({"commit": patch.stem, "workers": pack(selected, seconds)}, separators=(",", ":")))
        print(f"{patch.stem}: {len(selected)} tests", file=sys.stderr)


if __name__ == "__main__":
    main()
