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
from edits import (
    boundary_unreachable,
    changed_dict_key,
    data_only,
    enum_names,
    identity_swap,
    loaded_mentions,
    neutral_identity,
    proven_readers,
)
from pack import pack
from probe_small import confirm, should_probe

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


def _owning_dialect(path: str) -> str | None:
    """Dialect a dialects/parsers/generators/typing module belongs to."""
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "sqlglot":
        return None
    if parts[1] not in {"dialects", "parsers", "generators", "typing"}:
        return None
    name = parts[2]
    if not name.endswith(".py") or name == "__init__.py":
        return None
    return name[: -len(".py")]


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
        self.id_of = {test["number"]: test["id"].lower() for test in catalog}
        self.by_file: dict[str, list[int]] = {}
        self.test_modules: dict[str, list[str]] = {}
        for test in catalog:
            self.by_file.setdefault(test["file"], []).append(test["number"])
            self.test_modules.setdefault(module_name(test["file"]), []).append(test["id"])
        self.fixtures = build_fixture_index(root)
        self.imports = build_import_graph(root)
        self.enums = enum_names(root)
        self._mentions: dict[str, list[tuple[str, int]] | None] = {}
        self._key_lines: dict[str, set[int]] = {}

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
        # `<`/`<=` on dialect.version versus a tuple that cannot be equal to it
        # takes the same branch for every version this constructor stores.
        # `is`/`==` is not in that set: equal objects that are not the same
        # object can fail a test, so those lines keep the tests that ran them.
        if boundary_unreachable(old_tree, new_tree) or neutral_identity(old_tree, new_tree, self.enums):
            return set()
        # A value change, or an identity comparison, does not send new tests
        # into the line. Tests that never executed it cannot observe it.
        # Other control-flow edits stay wide.
        narrow = data_only(old_tree, new_tree) or identity_swap(old_tree, new_tree)
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
            # Coverage can record a different line of the same function than the
            # one the patch edits. The whole function is the set that still
            # contained every sample failure.
            boolop = tightest([node for node in ast.walk(tree) if isinstance(node, ast.BoolOp)], line)
            if boolop is not None:
                owners = tests_on_lines(self.coverage, path, boolop.lineno, _end(boolop))
                # A failing test can execute the function while coverage records a
                # different line of the same expression. Keep the whole function.
                owners |= tests_on_lines(self.coverage, path, _start(function), _end(function))
                if owners:
                    return owners
            if narrow:
                start = statement.lineno if statement is not None else line
                end = statement.end_lineno if statement is not None and statement.end_lineno else line
                selected = tests_on_lines(self.coverage, path, start, end)
                if selected:
                    return selected
            start = statement.lineno if statement is not None else line
            end = statement.end_lineno if statement is not None and statement.end_lineno else line
            selected = tests_on_lines(self.coverage, path, start, end)
            if not selected:
                tail = tests_on_lines(self.coverage, path, line, _end(function))
                if tail:
                    return tail
                return tests_on_lines(self.coverage, path, _start(function), _end(function))
            return selected
        if narrow:
            keyed = self._dict_key_readers(tree, line)
            if keyed:
                return self._dialect_readers(path, statement_at(tree, line), keyed, narrow)
        return self._scope_name(path, tree, line, tightest(classes, line), narrow)

    def _dialect_readers(
        self, path: str, statement: ast.AST | None, readers: set[int], narrow: bool
    ) -> set[int]:
        """Keep this dialect's tests and shared tests, drop other dialects.

        `TSQL.WEEK_OFFSET` is also named on BigQuery and Oracle. Those tests read
        their own class attribute. A shared optimizer test such as `test_simplify`
        has no dialect in its path and can still observe the override, so it stays.
        """
        if not narrow:
            return readers
        dialect = _owning_dialect(path)
        if dialect is None:
            return readers
        if isinstance(statement, ast.Assign) and not isinstance(statement.value, (ast.Dict, ast.Constant, ast.UnaryOp)):
            return readers
        narrowed = {
            number
            for number in readers
            if "/dialects/" not in self.id_of.get(number, "") or dialect in self.id_of.get(number, "")
        }
        # Another dialect's copy of this dict does not hold the overridden table.
        # A bare integer can still be read by a shared test with no dialect in its name.
        if isinstance(statement.value, ast.Dict):
            # Other dialects' copies of this table are not this override. If none
            # of the remaining readers are this dialect, keep the tests that
            # executed the class itself. An empty set here is a missed failure.
            return narrowed or self._class_observers(path, statement) or self._dialect_from_hot(path, statement)
        return narrowed or readers

    def _class_observers(self, path: str, statement: ast.AST) -> set[int]:
        """Tests that executed this class, skipping lines shared by the whole suite."""
        source = self.root / path
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return set()
        _, classes = functions_and_classes(tree)
        class_node = tightest(classes, statement.lineno)
        if class_node is None:
            return set()
        selected: set[int] = set()
        by_line = self.coverage.get(path, {})
        for lineno in range(_start(class_node), _end(class_node) + 1):
            owners = by_line.get(lineno, ())
            if 0 < len(owners) <= HOT_REFERENCE:
                selected.update(owners)
        return selected

    def _dialect_from_hot(self, path: str, statement: ast.AST) -> set[int]:
        """This dialect's tests on a line shared by the whole suite.

        Import-time tables are often executed only on a helper line that every
        test runs. The tests whose id names this dialect are the ones that
        loaded this class's copy.
        """
        dialect = _owning_dialect(path)
        names = assigned_names(statement)
        if dialect is None or len(names) != 1 or self.mentions(names[0]) is None:
            return set()
        selected: set[int] = set()
        for relative, lineno in self.mentions(names[0]) or []:
            owners = self.coverage.get(relative, {}).get(lineno, ())
            if len(owners) > HOT_REFERENCE:
                selected.update(number for number in owners if dialect in self.id_of.get(number, ""))
        return selected

    def _dict_key_readers(self, tree: ast.AST, line: int) -> set[int]:
        """Tests that execute a line naming this dict entry, not every importer of the literal."""
        key = changed_dict_key(tree, line)
        if key is None:
            return set()
        if key not in self._key_lines:
            owners: set[int] = set()
            needle = repr(key)
            for relative, by_line in self.coverage.items():
                source = self.root / relative
                if not source.is_file():
                    continue
                try:
                    text = source.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for lineno, tests in by_line.items():
                    if 1 <= lineno <= len(text) and needle in text[lineno - 1]:
                        owners |= tests
            self._key_lines[key] = owners
        owners = self._key_lines[key]
        if not owners or len(owners) > HOT_REFERENCE:
            return set()
        return set(owners)

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
            # The assignment line runs at import. Its coverage is every test that
            # loaded the module, not every test that observes the new value.
            # A line executed by most of the suite is a shared helper passing the
            # whole object through. It is not evidence that those tests look at
            # the constant that changed, so it must not widen the set to the file.
            defined = range(statement.lineno, (statement.end_lineno or statement.lineno) + 1)
            for relative, lineno in self.mentions(names[0]) or []:
                if relative == path and lineno in defined:
                    continue
                owners = self.coverage.get(relative, {}).get(lineno, set())
                if len(owners) > HOT_REFERENCE:
                    continue
                readers |= owners
            if readers:
                # Drop tests that only stored the same name on another class.
                # Tests that load it, including through a shared dialect helper, stay.
                loaded = loaded_mentions(
                    names[0], self.mentions, self.coverage, self.root, path, tree, line
                )
                kept = readers & loaded if loaded is not None else set()
                if kept:
                    return self._dialect_readers(path, statement, kept, narrow)
                return readers or self._line_or_class(path, tree, line, class_node)
            return readers or self._line_or_class(path, tree, line, class_node)
        useful = [name for name in names if self.mentions(name) is not None]
        if class_node is not None and not useful:
            class_name = class_node.name  # type: ignore[attr-defined]
            if self.mentions(class_name) is not None:
                useful.append(class_name)
        # A data-only edit does not change control flow, so tests that merely
        # imported the file cannot start failing.
        selected: set[int] = set() if narrow else tests_touching(self.coverage, path)
        if class_node is not None:
            selected |= tests_on_lines(self.coverage, path, _start(class_node), _end(class_node))
        for name in useful:
            for relative, lineno in self.mentions(name) or []:
                owners = self.coverage.get(relative, {}).get(lineno, set())
                if 0 < len(owners) <= HOT_REFERENCE:
                    selected |= owners
        if narrow and len(names) == 1 and selected:
            class_name = class_node.name if class_node is not None else None  # type: ignore[attr-defined]
            proven = proven_readers(
                names[0], class_name, self.mentions, self.coverage, self.root, HOT_REFERENCE
            )
            kept = selected & proven if proven is not None else set()
            if kept:
                return kept
        return selected or self._line_or_class(path, tree, line, class_node)

    def _line_or_class(
        self, path: str, tree: ast.AST, line: int, class_node: ast.AST | None
    ) -> set[int]:
        """Tests that executed this line or its class. Never the shared helper line."""
        selected = set(self.coverage.get(path, {}).get(line, ()))
        if selected:
            return selected
        if class_node is not None:
            selected = tests_on_lines(self.coverage, path, _start(class_node), _end(class_node))
            if selected:
                return selected
        return tests_touching(self.coverage, path)

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


def drop_slow_if_companions_pass(
    patch_text: str,
    catalog: list[dict],
    selected: set[int],
    seconds: dict[int, float],
    heavy: set[int],
) -> set[int]:
    """Drop a long test when every short companion passes.

    A 21s test only stays when a test under half a second, run on the patched
    tree, actually fails. Optimizer and executor fixtures are not used as that
    signal: replaying them costs more than a worker.
    """
    slow = {number for number in selected if seconds.get(number, 0.0) >= 3.0}
    companions = selected - slow
    if not slow or not companions or companions & heavy:
        return selected
    if len(companions) > MAX_COMPANIONS:
        return selected
    if max(seconds.get(number, 0.0) for number in companions) > 0.5:
        return selected
    failed = confirm(ROOT, patch_text, catalog, companions)
    if not failed:
        return set()
    if failed == companions:
        return selected
    return slow | failed


MAX_COMPANIONS = 8


def main() -> None:
    catalog = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    numbers = {test["id"]: test["number"] for test in catalog}
    seconds = {test["number"]: float(test["seconds"]) for test in catalog}
    coverage = load_coverage(COVERAGE_MAP, numbers)
    selector = Selector(ROOT, coverage, catalog)
    # These tests call check_file, which re-executes an entire SQL fixture.
    heavy = {
        test["number"]
        for test in catalog
        if test["file"] in {"tests/test_optimizer.py", "tests/test_executor.py"}
    }
    commits = sorted((DATA / "commits").glob("*.patch"))
    for patch in commits:
        text = patch.read_text(encoding="utf-8")
        selected = selector.choose(text)
        selected = drop_slow_if_companions_pass(text, catalog, selected, seconds, heavy)
        if should_probe(selected, seconds, text, heavy):
            selected = confirm(ROOT, text, catalog, selected)
        print(json.dumps({"commit": patch.stem, "workers": pack(selected, seconds)}, separators=(",", ":")))
        print(f"{patch.stem}: {len(selected)} tests", file=sys.stderr)


if __name__ == "__main__":
    main()
