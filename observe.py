"""Observe which comparison swaps are equivalent on baseline test inputs."""
import ast
import enum
import importlib.abc
import importlib.machinery
import json
import operator
import sys
from pathlib import Path

ROOT = Path.cwd()
PREFIX = str(ROOT / "sqlglot") + "/"
OBSERVED = {}
CURRENT_TEST = ""

REPLACEMENTS = {
    "Is": ("Eq", operator.is_, operator.eq),
    "IsNot": ("NotEq", operator.is_not, operator.ne),
    "Lt": ("LtE", operator.lt, operator.le),
    "LtE": ("Lt", operator.le, operator.lt),
    "Gt": ("GtE", operator.gt, operator.ge),
    "GtE": ("Gt", operator.ge, operator.gt),
}
# Ordering and equality swaps are pure on the types comparable() accepts, so
# every alternative is recorded. Identity keeps the single paired check.
ORDER_OPS = ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE")
ORDER_FUNCTIONS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtE": operator.le,
    "Gt": operator.gt,
    "GtE": operator.ge,
}


def comparable(left, right):
    if type(left) is not type(right):
        return False
    if type(left) in (int, str, bytes):
        return True
    return (type(left) is tuple and
            all(type(x) is type(y) and type(x) in (int, str, bytes)
                for x, y in zip(left, right)))


def identity_site(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    right = node.comparators[0]
    return ((isinstance(right, ast.Constant) and right.value is None)
            or (isinstance(right, ast.Attribute) and right.attr.isupper()
                and isinstance(right.value, ast.Name)))


class Instrument(ast.NodeTransformer):
    def __init__(self, filename):
        self.filename = filename

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        op = type(node.ops[0]).__name__
        if op not in ORDER_FUNCTIONS and op not in ("Is", "IsNot"):
            return node
        if op in ("Is", "IsNot") and not identity_site(node):
            return node
        key = f"{self.filename}:{node.lineno}:{node.col_offset}"
        return ast.copy_location(ast.Call(
            ast.Name("_blast_observe_identity", ast.Load()),
            [ast.Constant(key), node.left, node.comparators[0],
             ast.Constant(op)], []), node)


def _narrow(previous, equivalent):
    if previous is False:
        return False
    if not equivalent:
        return False
    if previous is None:
        return equivalent
    kept = [name for name in equivalent.split(",") if name in previous.split(",")]
    return ",".join(kept) if kept else False


def observe(key, left, right, op):
    if op in ("Is", "IsNot"):
        original = operator.is_ if op == "Is" else operator.is_not
        partner = "Eq" if op == "Is" else "NotEq"
        result = original(left, right)
        if right is None:
            safe = (left is None or type(left).__eq__ is object.__eq__
                    or type(left) in (str, int, bool, float, tuple, list, dict, set, bytes))
        else:
            safe = (isinstance(right, enum.Enum) and type(left) is type(right)
                    and type(left).__eq__ in (object.__eq__, str.__eq__, int.__eq__))
        equivalent = partner if safe and ORDER_FUNCTIONS[partner](left, right) == result else ""
    else:
        result = ORDER_FUNCTIONS[op](left, right)
        if comparable(left, right):
            equivalent = ",".join(
                name for name in ORDER_OPS
                if name != op and ORDER_FUNCTIONS[name](left, right) == result
            )
        else:
            equivalent = ""
    observations = OBSERVED.setdefault(key, {})
    observations[CURRENT_TEST] = _narrow(observations.get(CURRENT_TEST), equivalent)
    return result


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path)
        tree = Instrument(str(Path(self.path).relative_to(ROOT))).visit(ast.parse(source))
        return compile(ast.fix_missing_locations(tree), self.path, "exec",
                       dont_inherit=True)

    def exec_module(self, module):
        module._blast_observe_identity = observe
        super().exec_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sqlglot" or fullname.startswith("sqlglot."):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and isinstance(spec.origin, str) and spec.origin.startswith(PREFIX):
                spec.loader = Loader(fullname, spec.origin)
                return spec
        return None


def main():
    import pytest
    import escape_map
    from inline_pool import InlinePool

    escape_users = {}

    class Install:
        def pytest_sessionstart(self, session):
            escape_map.install()

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_protocol(self, item):
            global CURRENT_TEST
            CURRENT_TEST = item.nodeid
            escape_map.hits.clear()
            try:
                return (yield)
            finally:
                CURRENT_TEST = ""
                for path, key in escape_map.hits:
                    escape_users.setdefault(path, {}).setdefault(key, []).append(item.nodeid)

        def pytest_collection_finish(self):
            for name, module in tuple(sys.modules.items()):
                if name.startswith("tests.") and hasattr(module, "ProcessPoolExecutor"):
                    module.ProcessPoolExecutor = InlinePool

    sys.meta_path.insert(0, Finder())
    status = pytest.main(["-q", "-p", "no:cacheprovider", *(sys.argv[2:] or ["tests"])], plugins=[Install()])
    # An environment mismatch disables the shortcut.
    Path(sys.argv[1]).write_text(json.dumps({"comparisons": OBSERVED, "escapes": escape_users} if status == 0 else {}))


if __name__ == "__main__":
    main()
