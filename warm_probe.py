"""Start probe children from a parent that has already imported the pinned sqlglot.

A cold probe child imports sqlglot and the test helpers again, about 0.4 s of CPU each time. When a patch
changes only the bodies of functions defined directly in sqlglot modules, and none of those functions ran
while the parent imported, the child can keep the parent's modules: each changed function gets its patched
code object in place, so every reference to it (dispatch tables, partials, subclasses) runs the new body.
Packages are pointed at the patched tree, so a module the child imports for the first time loads its patched
source. Anything else, or any doubt, runs the child cold: it drops every module the parent loaded from the
checkout and imports the patched tree from scratch.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

WARM_MODULES = (
    "sqlglot",
    "sqlglot.optimizer",
    "sqlglot.executor",
    "sqlglot.planner",
    "tests.helpers",
    "tests.dialects.test_dialect",
)
READS_MAP = Path(os.environ.get("READS_MAP", "/app/maps/reads.json"))
_WRAPPER_DEPTH = 8


@dataclass
class Swap:
    """Patched code objects for functions of one module, by qualified name."""

    path: str
    module: str
    codes: dict[str, CodeType]


@dataclass
class _Shape:
    """A module's function bodies by qualified name, and its dump with those bodies emptied."""

    bodies: dict[str, list[str]]
    blanked: str


class Warm:
    """The parent's warm import, what ran during it, and which modules it loaded from the checkout."""

    def __init__(self) -> None:
        self.ready = False
        self.failed = False
        self.ran: set[tuple[str, str]] = set()
        self.owned: set[str] = set()
        self._text_read: set[str] | None = None
        self._old_shapes: dict[str, _Shape | None] = {}
        self._swaps: dict[str, list[Swap] | None] = {}

    def swaps(self, root: Path, patched: dict[str, str]) -> list[Swap] | None:
        """The code swaps that let a warm child stand in for a cold one; None when it cannot.

        A module the parent never loaded needs no swap, whatever the edit: the child imports it from the
        patched tree.
        """
        if not patched or self.failed or not self._eligible_paths(patched) or not self._import(root):
            return None
        key = hashlib.sha1(json.dumps(sorted(patched.items())).encode()).hexdigest()
        if key not in self._swaps:
            self._swaps[key] = self._find_swaps(root, patched)
        return self._swaps[key]

    def _find_swaps(self, root: Path, patched: dict[str, str]) -> list[Swap] | None:
        found: list[Swap] = []
        for path, text in patched.items():
            if _module_name(path) not in self.owned:
                continue
            if path not in self._old_shapes:
                self._old_shapes[path] = _old_shape(root / path)
            swap = _module_swap(root, path, self._old_shapes[path], text)
            if swap is None or any((str(root / path), name) in self.ran for name in swap.codes):
                return None
            if swap.codes:
                found.append(swap)
        return found

    def _eligible_paths(self, patched: dict[str, str]) -> bool:
        if self._text_read is None:
            self._text_read = _text_read_paths()
        return all(path.startswith("sqlglot/") and path.endswith(".py") for path in patched) and not (
            set(patched) & self._text_read
        )

    def _import(self, root: Path) -> bool:
        """Import WARM_MODULES once, recording every sqlglot function that runs meanwhile."""
        if self.ready or self.failed:
            return self.ready
        prefix = str(root / "sqlglot") + os.sep
        ran = self.ran
        monitoring = sys.monitoring
        tool = _free_tool()

        def started(code: CodeType, _offset: int) -> object:
            if code.co_filename.startswith(prefix):
                ran.add((code.co_filename, code.co_qualname))
            return monitoring.DISABLE

        saved = (list(sys.path), os.getcwd(), os.environ.get("SKIP_INTEGRATION"))
        before = set(sys.modules)
        monitoring.use_tool_id(tool, "warm_probe")
        try:
            monitoring.register_callback(tool, monitoring.events.PY_START, started)
            monitoring.set_events(tool, monitoring.events.PY_START)
            sys.path.insert(0, str(root))
            os.chdir(root)
            os.environ["SKIP_INTEGRATION"] = "0"
            for name in WARM_MODULES:
                importlib.import_module(name)
            self.ready = True
        except Exception:  # noqa: BLE001 - a parent that cannot import stays cold
            self.failed = True
        finally:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool)
            sys.path[:] = saved[0]
            os.chdir(saved[1])
            if saved[2] is None:
                os.environ.pop("SKIP_INTEGRATION", None)
            else:
                os.environ["SKIP_INTEGRATION"] = saved[2]
            self.owned = {name for name in set(sys.modules) - before if _under(sys.modules[name], root)}
        if self.failed:
            self.drop_owned()
        return self.ready

    def enter(self, root: Path, tree: Path, swaps: list[Swap] | None) -> None:
        """In the child: make the loaded modules behave as the patched tree, warm or cold."""
        if swaps is None or not self._apply(root, tree, swaps):
            self.drop_owned()

    def drop_owned(self) -> None:
        for name in self.owned:
            sys.modules.pop(name, None)

    def _apply(self, root: Path, tree: Path, swaps: list[Swap]) -> bool:
        for name in self.owned:
            module = sys.modules.get(name)
            locations = getattr(module, "__path__", None)
            if module is None or locations is None:
                continue
            entries = list(locations)
            if not all(_inside(entry, root) for entry in entries):
                return False
            moved = [str(tree / Path(entry).relative_to(root)) for entry in entries]
            module.__path__ = moved
            if module.__spec__ is not None:
                module.__spec__.submodule_search_locations = moved
        for swap in swaps:
            module = sys.modules.get(swap.module)
            if module is None:
                continue
            live = _functions(module)
            source = str(root / swap.path)
            for qualname, code in swap.codes.items():
                targets = [
                    function
                    for function in live.get(qualname, ())
                    if function.__code__.co_qualname == qualname and function.__code__.co_filename == source
                ]
                if not targets or any(function.__code__.co_freevars != code.co_freevars for function in targets):
                    return False
                for function in targets:
                    function.__code__ = code
        return True


def _under(module: ModuleType | None, root: Path) -> bool:
    location = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", None) or []), None)
    return isinstance(location, str) and _inside(location, root)


def _inside(location: str, root: Path) -> bool:
    return location == str(root) or location.startswith(str(root) + os.sep)


def _free_tool() -> int:
    for tool in range(5, -1, -1):
        if sys.monitoring.get_tool(tool) is None:
            return tool
    raise RuntimeError("no free sys.monitoring tool id")


def _text_read_paths() -> set[str]:
    try:
        data = json.loads(READS_MAP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"*"}
    return {path for entry in data.get("tests", {}).values() for path in entry.get("reads", [])}


def _old_shape(source: Path) -> _Shape | None:
    try:
        return _shape(ast.parse(source.read_text(encoding="utf-8")))
    except (OSError, SyntaxError, ValueError):
        return None


def _module_swap(root: Path, path: str, old: _Shape | None, text: str) -> Swap | None:
    """Patched code per changed function of one module; None unless only function bodies changed."""
    if old is None:
        return None
    try:
        new = _shape(ast.parse(text))
        new_code = compile(text, str(root / path), "exec")
    except (SyntaxError, ValueError):
        return None
    if old.blanked != new.blanked or old.bodies.keys() != new.bodies.keys():
        return None
    changed = [name for name, bodies in new.bodies.items() if old.bodies[name] != bodies]
    if any(len(new.bodies[name]) != 1 for name in changed):
        return None
    codes = _def_codes(new_code)
    if any(name not in codes for name in changed):
        return None
    return Swap(path, _module_name(path), {name: codes[name] for name in changed})


def _module_name(path: str) -> str:
    module = path[: -len(".py")].replace("/", ".")
    return module[: -len(".__init__")] if module.endswith(".__init__") else module


def _shape(tree: ast.Module) -> _Shape:
    """Dump each function body defined directly in the module or a class body, then empty it in place."""
    bodies: dict[str, list[str]] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                dumped = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                bodies.setdefault(prefix + node.name, []).append(dumped)
                node.body = [ast.Pass()]

    visit(tree.body, "")
    return _Shape(bodies, ast.dump(tree))


def _def_codes(module_code: CodeType) -> dict[str, CodeType]:
    """Function code objects of the module and its class bodies with a unique qualified name."""
    found: dict[str, CodeType] = {}
    duplicated: set[str] = set()

    def visit(code: CodeType) -> None:
        for const in code.co_consts:
            if not isinstance(const, CodeType) or "<" in const.co_qualname:
                continue
            if const.co_flags & inspect.CO_NEWLOCALS:
                if const.co_qualname in found:
                    duplicated.add(const.co_qualname)
                found[const.co_qualname] = const
            else:
                visit(const)

    visit(module_code)
    return {name: code for name, code in found.items() if name not in duplicated}


def _functions(module: ModuleType) -> dict[str, list[FunctionType]]:
    """Function objects of the module's own functions and classes, by qualified name."""
    found: dict[str, list[FunctionType]] = {}
    seen: set[int] = set()

    def add(value: object) -> None:
        for candidate in _unwrapped(value):
            if id(candidate) not in seen:
                seen.add(id(candidate))
                found.setdefault(candidate.__qualname__, []).append(candidate)

    def visit_class(cls: type, depth: int) -> None:
        for value in vars(cls).values():
            if isinstance(value, type):
                if value.__module__ == module.__name__ and depth < 4:
                    visit_class(value, depth + 1)
            else:
                add(value)

    for value in vars(module).values():
        if isinstance(value, type):
            if value.__module__ == module.__name__:
                visit_class(value, 0)
        else:
            add(value)
    return found


def _unwrapped(value: object) -> list[FunctionType]:
    """The plain functions behind an attribute: methods, static/class methods, properties, wrappers."""
    pending: list[tuple[object, int]] = [(value, 0)]
    found: list[FunctionType] = []
    while pending:
        item, depth = pending.pop()
        if depth > _WRAPPER_DEPTH or isinstance(item, (ModuleType, type)):
            continue
        if isinstance(item, FunctionType):
            found.append(item)
            inner = item.__dict__.get("__wrapped__")
        elif isinstance(item, (staticmethod, classmethod)):
            inner = item.__func__
        elif isinstance(item, property):
            pending.extend((part, depth + 1) for part in (item.fget, item.fset, item.fdel) if part is not None)
            continue
        else:
            inner = _inner_function(item)
        if inner is not None:
            pending.append((inner, depth + 1))
    return found


def _inner_function(item: object) -> object | None:
    """The function a cache wrapper, partial or cached property holds, without running lazy attribute hooks."""
    for name in ("__wrapped__", "func"):
        try:
            inner = object.__getattribute__(item, name)
        except AttributeError:
            continue
        if callable(inner):
            return inner
    return None


WARM = Warm()
