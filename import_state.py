"""Digest the state sqlglot builds at import, so an edit to code that runs then can be judged by its result.

Functions run while sqlglot imports (metaclasses, tries, templates parsed in
class bodies) leave what they compute in module globals and class attributes.
When every module global and class attribute of sqlglot and tests.helpers
digests the same after the patched import as after the pinned one, the edited
code changed nothing a later query reads, except by running again.
Classes and functions stand for themselves by name, their attributes, defaults
and closure cells; code is left out, since what it does when run is
judged per query; loggers, process-wide configuration, by name. Sets digest
in sorted order, dicts, lists and tuples in their order unless the map marks
the key `unordered` (its order on the pinned commit varies with hash seed,
import order or process), and keys whose value varies anyway are skipped.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import logging
import os
import pkgutil
import re
import sys
import types
from pathlib import Path

_OWN = ("sqlglot", "tests.helpers")
_PLAIN = (bool, int, float, complex, str, bytes)
_BUILTIN = (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType, types.MethodWrapperType)
_SKIPPED = {"__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__", "__static_attributes__"}


def state_path(tpc_map: Path) -> Path:
    """Where the pinned import state lives beside a TPC map."""
    return tpc_map.with_name(tpc_map.stem + "_state.json")


def import_everything() -> None:
    """Import every sqlglot module but __main__, and the test helpers, as the pinned snapshot did."""
    import sqlglot
    import tests.helpers  # noqa: F401 - its schemas are read by the TPC tests

    for module in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if not module.name.endswith("__main__"):
            with contextlib.suppress(Exception):
                __import__(module.name)


def _own(module: str) -> bool:
    return any(module == name or module.startswith(name + ".") for name in _OWN)


class _Canon:
    """Text standing for a value; classes reached through it are queued to be digested by attribute."""

    def __init__(self, roots: list[str]) -> None:
        self.ordered = True
        self.classes: list[type] = []
        self._roots = roots
        self._queued: set[int] = set()
        self._open: set[int] = set()

    def ref(self, cls: type) -> str:
        if _own(cls.__module__) and id(cls) not in self._queued:
            self._queued.add(id(cls))
            self.classes.append(cls)
        return f"T:{cls.__module__}.{cls.__qualname__}"

    def of(self, value: object) -> str:
        if type(value) is str:
            for root in self._roots:
                value = value.replace(root, "<root>")  # type: ignore[attr-defined]
            return repr(value)
        if value is None or type(value) in _PLAIN:
            return repr(value)
        if isinstance(value, type):
            return self.ref(value)
        if isinstance(value, types.ModuleType):
            return "M:" + value.__name__
        if isinstance(value, enum.Enum):
            return f"E:{self.ref(type(value))}.{value.name}"
        if isinstance(value, re.Pattern):
            return f"R:{value.pattern!r}:{value.flags}"
        if isinstance(value, logging.Logger):
            return "L:" + value.name
        if isinstance(value, _BUILTIN):
            return f"B:{getattr(value, '__module__', None)}.{getattr(value, '__qualname__', type(value).__name__)}"
        if id(value) in self._open:
            return "^"
        self._open.add(id(value))
        try:
            return self._nested(value)
        finally:
            self._open.discard(id(value))

    def _nested(self, value: object) -> str:
        kind = type(value)
        if kind in (list, tuple):
            items = list(map(self.of, value))  # type: ignore[call-overload]
            return kind.__name__ + "[" + ",".join(items if self.ordered else sorted(items)) + "]"
        if kind in (set, frozenset):
            return kind.__name__ + "{" + ",".join(sorted(map(self.of, value))) + "}"  # type: ignore[call-overload]
        if isinstance(value, dict):
            items = [self.of(key) + ":" + self.of(item) for key, item in value.items()]
            return f"D:{self.ref(kind)}{{" + ",".join(items if self.ordered else sorted(items)) + "}"
        if isinstance(value, types.FunctionType):
            cells = [_contents(cell) for cell in value.__closure__ or ()]
            return f"F:{value.__module__}.{value.__qualname__}" + self.of([value.__defaults__, value.__kwdefaults__, cells, value.__dict__])
        if isinstance(value, (staticmethod, classmethod)):
            return f"{kind.__name__}({self.of(value.__func__)})"
        if isinstance(value, property):
            return "P" + self.of([value.fget, value.fset, value.fdel])
        if isinstance(value, types.MethodType):
            return "BM" + self.of([value.__func__, value.__self__])
        state = dict(getattr(value, "__dict__", {}))
        for owner in kind.__mro__:
            for slot in getattr(owner, "__slots__", ()):
                if slot not in ("__dict__", "__weakref__") and hasattr(value, slot):
                    state[slot] = getattr(value, slot)
        return f"O:{self.ref(kind)}" + self.of(state)


def _contents(cell: types.CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return "<empty cell>"


def snapshot(unordered: frozenset[str] = frozenset()) -> dict[str, str]:
    """Digest per module global and class attribute: `module:name` and `module:Class.attr`. Classes made at
    import under one qualified name share keys, digested as a set. Paths under the checkouts sqlglot and the
    test helpers were imported from, or the working directory, digest relative to them."""
    canon = _Canon(_roots())
    found: dict[str, list[str]] = {}

    def put(key: str, value: object) -> None:
        canon.ordered = key not in unordered
        found.setdefault(key, []).append(_sha(canon.of(value)))

    for name in sorted(name for name in sys.modules if _own(name)):
        for attr, value in list(vars(sys.modules[name]).items()):
            if attr.startswith("__") or isinstance(value, types.ModuleType):
                continue
            if isinstance(value, type) and _own(value.__module__):
                canon.ref(value)
            else:
                put(f"{name}:{attr}", value)
    index = 0
    while index < len(canon.classes):
        cls = canon.classes[index]
        index += 1
        base = f"{cls.__module__}:{cls.__qualname__}"
        put(base + ".__bases__", [canon.ref(each) for each in cls.__bases__])
        for attr, value in list(vars(cls).items()):
            if attr not in _SKIPPED:
                put(f"{base}.{attr}", value)
    return {key: digests[0] if len(digests) == 1 else _sha(",".join(sorted(digests))) for key, digests in found.items()}


def _roots() -> list[str]:
    """Checkout directories, longest first: a warm probe child has modules from the pinned checkout and the
    patched tree."""
    found = {os.getcwd()}
    for name in ("sqlglot", "tests.helpers"):
        location = getattr(sys.modules.get(name), "__file__", None)
        if location:
            found |= {os.path.dirname(os.path.dirname(path)) for path in (location, os.path.realpath(location))}
    return sorted(found, key=len, reverse=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def record(first: tuple[str, ...] = ()) -> tuple[dict[str, str], dict[str, str]]:
    """After importing `first`, then everything: every key digested in its order, and every key digested
    unordered. Probes import in the warm parent's order, which can fill registries in another order."""
    for name in first:
        __import__(name)
    import_everything()
    ordered = snapshot()
    return ordered, snapshot(frozenset(ordered))


def combined(digests: dict[str, str], skip: set[str]) -> str:
    kept = sorted((key, digest) for key, digest in digests.items() if key not in skip)
    return hashlib.sha256(repr(kept).encode()).hexdigest()


def stable(runs: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, object]:
    """From (ordered, unordered) snapshots under different hash seeds: the keys to digest unordered or skip, and
    the combined digest the patched import must give."""
    ordered = [run[0] for run in runs]
    loose = [run[1] for run in runs]
    keys = set().union(*ordered)
    unordered = {key for key in keys if len({run.get(key) for run in ordered}) > 1}
    skip = {key for key in unordered if len({run.get(key) for run in loose}) > 1}
    unordered -= skip
    first = {key: (loose[0] if key in unordered else ordered[0]).get(key, "") for key in keys}
    return {"unordered": sorted(unordered), "skip": sorted(skip), "digest": combined(first, skip)}


def matches(pinned: dict) -> bool:
    """Whether the state imported in this process digests as the pinned import's; call before running queries."""
    try:
        import_everything()
        return combined(snapshot(frozenset(pinned["unordered"])), set(pinned["skip"])) == pinned["digest"]
    except Exception:  # noqa: BLE001 - a state that cannot be digested is not the pinned one
        return False
