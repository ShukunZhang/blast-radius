"""Record which expression classes' own arg_types each fixture case reads; for case_record.py.

Expr.__init_subclass__ derives each class's required_args from the arg_types
it resolves, and sqlglot reads both only by iterating, membership, length and
items. So an edit to the arg_types dict a class body assigns changes what is
read through that dict, or through the required_args of a class resolving it,
and nothing else. Once sqlglot.expressions.core has run, every such dict and
set is replaced by a subclass reporting reads under the dict's owner, so the
reads of import-time code in other modules count too.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

CORE = "sqlglot.expressions.core"
_DICT_READS = (
    "__getitem__", "get", "__contains__", "__iter__", "__len__", "__reversed__", "keys", "values", "items",
    "copy", "__eq__", "__ne__", "__or__", "__ror__", "__repr__", "__reduce__", "__reduce_ex__",
)  # fmt: skip
_SET_READS = (
    "__iter__", "__contains__", "__len__", "__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__",
    "__sub__", "__rsub__", "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__", "copy",
    "difference", "intersection", "union", "symmetric_difference", "issubset", "issuperset", "isdisjoint",
    "__repr__", "__reduce__", "__reduce_ex__",
)  # fmt: skip


class Reads:
    """The owners read go into the current bucket; None drops them, and quiet while deriving required_args.

    While `sites` is set (at import), a read by code a sqlglot class body runs is
    kept as that class-body line instead: what the line builds is what depends
    on the read.
    """

    def __init__(self) -> None:
        self.bucket: set[str] | None = None
        self.quiet = False
        self.root: Path | None = None
        self.sites: dict[str, set[tuple[str, int]]] | None = None

    def note(self, owner: str) -> None:
        if self.bucket is None or self.quiet:
            return
        site = self._class_site() if self.sites is not None else None
        if site is None:
            self.bucket.add(owner)
        else:
            self.sites.setdefault(owner, set()).add(site)  # type: ignore[union-attr]

    def _class_site(self) -> tuple[str, int] | None:
        """The sqlglot file and line of the innermost class body running, unless a module body is nearer."""
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code.co_name == "<module>":
                return None
            names = frame.f_locals
            if names is not frame.f_globals and "__module__" in names and "__qualname__" in names:
                try:
                    relative = Path(frame.f_code.co_filename).resolve().relative_to(self.root).as_posix()  # type: ignore[arg-type]
                except ValueError:
                    return None
                return (relative, frame.f_lineno) if relative.startswith("sqlglot/") else None
            frame = frame.f_back
        return None


def install(reads: Reads) -> None:
    """Report reads of expression arg_types from when sqlglot.expressions.core is first imported."""
    if CORE in sys.modules:
        raise RuntimeError(f"{CORE} was imported before arg reads could be tracked")
    sys.meta_path.insert(0, _AfterCore(lambda core: _track_all(core, reads)))


class _AfterCore(importlib.abc.MetaPathFinder):
    def __init__(self, then: Callable[[ModuleType], None]) -> None:
        self.then = then

    def find_spec(self, name: str, path: Sequence[str] | None, target: ModuleType | None = None) -> Any:
        if name != CORE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        original, then = spec.loader.exec_module, self.then

        def exec_module(module: ModuleType) -> None:
            original(module)
            then(module)

        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def key_of(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _reporting(base: type, names: tuple[str, ...], reads: Reads) -> type:
    def wrap(name: str) -> Callable[..., Any]:
        original = getattr(base, name)

        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            reads.note(self.owner)
            return original(self, *args, **kwargs)

        method.__name__ = name
        return method

    return type(f"Reporting{base.__name__}", (base,), {"__slots__": ("owner",), **{name: wrap(name) for name in names}})


def _track_all(core: ModuleType, reads: Reads) -> None:
    expr: type = core.Expr
    reporting_dict = _reporting(dict, _DICT_READS, reads)
    reporting_set = _reporting(set, _SET_READS, reads)

    def track(cls: type) -> None:
        if type(cls.__dict__.get("arg_types")) is dict:
            tracked = reporting_dict(cls.__dict__["arg_types"])
            tracked.owner = key_of(cls)
            type.__setattr__(cls, "arg_types", tracked)
        owner = next((klass for klass in cls.__mro__ if "arg_types" in klass.__dict__), None)
        if owner is not None and type(cls.__dict__.get("required_args")) is set:
            tracked = reporting_set(cls.__dict__["required_args"])
            tracked.owner = key_of(owner)
            type.__setattr__(cls, "required_args", tracked)

    stack = [expr]
    while stack:
        cls = stack.pop()
        track(cls)
        stack.extend(cls.__subclasses__())
    derive = expr.__dict__["__init_subclass__"].__func__

    def init_subclass(cls: type, **kwargs: Any) -> None:
        previous, reads.quiet = reads.quiet, True
        try:
            derive(cls, **kwargs)
        finally:
            reads.quiet = previous
        track(cls)

    expr.__init_subclass__ = classmethod(init_subclass)  # type: ignore[assignment]
