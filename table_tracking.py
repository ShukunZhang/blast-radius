"""A dict that records which keys are looked up in a class-level table, for `collect_classes.py`.

Every way of reading the whole table (iteration, `keys()`, copies, `|`)
records `ALL`, so a key absent from the record was never read through it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

ALL = "*"
SEEN: set[tuple[str, str, str]] = set()


def label(key: object) -> str:
    """The name a dict literal uses for `key`: the string, the class name, or the enum member name."""
    if isinstance(key, str):
        return key
    if isinstance(key, type):
        return key.__name__
    name = getattr(key, "name", None)
    return name if isinstance(name, str) else repr(key)


class TrackedTable(dict):  # type: ignore[type-arg]
    owner = ""
    attr = ""

    def _note(self, key: str) -> None:
        SEEN.add((self.owner, self.attr, key))

    def get(self, key: Any, default: Any = None) -> Any:
        self._note(label(key))
        return dict.get(self, key, default)

    def __getitem__(self, key: Any) -> Any:
        self._note(label(key))
        return dict.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        self._note(label(key))
        return dict.__contains__(self, key)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._note(label(key))
        return dict.setdefault(self, key, default)

    def pop(self, key: Any, *default: Any) -> Any:
        self._note(label(key))
        return dict.pop(self, key, *default)

    def __iter__(self):  # type: ignore[no-untyped-def]
        self._note(ALL)
        return dict.__iter__(self)

    def keys(self):  # type: ignore[no-untyped-def]
        self._note(ALL)
        return dict.keys(self)

    def items(self):  # type: ignore[no-untyped-def]
        self._note(ALL)
        return dict.items(self)

    def values(self):  # type: ignore[no-untyped-def]
        self._note(ALL)
        return dict.values(self)

    def copy(self) -> dict:  # type: ignore[type-arg]
        self._note(ALL)
        return dict(dict.items(self))

    def __or__(self, other: Any) -> Any:
        self._note(ALL)
        return dict(dict.items(self)) | other

    def __ror__(self, other: Any) -> Any:
        self._note(ALL)
        return other | dict(dict.items(self))

    def __eq__(self, other: object) -> bool:
        self._note(ALL)
        return dict.__eq__(self, other)

    def __repr__(self) -> str:
        self._note(ALL)
        return dict.__repr__(self)

    def __reduce_ex__(self, protocol: Any) -> Any:
        self._note(ALL)
        return dict, (dict(dict.items(self)),)

    def __copy__(self) -> dict:  # type: ignore[type-arg]
        return self.copy()

    def __deepcopy__(self, memo: dict) -> dict:  # type: ignore[type-arg]
        import copy

        self._note(ALL)
        return copy.deepcopy(dict(dict.items(self)), memo)

    __hash__ = None  # type: ignore[assignment]


def subclasses(roots: Iterable[type]) -> list[type]:
    found: list[type] = []
    pending = list(roots)
    while pending:
        cls = pending.pop()
        if cls in found:
            continue
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def wrap_tables(classes: Iterable[type], key_of: Callable[[type], str | None]) -> dict[str, list[str]]:
    """Replace each upper-case dict defined on these classes with a tracked copy; returns what was wrapped."""
    wrapped: dict[str, list[str]] = {}
    for cls in classes:
        owner = key_of(cls)
        if owner is None:
            continue
        for attr, value in list(vars(cls).items()):
            if attr.isupper() and type(value) is dict:
                table = TrackedTable(value)
                table.owner, table.attr = owner, attr
                setattr(cls, attr, table)
                wrapped.setdefault(owner, []).append(attr)
    return wrapped
