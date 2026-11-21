"""Record, per test, which dialect component classes it used on the pinned commit.

Class-level tables such as `DorisParser.FUNCTIONS` run at import, so line
coverage credits them to whichever test imported the module first. The tests
that can observe such an edit are the ones that used that class or a subclass.
This hooks the constructors and entry points of parsers, generators,
tokenizers, dialects and the type annotator, and stores every class in the MRO
of each object a test used, including the classes of its dialect.

It also stores the MRO of every recorded class, and which expression classes
each generator class looked up in its dispatch table. That table is built once
per generator class from `TRANSFORMS` and kept across tests, so an entry is
observed exactly by the tests that dispatched its expression class.

    python collect_classes.py SQLGLOT_CHECKOUT maps/classes.json
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.process as process_module
import functools
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path


def _key(cls: type) -> str | None:
    module = getattr(cls, "__module__", "") or ""
    if not module.startswith("sqlglot"):
        return None
    return f"{module}:{cls.__qualname__}"


def _import_everything() -> None:
    """Create every dialect class up front, so its tables are tracked before any test reads them."""
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 - optional extras are not dependencies of these tests
            continue


def main() -> None:
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    os.environ.setdefault("SKIP_INTEGRATION", "0")

    from inline_pool import InlinePool

    concurrent.futures.ProcessPoolExecutor = InlinePool  # type: ignore[misc]
    process_module.ProcessPoolExecutor = InlinePool  # type: ignore[misc]

    from sqlglot.dialects.dialect import Dialect
    from sqlglot.generator import Generator
    from sqlglot.optimizer.annotate_types import TypeAnnotator
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer

    import sqlglot.generator as generator_module

    current: set[str] = set()
    dispatched: set[tuple[str, str]] = set()
    types: dict[str, type] = {}

    class Tracked(dict):  # type: ignore[type-arg]
        owner = ""

        def get(self, key, default=None):  # type: ignore[no-untyped-def]
            dispatched.add((self.owner, getattr(key, "__name__", repr(key))))
            return dict.get(self, key, default)

    build_dispatch = generator_module._build_dispatch

    def tracked_dispatch(cls: type) -> dict:  # type: ignore[type-arg]
        table = Tracked(build_dispatch(cls))
        table.owner = _key(cls) or ""
        return table

    generator_module._build_dispatch = tracked_dispatch
    generator_module._DISPATCH_CACHE.clear()

    import table_tracking

    _import_everything()
    tables = table_tracking.subclasses([Parser, Generator, Tokenizer, Dialect])
    tracked = table_tracking.wrap_tables(tables, _key)
    for cls in tables:
        key = _key(cls)
        if key is not None:
            types[key] = cls

    def note(obj: object) -> None:
        classes = list(type(obj).__mro__)
        dialect = obj if isinstance(obj, Dialect) else getattr(obj, "dialect", None)
        if isinstance(dialect, Dialect):
            for attribute in ("tokenizer_class", "parser_class", "generator_class"):
                component = getattr(type(dialect), attribute, None)
                if isinstance(component, type):
                    classes.extend(component.__mro__)
            classes.extend(type(dialect).__mro__)
        for cls in classes:
            key = _key(cls)
            if key is not None:
                current.add(key)
                types[key] = cls

    def wrap(owner: type, name: str) -> None:
        original = owner.__dict__.get(name)
        if original is None:
            return

        @functools.wraps(original)
        def hooked(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            note(self)
            return original(self, *args, **kwargs)

        setattr(owner, name, hooked)

    for owner, names in (
        (Parser, ("__init__", "parse", "parse_into")),
        (Generator, ("__init__", "generate")),
        (Tokenizer, ("__init__", "tokenize")),
        (Dialect, ("__init__", "parse", "parse_into", "generate", "tokenize", "parser", "generator")),
        (TypeAnnotator, ("__init__",)),
    ):
        for name in names:
            wrap(owner, name)

    import pytest

    used: dict[str, list[str]] = {}
    lookups: dict[str, set[tuple[str, str]]] = {}
    keyed: dict[str, set[tuple[str, str, str]]] = {}

    class Plugin:
        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtest_protocol(self, item, nextitem):  # type: ignore[no-untyped-def]
            current.clear()
            dispatched.clear()
            table_tracking.SEEN.clear()
            yield
            used[item.nodeid] = sorted(current)
            lookups[item.nodeid] = set(dispatched)
            keyed[item.nodeid] = set(table_tracking.SEEN)

    pytest.main(
        [
            "tests", "-q", "--tb=no", "-p", "no:cacheprovider",
            "--ignore=tests/test_docs.py", "--ignore=tests/leakcheck.py", "--ignore=tests/gen_fixtures.py",
        ],
        plugins=[Plugin()],
    )
    owners = {owner for pairs in lookups.values() for owner, _name in pairs if owner}
    classes = sorted({key for keys in used.values() for key in keys} | owners | set(tracked))
    index = {key: number for number, key in enumerate(classes)}
    mro = {
        str(index[key]): [index[k] for k in map(_key, types[key].__mro__) if k in index]
        for key in classes
        if key in types
    }
    nodeids = list(used)
    position = {nodeid: number for number, nodeid in enumerate(nodeids)}
    readers: dict[str, dict[str, list[int]]] = {}
    for nodeid, pairs in lookups.items():
        for owner, name in pairs:
            if owner:
                readers.setdefault(str(index[owner]), {}).setdefault(name, []).append(position[nodeid])
    table_readers: dict[str, dict[str, dict[str, list[int]]]] = {}
    for nodeid, triples in keyed.items():
        for owner, attr, key in triples:
            entries = table_readers.setdefault(str(index[owner]), {}).setdefault(attr, {})
            entries.setdefault(key, []).append(position[nodeid])
    payload = {
        "classes": classes,
        "tests": {nodeid: [index[k] for k in used[nodeid]] for nodeid in nodeids},
        "mro": mro,
        "dispatch": readers,
        "tracked": {str(index[owner]): attrs for owner, attrs in tracked.items()},
        "tables": table_readers,
    }
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"[classes] tests={len(used)} classes={len(classes)}", file=sys.stderr)


if __name__ == "__main__":
    main()
