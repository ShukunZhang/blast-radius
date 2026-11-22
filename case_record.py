"""Record, per fixture case of one test, the sqlglot function lines it runs and its CPU; for collect_cases.py.

A fixture case is one item a test draws from tests.helpers' fixture loaders.
Its lines are those run while the test works on that item: from the moment
the loader yields it until the next is drawn, and inside a subTest titled as
check_file titles it, where check_file checks pooled results after the loop.
Every other line reaches all cases ("full"): function lines run at import,
by setUpClass, outside any case, or only while caches are cold, which the
first of two runs of the test fills. Reads of expression arg_types
(case_args.py) are split the same way.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from case_args import Reads, install
from case_proof import LOADERS

HERE = Path(__file__).resolve().parent
Lines = set[tuple[str, int]]
REST = -1


class _Cases:
    """Tracks which fixture case the running test works on and reports each change of case."""

    def __init__(self, on_switch: Callable[[int], None]) -> None:
        self.on_switch = on_switch
        self.calls: list[list[Any]] = []
        self.count = 0
        self.titles: dict[str, list[int]] = {}
        self.current = REST

    def reset(self) -> None:
        self.calls, self.count, self.titles = [], 0, {}
        self.switch(REST)

    def switch(self, key: int) -> None:
        self.current = key
        self.on_switch(key)

    def loader(self, original: Callable[[str], Iterator[Any]]) -> Callable[[str], Iterator[Any]]:
        def wrapper(filename: str) -> Iterator[Any]:
            call = [filename, 0]
            self.calls.append(call)
            for index, item in enumerate(original(filename)):
                key = self.count
                self.count += 1
                call[1] += 1
                if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], dict):
                    self.titles.setdefault(item[0].get("title") or f"{index + 1}, {item[1]}", []).append(key)
                self.switch(key)
                yield item
            self.switch(REST)

        return wrapper

    def sub_test(self, original: Callable[..., Any]) -> Callable[..., Any]:
        cases = self

        def wrapper(test: unittest.TestCase, *args: Any, **params: Any) -> Any:
            keys = cases.titles.get(args[0]) if args and isinstance(args[0], str) else None
            context = original(test, *args, **params)
            return context if keys is None else cases._within(context, keys[0] if len(keys) == 1 else REST)

        return wrapper

    @contextlib.contextmanager
    def _within(self, context: Any, key: int) -> Iterator[None]:
        with context:
            previous = self.current
            self.switch(key)
            try:
                yield
            finally:
                self.switch(previous)


def _prepare(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(HERE))
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    from probe_small import _install_inline_pool

    _install_inline_pool()


def _load(test_id: str, cases: _Cases) -> unittest.TestCase:
    """The test case, with its module's fixture loaders and subTest routed through cases."""
    file_name, class_name, method = test_id.split("::")
    module = importlib.import_module(file_name[:-3].replace("/", "."))
    helpers = importlib.import_module("tests.helpers")
    for name in LOADERS:
        wrapped = cases.loader(getattr(helpers, name))
        setattr(helpers, name, wrapped)
        if hasattr(module, name):
            setattr(module, name, wrapped)
    unittest.TestCase.subTest = cases.sub_test(unittest.TestCase.subTest)  # type: ignore[method-assign]
    case_cls = getattr(module, class_name)
    case_cls.setUpClass()
    return case_cls(method)


def _run(test: unittest.TestCase) -> None:
    result = unittest.TestResult()
    test.run(result)
    if not result.wasSuccessful():
        raise SystemExit(f"{test.id()} fails on the pinned commit: {result.errors + result.failures}")


def record_lines(root: Path, test_id: str) -> dict[str, Any]:
    """Cases drawn per loader call, per-case lines and arg_types reads as hex bitmasks over case keys, and
    the lines and arg_types owners that reach every case."""
    reads = Reads()
    install(reads)
    _prepare(root)
    import tpc_queries
    from collect_tpc import _import_everything

    recorder = tpc_queries._Recorder(root)
    full: Lines = set()
    args_full: set[str] = set()
    sinks: list[tuple[Lines, set[str]]] = []
    route: dict[str, tuple[Lines, set[str]] | None] = {"case": None}

    def on_switch(key: int) -> None:
        if route["case"] is None:
            while key >= len(sinks):
                sinks.append((set(), set()))
        lines, owners = route["case"] or ((full, args_full) if key == REST else sinks[key])
        recorder.switch(lines)
        reads.bucket = owners

    cases = _Cases(on_switch)
    recorder.switch(full)
    reads.bucket, reads.root, reads.sites = args_full, root.resolve(), {}
    _import_everything(root)
    derived, reads.sites = reads.sites, None
    test = _load(test_id, cases)
    used: dict[int, set[str]] = {}
    _hook_classes(cases, used, route)
    warm: tuple[Lines, set[str]] = (set(), set())
    route["case"] = warm
    cases.reset()
    _run(test)
    route["case"] = None
    sinks.clear()
    cases.reset()
    _run(test)
    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
    reads.bucket = None
    full |= warm[0] - set().union(*(lines for lines, _owners in sinks))
    args_full |= warm[1] - set().union(*(owners for _lines, owners in sinks))
    return {
        "calls": cases.calls,
        "lines": _line_masks([lines for lines, _owners in sinks]),
        "full": tpc_queries._by_path(full),
        "classes": _class_masks(used, cases.count),
        "args": _owner_masks([owners for _lines, owners in sinks]),
        "args_full": sorted(args_full),
        "args_derived": {owner: sorted(sites) for owner, sites in derived.items()},
    }


def _line_masks(sinks: list[Lines]) -> dict[str, dict[str, str]]:
    masks: dict[str, dict[str, int]] = {}
    for key, lines in enumerate(sinks):
        for path, line in lines:
            by_line = masks.setdefault(path, {})
            by_line[str(line)] = by_line.get(str(line), 0) | (1 << key)
    return {path: {line: format(mask, "x") for line, mask in by_line.items()} for path, by_line in masks.items()}


def _owner_masks(sinks: list[set[str]]) -> dict[str, str]:
    masks: dict[str, int] = {}
    for key, owners in enumerate(sinks):
        for owner in owners:
            masks[owner] = masks.get(owner, 0) | (1 << key)
    return {owner: format(mask, "x") for owner, mask in masks.items()}


def _hook_classes(cases: _Cases, used: dict[int, set[str]], route: dict[str, Any]) -> None:
    """Note, per case, every class in the MRO of the dialect components it used (as collect_classes per test)."""
    from collect_classes import _key, hook_entry_points, used_classes

    def note(obj: object) -> None:
        if route["case"] is None:
            bucket = used.setdefault(cases.current, set())
            bucket.update(key for key in map(_key, used_classes(obj)) if key is not None)

    hook_entry_points(note)


def _class_masks(used: dict[int, set[str]], count: int) -> dict[str, str]:
    """Per class, a hex bitmask of the cases that used it; a class used outside any case marks them all."""
    masks: dict[str, int] = {}
    for key, classes in used.items():
        bits = (1 << count) - 1 if key == REST else 1 << key
        for name in classes:
            masks[name] = masks.get(name, 0) | bits
    return {name: format(mask, "x") for name, mask in masks.items()}


def record_seconds(root: Path, test_id: str) -> dict[str, Any]:
    """CPU per case and outside cases on a warm run, and of importing and setting the test up."""
    from case_proof import preload

    preload()
    start = time.process_time()
    _prepare(root)
    spent: dict[int, float] = {}
    clock = {"key": REST, "at": time.process_time()}

    def on_switch(key: int) -> None:
        now = time.process_time()
        spent[clock["key"]] = spent.get(clock["key"], 0.0) + now - clock["at"]
        clock.update(key=key, at=now)

    cases = _Cases(on_switch)
    test = _load(test_id, cases)
    setup_s = time.process_time() - start
    _run(test)
    spent.clear()
    cases.reset()
    _run(test)
    on_switch(REST)
    return {
        "setup_s": round(setup_s, 3),
        "rest_s": round(spent.pop(REST, 0.0), 3),
        "case_s": [round(spent.get(key, 0.0), 4) for key in range(cases.count)],
    }
