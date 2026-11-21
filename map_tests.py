"""Build a baseline-only map of executed lines and fixture/source reads."""
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from inline_pool import InlinePool

ROOT = str(Path.cwd()) + "/"
EVENTS = sys.monitoring.events
TOOL = 3
lines = defaultdict(set)
reads = set()


def line(code, number):
    lines[code.co_filename[len(ROOT):]].add(number)
    return sys.monitoring.DISABLE


def start(code, offset):
    name = code.co_filename
    if name.startswith(ROOT + "sqlglot/") or name.startswith(ROOT + "tests/"):
        sys.monitoring.set_local_events(TOOL, code, EVENTS.LINE)
    return sys.monitoring.DISABLE


def audit(event, args):
    if event == "open" and isinstance(args[0], str):
        name = args[0]
        if "/fixtures/" in name or name.startswith("tests/fixtures/"):
            path = os.path.abspath(name)
            if path.startswith(ROOT):
                reads.add(path[len(ROOT):])


def snapshot():
    global lines, reads
    result = ({name: sorted(nums) for name, nums in lines.items()}, sorted(reads))
    lines, reads = defaultdict(set), set()
    sys.monitoring.restart_events()
    return result


class Capture:
    def __init__(self):
        self.results = {}
        self.collection = None
        self.common = {}

    def pytest_collection_finish(self):
        self.collection = snapshot()
        from sqlglot.dialects.dialect import Dialect
        original = Dialect.get_or_raise.__func__

        def get_or_raise(cls, dialect):
            result = original(cls, dialect)
            for ancestor in type(result).__mro__:
                module = ancestor.__module__
                if module.startswith("sqlglot.dialects."):
                    reads.add("dialect:" + module.rsplit(".", 1)[-1])
            return result

        Dialect.get_or_raise = classmethod(get_or_raise)
        for item in pytest_modules():
            if hasattr(item, "ProcessPoolExecutor"):
                item.ProcessPoolExecutor = InlinePool

    @pytest.hookimpl(wrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        if fixturedef.scope in ("class", "module", "session") and fixturedef.func.__module__ != 'builtins':
            before = snapshot()
            try:
                return (yield)
            finally:
                key = (fixturedef.scope, request.node.nodeid)
                self.common.setdefault(key, []).append(snapshot())
                for name, nums in before[0].items():
                    lines[name].update(nums)
                reads.update(before[1])
        else:
            return (yield)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item):
        snapshot()
        try:
            return (yield)
        finally:
            self.results[item.nodeid] = snapshot()


def pytest_modules():
    return [module for name, module in tuple(sys.modules.items()) if name.startswith("tests.")]


def main():
    sys.dont_write_bytecode = True
    capture = Capture()
    sys.monitoring.use_tool_id(TOOL, "impact")
    sys.monitoring.register_callback(TOOL, EVENTS.PY_START, start)
    sys.monitoring.register_callback(TOOL, EVENTS.LINE, line)
    sys.monitoring.set_events(TOOL, EVENTS.PY_START)
    sys.addaudithook(audit)
    cache = {}

    def record(function, args, kwargs, key):
        before = snapshot()
        try:
            payload = pickle.dumps(function(*args, **kwargs))
            covered, _ = snapshot()
            cache[key] = [payload.hex(), covered]
            return payload
        finally:
            for name, nums in before[0].items():
                lines[name].update(nums)
            reads.update(before[1])
            if "covered" in locals():
                for name, nums in covered.items():
                    lines[name].update(nums)

    InlinePool.recording = record
    status = pytest.main(["-q", "-p", "no:cacheprovider", *(sys.argv[2:] or ["tests"])],
                         plugins=[capture])
    sys.monitoring.set_events(TOOL, 0)
    result = {}
    for node, (covered, files) in capture.results.items():
        merged = {name: set(nums) for name, nums in covered.items()}
        paths = set(files)
        parts = node.split("::")
        for key in (("session", ""), ("module", parts[0]), ("class", "::".join(parts[:2]))):
            for extra_lines, extra_paths in capture.common.get(key, []):
                for name, numbers in extra_lines.items():
                    merged.setdefault(name, set()).update(numbers)
                paths.update(extra_paths)
        result[node] = [{name: sorted(nums) for name, nums in merged.items()}, sorted(paths)]
    if not result or capture.collection is None:
        raise SystemExit(status or 1)
    Path(sys.argv[1] + ".jobs").write_text(json.dumps(cache))
    Path(sys.argv[1]).write_text(json.dumps({"tests": result, "collection": capture.collection}))
    print(f"Mapped {len(result)} tests, pytest status {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
