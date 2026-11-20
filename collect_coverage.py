"""Collect per-test line coverage of the pinned sqlglot commit.

The optimizer and executor suites ship work to spawned processes. Coverage is
recorded in this process, so those pools are replaced with an inline executor
before pytest imports the tests. Run from a checkout of eb11aec with the
sqlglot package importable.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import coverage


class _InlineFuture:
    def __init__(self, fn, args, kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._done = False
        self._result = None
        self._error: BaseException | None = None

    def result(self, timeout=None):
        self._run()
        if self._error is not None:
            raise self._error
        return self._result

    def exception(self, timeout=None):
        self._run()
        return self._error

    def _run(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:  # noqa: BLE001 - mirror a real future
            self._error = exc


class _Future:
    """concurrent.futures.Future-compatible result of inline execution."""

    def __init__(self) -> None:
        import concurrent.futures

        self._inner: concurrent.futures.Future = concurrent.futures.Future()

    def result(self, timeout=None):
        return self._inner.result(timeout=timeout)

    def exception(self, timeout=None):
        return self._inner.exception(timeout=timeout)

    def set_result(self, value) -> None:
        self._inner.set_result(value)

    def set_exception(self, exc) -> None:
        self._inner.set_exception(exc)

    def done(self) -> bool:
        return self._inner.done()


class InlinePool:
    """Run pool tasks in the current process so line coverage is attributed."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, /, *args, **kwargs):
        # A spawned worker pickles arguments on the way in and the return value
        # on the way out. Expr.__reduce__ calls serde.dump, so a dump() bug fails
        # pool tests that never call dump() themselves.
        import pickle

        def _roundtrip(value):
            try:
                return pickle.loads(pickle.dumps(value))
            except Exception:
                return value

        fn, args, kwargs = _roundtrip((fn, args, kwargs))
        future = _Future()
        try:
            future.set_result(_roundtrip(fn(*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future._inner

    def shutdown(self, wait=True, cancel_futures=False) -> None:
        return None


def _install_inline_pool() -> None:
    import concurrent.futures
    import concurrent.futures.process as process_mod

    concurrent.futures.ProcessPoolExecutor = InlinePool
    process_mod.ProcessPoolExecutor = InlinePool


def _write_map(path: Path, tests: list[str], file_lines: dict[str, dict[int, list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"BRC1")
        handle.write(struct.pack("<I", len(tests)))
        for nodeid in tests:
            raw = nodeid.encode("utf-8")
            handle.write(struct.pack("<H", len(raw)))
            handle.write(raw)
        handle.write(struct.pack("<I", len(file_lines)))
        for rel, lines in sorted(file_lines.items()):
            raw = rel.encode("utf-8")
            handle.write(struct.pack("<H", len(raw)))
            handle.write(raw)
            items = sorted(lines.items())
            handle.write(struct.pack("<I", len(items)))
            for lineno, ids in items:
                ids = sorted(set(ids))
                handle.write(struct.pack("<IH", lineno, len(ids)))
                handle.write(struct.pack(f"<{len(ids)}H", *ids))


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["COVERAGE_CORE"] = "ctrace"
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    _install_inline_pool()

    cov = coverage.Coverage(
        data_file=str(Path("/tmp/sg-coverage") / "data"),
        source=["sqlglot"],
        branch=False,
        config_file=False,
    )
    cov.set_option("run:core", "ctrace")
    cov.set_option("run:parallel", False)
    cov.start()
    cov.switch_context("")

    import pytest

    failed: list[str] = []
    started = time.perf_counter()
    seen = 0

    class Plugin:
        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtest_protocol(self, item, nextitem):
            nonlocal seen
            cov.switch_context(item.nodeid)
            seen += 1
            if seen % 25 == 0:
                print(
                    f"[coverage] {seen} tests, {time.perf_counter() - started:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            yield

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.failed:
                failed.append(report.nodeid)

    print("[coverage] starting pytest", file=sys.stderr, flush=True)
    exit_code = pytest.main(
        [
            "tests",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "--ignore=tests/test_docs.py",
            "--ignore=tests/leakcheck.py",
            "--ignore=tests/gen_fixtures.py",
        ],
        plugins=[Plugin()],
    )
    cov.stop()
    data = cov.get_data()
    print(
        f"[coverage] pytest exit {exit_code}, failures {len(failed)}, "
        f"{time.perf_counter() - started:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    for nodeid in failed[:30]:
        print(f"  FAIL {nodeid}", file=sys.stderr)

    contexts = sorted(context for context in data.measured_contexts() if context)
    index = {nodeid: number for number, nodeid in enumerate(contexts)}
    file_lines: dict[str, dict[int, list[int]]] = {}
    measured = data.measured_files()
    for filename in measured:
        path = Path(filename)
        try:
            rel = path.resolve().relative_to(repo).as_posix()
        except ValueError:
            continue
        if not rel.startswith("sqlglot/") or not rel.endswith(".py"):
            continue
        by_line = data.contexts_by_lineno(filename)
        dest: dict[int, list[int]] = {}
        for lineno, names in by_line.items():
            ids = [index[name] for name in names if name in index]
            if ids:
                dest[int(lineno)] = ids
        if dest:
            file_lines[rel] = dest

    _write_map(out, contexts, file_lines)
    covered = sum(len(lines) for lines in file_lines.values())
    print(
        f"[coverage] wrote {out} tests={len(contexts)} files={len(file_lines)} lines={covered}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
