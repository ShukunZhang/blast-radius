"""Re-run only tiny candidate sets and drop the ones that pass.

A set whose longest test is under half a second costs about two seconds just
to open a worker. Confirming those tests in-process costs less than that.
Anything slow, failed, or inconclusive stays selected.
"""

from __future__ import annotations

import gc
import importlib
import os
import select
import signal
import sys
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from warm_probe import WARM

MAX_SECONDS = 0.5
MAX_COUNT = 8
ALARM_S = 3.0


def should_probe(
    selected: set[int], seconds: dict[int, float], patch_text: str, heavy: set[int]
) -> bool:
    """Re-run a tiny product edit only when confirmation is cheaper than a worker.

    Fixture-only patches already name their test. Optimizer and executor tests
    rerun a whole SQL fixture, which costs more CPU than the 2s a worker adds,
    so those sets stay selected.
    """
    if not selected or len(selected) > MAX_COUNT or selected & heavy:
        return False
    if "\ndiff --git a/sqlglot/" not in "\n" + patch_text:
        return False
    return max(seconds.get(number, 0.0) for number in selected) <= MAX_SECONDS


def _overlay(base: Path, dest: Path, patched: dict[str, str]) -> None:
    for entry in base.iterdir():
        here = dest / entry.name
        if entry.name in patched and entry.is_file():
            here.write_text(patched[entry.name])
            continue
        nested = {
            path.split("/", 1)[1]: text
            for path, text in patched.items()
            if path.startswith(entry.name + "/")
        }
        if nested:
            here.mkdir()
            _overlay(entry, here, nested)
        else:
            here.symlink_to(entry, target_is_directory=entry.is_dir())


def _modifies_only(patch_text: str) -> bool:
    """Every file of the patch is an existing text file edited in place."""
    from diffutil import parse_patch

    return all(
        not file_diff.binary and file_diff.old_path is not None and file_diff.old_path == file_diff.new_path
        for file_diff in parse_patch(patch_text)
    )


def _apply(root: Path, patch_text: str) -> dict[str, str]:
    from diffutil import apply_hunks, parse_patch

    written: dict[str, str] = {}
    for file_diff in parse_patch(patch_text):
        path = file_diff.path
        if not path or file_diff.old_path is None:
            continue
        original = (root / path).read_text(encoding="utf-8")
        written[path] = apply_hunks(original, file_diff.hunks)
    return written


def _install_inline_pool() -> None:
    """Run pool work in this process so a patched module stays visible.

    Calls and results still go through pickle, as with a process pool, since
    pickling runs sqlglot code too (expressions pickle through serde).
    """
    import concurrent.futures
    import pickle

    class InlineExecutor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            future = concurrent.futures.Future()
            try:
                call, call_args, call_kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                future.set_result(pickle.loads(pickle.dumps(call(*call_args, **call_kwargs))))
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, *_args: object, **_kwargs: object) -> None:
            return None

    original = concurrent.futures.ProcessPoolExecutor
    concurrent.futures.ProcessPoolExecutor = InlineExecutor  # type: ignore[misc]
    for name, module in list(sys.modules.items()):
        if name.startswith("tests") and getattr(module, "ProcessPoolExecutor", None) is original:
            module.ProcessPoolExecutor = InlineExecutor  # type: ignore[attr-defined]


def _keep_failing(catalog: list[dict], numbers: set[int]) -> set[int]:
    """Run the tests. Return the numbers that did not cleanly pass."""
    by_number = {test["number"]: test for test in catalog}
    keep: set[int] = set()

    def _alarm(*_args: object) -> None:
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    _install_inline_pool()
    modules: dict[str, object] = {}
    prepared: set[type] = set()
    for number in numbers:
        test = by_number[number]
        file_name, class_name, method = test["id"].split("::")
        module_name = file_name[:-3].replace("/", ".")
        try:
            module = modules.get(module_name)
            if module is None:
                signal.setitimer(signal.ITIMER_REAL, 4.0)
                module = importlib.import_module(module_name)
                modules[module_name] = module
                signal.setitimer(signal.ITIMER_REAL, 0)
            signal.setitimer(signal.ITIMER_REAL, ALARM_S)
            case_cls = getattr(module, class_name)
            if case_cls not in prepared:
                case_cls.setUpClass()
                prepared.add(case_cls)
            result = unittest.TestResult()
            case_cls(method).run(result)
            if not result.wasSuccessful():
                keep.add(number)
        except Exception:
            keep.add(number)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return keep


def test_modules(test_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted({test_id.split("::")[0][:-3].replace("/", ".") for test_id in test_ids}))


def run_patched(
    root: Path, patch_text: str, job: Callable[[], object], limit_s: float, modules: tuple[str, ...] = ()
) -> object | None:
    """Run job in a child that imports the patched tree. None when it fails or overruns.

    A warm child inherits `modules` (the test modules the job imports) from the parent, which imports them once.
    """
    import json

    patched = _apply(root, patch_text)
    swaps = WARM.swaps(root, patched) if _modifies_only(patch_text) else None
    if swaps is not None and WARM.include(root, modules):
        swaps = WARM.swaps(root, patched)
    with tempfile.TemporaryDirectory(prefix="probe-") as folder:
        tree = Path(folder) / "tree"
        tree.mkdir()
        _overlay(root, tree, patched)
        result_path = Path(folder) / "result.json"
        # Frozen, the parent's objects stay out of the child's collections, which would otherwise scan
        # (and copy) all of them.
        gc.freeze()
        pid = os.fork()
        if pid:
            gc.unfreeze()
        else:
            try:
                with open(os.devnull, "w") as silence:
                    os.dup2(silence.fileno(), 1)
                    os.dup2(silence.fileno(), 2)
                os.setsid()
                os.chdir(tree)
                sys.path.insert(0, str(tree))
                os.environ["SKIP_INTEGRATION"] = "0"
                WARM.enter(root, tree, swaps)
                result_path.write_text(json.dumps(job()))
                os._exit(0)
            except BaseException:
                os._exit(1)
        if not _reaped(pid, limit_s):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None


def _reaped(pid: int, limit_s: float) -> bool:
    """Whether the child exited within limit_s, reaped if so; sleeps on a pidfd where the kernel has them."""
    deadline = time.monotonic() + limit_s
    try:
        handle = os.pidfd_open(pid)
    except (AttributeError, OSError):
        handle = None
    if handle is not None:
        try:
            select.select([handle], [], [], limit_s)
        finally:
            os.close(handle)
        return bool(os.waitpid(pid, os.WNOHANG)[0])
    while not os.waitpid(pid, os.WNOHANG)[0]:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def confirm(root: Path, patch_text: str, catalog: list[dict], selected: set[int]) -> set[int]:
    """Return selected with confirmed passes removed. A dead child keeps them all."""
    ids = [test["id"] for test in catalog if test["number"] in selected]
    keep = run_patched(
        root,
        patch_text,
        lambda: sorted(_keep_failing(catalog, selected)),
        8.0 + ALARM_S * len(selected),
        test_modules(ids),
    )
    if not isinstance(keep, list):
        return selected
    return set(keep) & selected
