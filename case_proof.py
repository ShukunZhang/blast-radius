"""Drop fixture-driven tests whose fixture cases a patch reaches all pass on the patched tree.

maps/cases.json (collect_cases.py) holds, per test that draws SQL fixture
cases, the sqlglot lines each case runs, the lines that reach every case,
and the CPU of each case. tpc_edits maps the patch onto pinned lines; an edit
outside function bodies reaches the cases case_attrs proves can read what it
binds, else every case. A test none of whose cases or shared lines run an
edited function line, with no other edit, is unchanged. Else a probe runs the
test on the patched tree with its fixture loaders drawing only the reached
cases, and the test is dropped when that run passes and costs less than it
saves.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import signal
import unittest
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analyze import module_name
from case_attrs import CaseCoverage, Mentions, arg_types_owner, attribute_cases, keys_of
from class_users import ClassUsers
from diffutil import DiffApplyError, FileDiff, apply_hunks, parse_patch
from probe_small import _install_inline_pool, run_patched, test_modules
from tpc_edits import file_reach

CASES_MAP = Path(os.environ.get("CASES_MAP", "/app/maps/cases.json"))
LOADERS = ("load_sql_fixture_pairs", "load_sql_fixtures")
# Forking the probe; a probe that runs cases finds a break part of the time, after spending part of its estimate.
PROBE_BASE_S = 0.15
RISK = 0.3
# About 3 in 10 probed tests fail on the patched tree, and then the probe saves nothing.
FAIL_P = 0.3
_INERT_SUFFIXES = (".md", ".rst", ".txt")


@dataclass
class FileReach:
    """Where an edit to one sqlglot file takes effect: pinned spans, and the edited lines outside function bodies."""

    spans: list[tuple[int, int]]
    outside: list[tuple[str, int]]
    trees: dict[str, ast.Module]
    everywhere: bool = False


Reach = dict[str, FileReach]


@dataclass
class CasePlan:
    test_id: str
    keys: list[int] | None
    cpu_s: float
    probe: bool = True
    # Modules whose import the patch may break or change: the probe imports them before any test.
    modules: set[str] = field(default_factory=set)


class CaseProof:
    def __init__(self, path: Path, root: Path, users: ClassUsers | None = None, mentions: Mentions | None = None) -> None:
        self.root = root
        self.users, self.mentions = users, mentions
        self.tests: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._full: dict[str, dict[str, set[int]]] = {}
        self._trees: dict[str, ast.Module | None] = {}

    def drop_passing(
        self, patch_text: str, selected: set[int], tests: dict[int, dict], cost: Callable[[set[int]], float]
    ) -> set[int]:
        """Selected without the fixture-driven tests proven to pass, when proving costs less than running them."""
        candidates = {number: tests[number]["id"] for number in selected if tests[number]["id"] in self.tests}
        if not candidates:
            return selected
        reach = self._reach(parse_patch(patch_text))
        if reach is None:
            return selected
        plans = {number: self._plan(test_id, reach) for number, test_id in candidates.items()}
        unchanged = {number for number, plan in plans.items() if not plan.probe}
        remaining = selected - unchanged
        probing = {number: plan for number, plan in plans.items() if number not in unchanged}
        chosen = _most_saving(probing, remaining, cost, self.tests)
        if not chosen:
            return remaining
        jobs = {plan.test_id: (self.tests[plan.test_id]["calls"], plan.keys) for plan in chosen.values()}
        modules = sorted(set().union(*(plan.modules for plan in chosen.values())))
        limit = 5.0 + 3.0 * sum(plan.cpu_s for plan in chosen.values())
        preload()
        passing = run_patched(self.root, patch_text, lambda: replay(jobs, modules), limit, test_modules(list(jobs)))
        if not isinstance(passing, list):
            return remaining
        return remaining - {number for number, plan in chosen.items() if plan.test_id in passing}

    def _reach(self, files: list[FileDiff]) -> Reach | None:
        """Per edited sqlglot file, the pinned line spans an edit reaches and whether it reaches every case."""
        reach: Reach = {}
        for file_diff in files:
            path = file_diff.path or ""
            if file_diff.binary or file_diff.new_path is None or file_diff.new_path != (file_diff.old_path or file_diff.new_path):
                return None
            if not path.startswith("sqlglot/"):
                if path.startswith("tests/") or not path.endswith(_INERT_SUFFIXES):
                    return None
                continue
            if file_diff.old_path is None:
                continue
            if not path.endswith(".py"):
                reach[path] = FileReach([], [], {}, everywhere=True)
                continue
            try:
                old_text = (self.root / path).read_text(encoding="utf-8")
                trees = {"old": ast.parse(old_text), "new": ast.parse(apply_hunks(old_text, file_diff.hunks))}
                found = file_reach(trees["old"], trees["new"], file_diff.hunks, set())
            except (OSError, SyntaxError, DiffApplyError, ValueError):
                return None
            reach[path] = FileReach(found.spans, found.outside, trees)
        return reach

    def _plan(self, test_id: str, reach: Reach) -> CasePlan:
        entry = self.tests[test_id]
        full = self._full_lines(test_id)
        every = False
        mask = 0
        modules: set[str] = set()
        for path, found in reach.items():
            lines = entry["lines"].get(path, {})
            shared = full.get(path, set())
            every |= found.everywhere
            for start, end in found.spans:
                for line in range(start, end + 1):
                    every |= line in shared
                    mask |= int(lines.get(str(line), "0"), 16)
            for side, line in found.outside:
                owner = arg_types_owner(path, found.trees, side, line) if "args" in entry else None
                if owner is not None:
                    readers, derived = self._arg_cases(test_id, owner, set())
                    if derived:
                        modules |= derived | {module_name(path)}
                else:
                    modules.add(module_name(path))
                    readers = self._attribute_readers(test_id, path, found.trees[side], line, side == "old")
                if readers is None:
                    every = True
                else:
                    mask |= sum(1 << key for key in readers)
        keys = None if every else [key for key in range(len(entry["case_s"])) if (mask >> key) & 1]
        if keys == [] and not modules:
            return CasePlan(test_id, [], 0.0, probe=False)
        chosen = entry["case_s"] if keys is None else [entry["case_s"][key] for key in keys]
        return CasePlan(test_id, keys, entry["rest_s"] + sum(chosen) * (1 + RISK), modules=modules)

    def _arg_cases(self, test_id: str, owner: str, seen: set[str]) -> tuple[set[int] | None, set[str]]:
        """Cases that read `owner`'s arg_types, directly or through what class bodies built from it at import
        (themselves arg_types, or attributes their readers read), and those class bodies' modules."""
        entry = self.tests[test_id]
        if owner in entry["args_full"]:
            return None, set()
        seen.add(owner)
        cases = keys_of(int(entry["args"].get(owner, "0"), 16))
        modules: set[str] = set()
        for path, line in entry.get("args_derived", {}).get(owner, []):
            tree = self._pinned_tree(path)
            if tree is None:
                return None, set()
            modules.add(module_name(path))
            copier = arg_types_owner(path, {"old": tree, "new": tree}, "old", line)
            if copier is None:
                found, more = self._attribute_readers(test_id, path, tree, line, False), set()
            elif copier in seen:
                continue
            else:
                found, more = self._arg_cases(test_id, copier, seen)
            if found is None:
                return None, set()
            cases |= found
            modules |= more
        return cases, modules

    def _pinned_tree(self, path: str) -> ast.Module | None:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse((self.root / path).read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                self._trees[path] = None
        return self._trees[path]

    def _attribute_readers(self, test_id: str, path: str, tree: ast.Module, line: int, original: bool) -> set[int] | None:
        if self.users is None or self.mentions is None:
            return None
        coverage = CaseCoverage(self.tests[test_id], self._full_lines(test_id))
        return attribute_cases(self.users, self.mentions, test_id, coverage, path, tree, line, original)

    def _full_lines(self, test_id: str) -> dict[str, set[int]]:
        if test_id not in self._full:
            self._full[test_id] = {path: set(lines) for path, lines in self.tests[test_id]["full"].items()}
        return self._full[test_id]


def _most_saving(
    plans: dict[int, CasePlan], selected: set[int], cost: Callable[[set[int]], float], tests: dict[str, dict[str, Any]]
) -> dict[int, CasePlan]:
    """The tests whose dropping saves more than probing them: the better of adding tests one by one and
    removing them one by one from all, since dropping several tests can free a worker none frees alone."""
    total = cost(selected)

    def gain(numbers: set[int]) -> float:
        if not numbers:
            return 0.0
        modules = {plans[number].test_id.split("::")[0]: plans[number].test_id for number in numbers}
        setup = PROBE_BASE_S + sum(tests[test_id]["setup_s"] for test_id in modules.values())
        probe = sum(plans[number].cpu_s for number in numbers)
        return (total - cost(selected - numbers)) * (1.0 - FAIL_P) - probe - setup

    adding: set[int] = set()
    while candidates := [number for number in plans if number not in adding]:
        best = max(candidates, key=lambda number: gain(adding | {number}))
        if gain(adding | {best}) <= gain(adding):
            break
        adding.add(best)
    removing = set(plans)
    while removing:
        worst = max(removing, key=lambda number: gain(removing - {number}))
        if gain(removing - {worst}) < gain(removing):
            break
        removing.discard(worst)
    chosen = max((adding, removing), key=gain)
    return {number: plans[number] for number in chosen} if gain(chosen) > 0.0 else {}


def preload() -> None:
    """Import the test modules' third-party dependencies once, so every forked probe inherits them."""
    for name in _PRELOADED:
        try:
            importlib.import_module(name)
        except ImportError:
            continue


_PRELOADED = ("pandas", "pandas.testing", "duckdb")


class _Draw:
    """Fixture loaders that draw only the allowed cases, counting every case as the recording did."""

    def __init__(self) -> None:
        self.allowed: set[int] | None = None
        self.calls: list[list[Any]] = []
        self.count = 0
        self.failed = False

    def loader(self, original: Callable[[str], Iterator[Any]]) -> Callable[[str], Iterator[Any]]:
        def wrapper(filename: str) -> Iterator[Any]:
            call = [filename, 0]
            self.calls.append(call)
            for item in original(filename):
                if self.failed:
                    return
                key = self.count
                self.count += 1
                call[1] += 1
                if self.allowed is None or key in self.allowed:
                    yield item

        return wrapper


class _StopDrawing(unittest.TestResult):
    """A result that stops the fixture draw at the first failing case: the test stays selected either way."""

    def __init__(self, draw: _Draw) -> None:
        super().__init__()
        self.draw = draw

    def addSubTest(self, test: unittest.TestCase, subtest: unittest.TestCase, err: Any) -> None:
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.draw.failed = True


def replay(jobs: dict[str, tuple[list[list[Any]], list[int] | None]], modules: list[str]) -> list[str]:
    """In the patched child: none when importing `modules` fails, else the tests that pass drawing only their
    listed cases and draw the cases recorded."""
    try:
        for name in modules:
            importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a module the patch breaks fails every test
        return []
    _install_inline_pool()
    draw = _Draw()
    helpers = importlib.import_module("tests.helpers")
    originals = {name: getattr(helpers, name) for name in LOADERS}
    passing: list[str] = []
    prepared: set[type] = set()

    def alarm(*_args: object) -> None:
        raise TimeoutError()

    signal.signal(signal.SIGALRM, alarm)
    for test_id, (calls, keys) in jobs.items():
        file_name, class_name, method = test_id.split("::")
        try:
            module = importlib.import_module(file_name[:-3].replace("/", "."))
            for name, original in originals.items():
                wrapped = draw.loader(original)
                setattr(helpers, name, wrapped)
                if hasattr(module, name):
                    setattr(module, name, wrapped)
            case_cls = getattr(module, class_name)
            if case_cls not in prepared:
                case_cls.setUpClass()
                prepared.add(case_cls)
            draw.allowed, draw.calls, draw.count = (None if keys is None else set(keys)), [], 0
            draw.failed = False
            signal.setitimer(signal.ITIMER_REAL, 60.0)
            result = _StopDrawing(draw)
            case_cls(method).run(result)
            if result.wasSuccessful() and not draw.failed and draw.calls == calls:
                passing.append(test_id)
        except Exception:  # noqa: BLE001 - a test that cannot be run stays selected
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return passing
