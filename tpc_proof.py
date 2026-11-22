"""Drop test_execute_tpch/tpcds when a patch provably leaves them passing.

Both tests compare DuckDB with sqlglot's Python executor on fixture queries.
maps/tpc.json holds, from the pinned commit where they pass, digests per
executed query of its DuckDB SQL, its plan and its result (tpc_fingerprint),
the lines each query runs while planned and while executed (tpc_queries),
the lines every query shares, the names the post-planning code reads, and
per query the names its compiled code can look up in the executor's ENV.
tpc_edits maps the patch onto those lines; the probe then skips queries that
run no edited line, checks the plan of those that run one only while being
planned (only as far as the last optimizer rule running one, when no later
step does), and executes the rest, comparing results with the pinned ones.
An edited line that ran at import is also judged by what the import left:
the probe digests that state (import_state) before any query, and executes
every query unless it matches the pinned one.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from diffutil import DiffApplyError, FileDiff, apply_hunks, parse_patch
from import_state import state_path
from probe_small import run_patched
from tpc_edits import FULL, PLAN_ALL, Reach, file_reach
from tpc_fingerprint import CHECK, EXECUTE, POST, SKIP, STAGED

TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds",
}
TPC_TABLES = Path(os.environ.get("TPC_TABLES", "/app/tpc_tables.pickle"))
# Measured CPU of a probe: forking and importing sqlglot; loading, building and shipping the tables a query
# reads, or all of them; parsing a query the test only ships. Per-query plan and run CPU come from the map.
PROBE_BASE_S = 0.25
TABLES_S = 0.35
ALL_TABLES_S = 1.0
PARSE_S = 0.01
# A probe that executes queries sometimes finds a break, usually within its first queries.
EXECUTE_RISK = 0.15
# Importing the rest of sqlglot and digesting its state, for an edit to code run at import; how often that state
# then differs, so every query runs.
STATE_S = 2.0
STATE_RISK = 0.5
PROBE_LIMIT_S = 60.0
_GUARDED_TESTS = {"tests/test_executor.py", "tests/helpers.py", "tests/__init__.py", "tests/conftest.py"}
_INERT_SUFFIXES = (".md", ".rst", ".txt")


@dataclass
class Plan:
    actions: list[int]
    full: bool
    needs_probe: bool
    cpu_s: float
    # An edited line ran at import: the import state must digest as pinned, or every query runs. cpu_s leaves out
    # the STATE_S a probe spends once on it.
    state: bool = False


class TpcProof:
    def __init__(self, path: Path, root: Path) -> None:
        self.root = root
        self.suites: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        state = state_path(path)
        self.state: dict | None = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
        self._shared: dict[str, dict[str, set[int]]] = {}
        self._imported: dict[str, dict[str, set[int]]] = {}

    def drop_passing(
        self, patch_text: str, selected: set[int], tests: dict[int, dict], cost: Callable[[set[int]], float]
    ) -> set[int]:
        """Selected without the TPC tests proven to pass, when proving costs less than running them."""
        candidates = {
            number: TESTS[tests[number]["id"]] for number in selected if tests[number]["id"] in TESTS
        }
        candidates = {number: suite for number, suite in candidates.items() if suite in self.suites}
        if not candidates or cost(selected) - cost(selected - set(candidates)) <= PROBE_BASE_S:
            return selected
        files = parse_patch(patch_text)
        plans = {suite: self._plan(files, suite) for suite in set(candidates.values())}
        chosen = _most_saving(candidates, plans, selected, cost)
        if not chosen:
            return selected
        probing = {suite: plans[suite] for suite in set(chosen.values()) if plans[suite].needs_probe}  # type: ignore[union-attr]
        proven = set(chosen.values()) - set(probing)
        if probing:
            expected = {suite: (self.suites[suite]["digests"], plan.actions, plan.full, plan.state) for suite, plan in probing.items()}
            found = run_patched(self.root, patch_text, lambda: _passing(expected, self.state), PROBE_LIMIT_S)
            if isinstance(found, list):
                proven |= set(found) & set(probing)
        return selected - {number for number, suite in chosen.items() if suite in proven}

    def _plan(self, files: list[FileDiff], suite: str) -> Plan | None:
        """What the probe must do per query for the test to count as passing; None when it cannot stand in."""
        data = self.suites[suite]
        digests: list = data["digests"]
        names = set(data["names"])
        full = plan_all = needs_probe = state = False
        plan_mask = run_mask = 0
        last: dict[int, int] = {}
        for file_diff in files:
            path = file_diff.path or ""
            if file_diff.binary or (file_diff.old_path is not None and file_diff.new_path != file_diff.old_path):
                return None
            if not path.startswith("sqlglot/"):
                if path in _GUARDED_TESTS or path.startswith("tests/fixtures/optimizer/tpc-"):
                    return None
                if not (path.startswith("tests/") or path.endswith(_INERT_SUFFIXES)):
                    return None
                continue
            if file_diff.old_path is None:
                continue
            reach = self._file_reach(file_diff, names) if path.endswith(".py") else None
            if reach is None:
                full = needs_probe = True
                continue
            needs_probe |= reach.needs_probe
            full |= reach.level == FULL
            plan_all |= reach.level == PLAN_ALL
            env_names = data["queries"].get("env")
            full |= bool(reach.env_keys) and env_names is None
            for key in reach.env_keys:
                run_mask |= (env_names or {}).get(key, 0)
            shared = self._lines(self._shared, suite, "shared").get(path, set())
            imported = self._lines(self._imported, suite, "imported").get(path, set())
            plan_lines = data["queries"]["plan"].get(path, {})
            run_lines = data["queries"]["run"].get(path, {})
            stage_lines = data["queries"].get("stage", {}).get(path, {})
            for start, end in reach.spans:
                for line in range(start, end + 1):
                    full |= line in shared
                    state |= line in imported
                    hit = plan_lines.get(str(line), 0)
                    plan_mask |= hit
                    run_mask |= run_lines.get(str(line), 0)
                    if hit:
                        _end_stages(last, hit, stage_lines.get(str(line), {}))
        full |= state and self.state is None
        state &= not full
        actions = []
        for index, entry in enumerate(digests):
            if full or (run_mask >> index) & 1:
                actions.append(EXECUTE if entry is not None else CHECK)
            elif plan_all or ((plan_mask >> index) & 1):
                actions.append(_check(entry, POST if plan_all else last.get(index, POST)))
            else:
                actions.append(SKIP)
        needs_probe |= state or any(action != SKIP for action in actions)
        cpu = _estimate(actions, full, digests) if needs_probe else 0.0
        if state:
            every = _estimate(_every(digests), True, digests)
            cpu += STATE_RISK * every
            if cpu + STATE_S >= every:
                return Plan(_every(digests), True, True, every)
        return Plan(actions, full, needs_probe, cpu, state)

    def _file_reach(self, file_diff: FileDiff, names: set[str]) -> Reach | None:
        path = file_diff.old_path or ""
        try:
            old_text = (self.root / path).read_text(encoding="utf-8")
            return file_reach(ast.parse(old_text), ast.parse(apply_hunks(old_text, file_diff.hunks)), file_diff.hunks, names, path)
        except (OSError, SyntaxError, DiffApplyError, ValueError):
            return None

    def _lines(self, cache: dict[str, dict[str, set[int]]], suite: str, kind: str) -> dict[str, set[int]]:
        if suite not in cache:
            cache[suite] = {path: set(lines) for path, lines in self.suites[suite]["queries"].get(kind, {}).items()}
        return cache[suite]


def _end_stages(last: dict[int, int], hit: int, by_stage: dict[str, int]) -> None:
    """Raise each hit query's last edited stage to this line's; POST where the line's stage is unknown."""
    covered = 0
    for stage, mask in by_stage.items():
        covered |= mask
        _raise_to(last, hit & mask, int(stage))
    _raise_to(last, hit & ~covered, POST)


def _raise_to(last: dict[int, int], mask: int, stage: int) -> None:
    index = 0
    while mask:
        if mask & 1:
            last[index] = max(last.get(index, 0), stage)
        mask >>= 1
        index += 1


def _check(entry: dict | None, stage: int) -> int:
    """The cheapest check of a query whose edited lines all end by `stage`."""
    if entry is None or stage >= POST or entry.get("stages", [None] * POST)[stage] is None:
        return CHECK
    return STAGED + stage


def _estimate(actions: list[int], full: bool, digests: list) -> float:
    """Expected CPU of a probe running these actions, risk of finding a break included."""
    cpu = PROBE_BASE_S + (ALL_TABLES_S if full else TABLES_S if EXECUTE in actions else 0.0)
    executing = 0.0
    for action, entry in zip(actions, digests):
        if action == SKIP:
            continue
        if entry is None:
            cpu += PARSE_S
        elif action == CHECK:
            cpu += entry["check_s"]
        elif action >= STAGED:
            cpu += entry["stage_s"][action - STAGED]
        else:
            executing += entry["execute_s"]
    return cpu + executing * (1 + EXECUTE_RISK)


def _most_saving(
    candidates: dict[int, str], plans: dict[str, Plan | None], selected: set[int], cost: Callable[[set[int]], float]
) -> dict[int, str]:
    """The provable TPC tests whose dropping saves the most beyond the probe's CPU; empty when none pays."""
    provable = [number for number, suite in candidates.items() if plans[suite] is not None]
    best: dict[int, str] = {}
    best_gain = 0.0
    total = cost(selected)
    for size in range(1, len(provable) + 1):
        for numbers in combinations(provable, size):
            chosen = [plans[candidates[number]] for number in numbers]
            probe = sum(plan.cpu_s for plan in chosen) + (STATE_S if any(plan.state for plan in chosen) else 0.0)  # type: ignore[union-attr]
            gain = total - cost(selected - set(numbers)) - probe
            if gain > best_gain:
                best, best_gain = {number: candidates[number] for number in numbers}, gain
    return best


def prepare_tables(root: Path) -> None:
    """Write TPC_TABLES from the pinned checkout, in a child that imports the test module."""
    code = (
        "import os, sys; from pathlib import Path; "
        f"sys.path[:0] = [{str(root)!r}, {str(Path(__file__).resolve().parent)!r}]; os.chdir({str(root)!r}); "
        f"import tpc_fingerprint; tpc_fingerprint.write_tables(Path({str(TPC_TABLES)!r}))"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, env={**os.environ, "SKIP_INTEGRATION": "0"})


def _every(digests: list) -> list[int]:
    return [EXECUTE if entry is not None else CHECK for entry in digests]


def _passing(expected: dict[str, tuple[list, list[int], bool, bool]], pinned_state: dict | None) -> list[str]:
    """The suites whose replay passes; where an edit ran at import, every query runs unless the state matches."""
    import import_state
    import tpc_fingerprint

    same = pinned_state is not None and any(state for *_, state in expected.values()) and import_state.matches(pinned_state)
    passing = []
    for suite, (entries, actions, full, state) in expected.items():
        if state and not same:
            actions, full = _every(entries), True
        if tpc_fingerprint.replay(suite, entries, actions, full, TPC_TABLES):
            passing.append(suite)
    return passing
