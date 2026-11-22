"""Record, per TPC query, the sqlglot function lines it runs; collect_tpc.py stores them in maps/tpc.json.

The probe (tpc_fingerprint.replay) skips a query that runs none of the edited
lines, checks the plan of one that runs them only while it is parsed,
pickled, transpiled, optimized or planned ("plan" lines), and executes one
that runs them afterwards ("run" lines). Lines that build and ship the
tables, or run for the first query only and then serve later ones from a
cache, are "shared": an edit there reaches every query. For plan lines it
also records the last stage of tpc_fingerprint._plan running them ("stage"),
so a query whose edited lines all end by optimizer rule k is checked after
its first k rules.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
Lines = set[tuple[str, int]]


class _Recorder:
    """Function lines of sqlglot run while a sink is active, each recorded once per switch of sink."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.package = str(root / "sqlglot") + os.sep
        self.sink: Lines | None = None
        self._paths: dict[types.CodeType, str | None] = {}
        monitoring = sys.monitoring
        monitoring.use_tool_id(monitoring.COVERAGE_ID, "tpc_queries")
        monitoring.register_callback(monitoring.COVERAGE_ID, monitoring.events.LINE, self._on_line)
        monitoring.set_events(monitoring.COVERAGE_ID, monitoring.events.LINE)

    def _on_line(self, code: types.CodeType, line: int) -> object:
        if code not in self._paths:
            inside = code.co_filename.startswith(self.package) and code.co_flags & inspect.CO_OPTIMIZED
            self._paths[code] = os.path.relpath(code.co_filename, self.root).replace(os.sep, "/") if inside else None
        path = self._paths[code]
        if path is not None and self.sink is not None:
            self.sink.add((path, line))
        return sys.monitoring.DISABLE

    @contextlib.contextmanager
    def into(self, sink: Lines) -> Iterator[None]:
        previous = self.sink
        self.switch(sink)
        try:
            yield
        finally:
            self.switch(previous)

    def switch(self, sink: Lines | None) -> None:
        self.sink = sink
        sys.monitoring.restart_events()


def _hook_plan_steps(recorder: _Recorder, current: dict[str, Lines]) -> None:
    """Record what execute() runs to find tables, build the schema, optimize, log the SQL and plan as plan lines."""
    import sqlglot
    import sqlglot.executor as executor_module
    from sqlglot import exp

    def planning(function: Callable) -> Callable:
        def wrapper(*args: object, **kwargs: object) -> object:
            with recorder.into(current["plan"]):
                return function(*args, **kwargs)

        return wrapper

    sqlglot.find_tables = planning(sqlglot.find_tables)
    for name in ("optimize", "Plan", "ensure_schema"):
        setattr(executor_module, name, planning(getattr(executor_module, name)))
    owner = next(cls for cls in exp.Select.__mro__ if "sql" in vars(cls))
    original, caller = owner.sql, executor_module.execute.__code__

    def sql(self: object, *args: object, **kwargs: object) -> object:
        if sys._getframe(1).f_code is caller:
            with recorder.into(current["plan"]):
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    owner.sql = sql


def record_lines(root: Path, suite: str) -> dict[str, dict]:
    """Plan and run lines per query as bitmasks over query indexes, and the shared lines."""
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(HERE))
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    from collect_tpc import _import_everything

    _import_everything(root)
    import tpc_fingerprint as fp
    from sqlglot import transpile

    recorder = _Recorder(root)
    current: dict[str, Lines] = {}
    _hook_plan_steps(recorder, current)
    schema, pairs = fp._parts(suite)
    shared: Lines = set()
    with recorder.into(shared):
        tables = fp._Tables(None, suite, schema)
        tables.ship_all()

    def run(index: int, plan: Lines, ran: Lines, stages: list[Lines] | None = None) -> None:
        meta, sql = pairs[index]
        current.update(plan=plan, run=ran)
        with recorder.into(plan):
            received = fp._received(sql)
            if meta.get("execute"):
                transpile(sql, write="duckdb")
        if meta.get("execute"):
            if stages is None:
                with recorder.into(plan):
                    fp._plan(schema, fp._received(sql))
            else:
                with recorder.into(stages[0]):
                    fp._plan(schema, fp._received(sql), lambda stage: recorder.switch(stages[stage]))
                plan.update(*stages)
            with recorder.into(ran):
                fp._result(fp._execute(tables, received))

    warm: Lines = set()
    for index in range(len(pairs)):
        run(index, warm, warm)
    plans: list[Lines] = []
    runs: list[Lines] = []
    staged: list[list[Lines]] = []
    for index in range(len(pairs)):
        plans.append(set())
        runs.append(set())
        staged.append([set() for _ in range(fp.POST + 1)])
        run(index, plans[-1], runs[-1], staged[-1])
    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
    every = set().union(*plans, *runs)
    return {
        "plan": _masks(plans),
        "run": _masks(runs),
        "shared": _by_path(shared | (warm - every)),
        "stage": _stage_masks(_last_stages(staged, fp.POST)),
    }


# optimize() hands each rule its arguments, which no stage digest holds.
_UNSTAGED = {"sqlglot/optimizer/optimizer.py"}
LastStages = dict[tuple[str, int, int], int]


def _last_stages(staged: list[list[Lines]], post: int) -> LastStages:
    """Per line and query, the last stage of tpc_fingerprint._plan that runs it."""
    last: LastStages = {}
    for index, stages in enumerate(staged):
        for stage, lines in enumerate(stages):
            for path, line in lines:
                last[(path, line, index)] = post if path in _UNSTAGED else stage
    return last


def _stage_masks(last: LastStages) -> dict[str, dict[str, dict[str, int]]]:
    """Per line and stage, a bitmask of the queries whose last stage running the line is that stage."""
    masks: dict[str, dict[str, dict[str, int]]] = {}
    for (path, line, index), stage in last.items():
        by_stage = masks.setdefault(path, {}).setdefault(str(line), {})
        by_stage[str(stage)] = by_stage.get(str(stage), 0) | (1 << index)
    return masks


def _unmask(masks: dict[str, dict[str, dict[str, int]]], into: LastStages) -> None:
    """Add stage masks to per line and query last stages, keeping the later stage."""
    for path, by_line in masks.items():
        for line, by_stage in by_line.items():
            for stage, mask in by_stage.items():
                index = 0
                while mask:
                    if mask & 1:
                        key = (path, int(line), index)
                        into[key] = max(into.get(key, 0), int(stage))
                    mask >>= 1
                    index += 1


def _masks(per_query: list[Lines]) -> dict[str, dict[str, int]]:
    masks: dict[str, dict[str, int]] = {}
    for index, lines in enumerate(per_query):
        for path, line in lines:
            by_line = masks.setdefault(path, {})
            by_line[str(line)] = by_line.get(str(line), 0) | (1 << index)
    return masks


def _by_path(lines: Lines) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path, line in sorted(lines):
        found.setdefault(path, []).append(line)
    return found


def merge(runs: list[dict[str, dict]]) -> dict[str, dict]:
    """The union of recordings made under different hash seeds, whose set orders take different paths."""
    merged: dict[str, dict] = {"plan": {}, "run": {}, "shared": {}}
    last: LastStages = {}
    for found in runs:
        _unmask(found.get("stage", {}), last)
        for kind in ("plan", "run"):
            for path, by_line in found[kind].items():
                into = merged[kind].setdefault(path, {})
                for line, mask in by_line.items():
                    into[line] = into.get(line, 0) | mask
        for path, lines in found["shared"].items():
            merged["shared"][path] = sorted(set(merged["shared"].get(path, [])) | set(lines))
    merged["stage"] = _stage_masks(last)
    return merged
