"""Record and replay, per TPC query, what test_execute_tpch/tpcds need from sqlglot.

Mirrors TestExecutor._mp_execute: the test parses each query and pickles it
to a worker, which returns None unless the fixture marks it `execute`; then
it runs sqlglot.executor.execute (find tables, build the schema, optimize,
plan, run) and the test compares the result with DuckDB running the query
transpiled by sqlglot. Code the test runs after planning is recorded as
runtime lines by `collect_tpc.py`. Runs with the working directory at the
sqlglot checkout.
"""

from __future__ import annotations

import enum
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any


_LINKS = ("dependencies", "dependents")


def _expr(value: Any, parent: Any, seen: dict[Any, Any]) -> Any:
    """An expression with every attribute the executor relies on, tree links checked by identity."""
    from sqlglot import exp

    def child(item: Any, index: int | None, key: str) -> Any:
        if isinstance(item, exp.Expr):
            links = (item.parent is value, item.arg_key == key, item.index == index)
            return (links, _expr(item, None, seen))
        return _canon(item, seen)

    args = []
    for key, arg in sorted(value.args.items()):
        if isinstance(arg, list):
            args.append((key, [child(item, index, key) for index, item in enumerate(arg)]))
        else:
            args.append((key, child(arg, None, key)))
    return (
        type(value).__qualname__,
        args,
        None if parent is None else (type(parent).__qualname__, value.arg_key, value.index),
        _canon(value.comments, seen),
        _canon(value._type, seen),
        _canon(value._meta, seen),
        value._hash is None,
    )


def _shallow(step: Any, seen: dict[Any, Any]) -> tuple[str, str]:
    """A step's name and its fields other than its links, rendered once per digest."""
    key = ("shallow", id(step))
    if key not in seen:
        fields = {name: item for name, item in vars(step).items() if name not in _LINKS}
        seen[key] = (str(step.name), repr(_canon(fields, {})))
    return seen[key]


def _step(value: Any, seen: dict[Any, Any]) -> Any:
    """A plan step numbered in a traversal ordered by name and shallow content, so set order is irrelevant."""
    if id(value) in seen:
        return ("step", seen[id(value)])
    seen[id(value)] = seen.get("next", 0)
    seen["next"] = seen[id(value)] + 1
    dependencies = [_step(step, seen) for step in sorted(value.dependencies, key=lambda step: _shallow(step, seen))]
    dependents = sorted(repr(_shallow(step, seen)) for step in value.dependents)
    return ("Step", type(value).__qualname__, _shallow(value, seen)[1], dependencies, dependents)


def _canon(value: Any, seen: dict[Any, Any]) -> Any:
    """A deterministic, structural rendering: sets are sorted and plan steps numbered once."""
    from sqlglot import exp
    from sqlglot.planner import Step

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, type):
        return f"class {value.__module__}.{value.__qualname__}"
    if isinstance(value, exp.Expr):
        return _expr(value, value.parent, seen)
    if isinstance(value, Step):
        return _step(value, seen)
    if isinstance(value, dict):
        return ("dict", sorted((repr(_canon(key, seen)), _canon(item, seen)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return [_canon(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        return ("set", sorted(repr(_canon(item, seen)) for item in value))
    if hasattr(value, "__dict__"):
        return (type(value).__qualname__, _canon(vars(value), seen))
    return (type(value).__qualname__, repr(value))


def _parts(suite: str) -> tuple[dict, list[tuple[dict, str]]]:
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

    schema = TPCH_SCHEMA if suite == "tpch" else TPCDS_SCHEMA
    pairs = [(meta, sql) for meta, sql, _ in load_sql_fixture_pairs(f"optimizer/tpc-{suite[3:]}/tpc-{suite[3:]}.sql")]
    return schema, pairs


def _sha(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _received(sql: str) -> Any:
    """The query as the test's worker receives it: parsed, then pickled across."""
    from sqlglot import parse_one

    return pickle.loads(pickle.dumps(parse_one(sql)))


def _plan(schema: dict, received: Any) -> str:
    """What execute() derives before touching data: the tables named, the schema, the optimized query, the plan."""
    from sqlglot import find_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema

    names = sorted(table.name for table in find_tables(received))
    built = ensure_schema(schema, dialect=None)
    optimized = optimize(received, built, leave_tables_isolated=True, dialect=None)
    return _sha(
        [
            names,
            _canon(built.supported_table_args, {}),
            _canon(optimized, {}),
            optimized.sql(pretty=True),
            _canon(vars(Plan(optimized)), {}),
        ]
    )


SKIP, CHECK, EXECUTE = 0, 1, 2
_SUITES = ("tpch", "tpcds")


def _loaded_rows() -> dict[str, dict[str, list]]:
    """The rows TestExecutor.setUpClass reads from the CSV files, per suite and table."""
    import tests.test_executor as module

    module.TestExecutor.setUpClass()
    found = {"tpch": module.TestExecutor.tpch_tables, "tpcds": module.TestExecutor.tpcds_tables}
    return {suite: {name: table.rows for name, table in found[suite].items()} for suite in _SUITES}


def write_tables(path: Path) -> None:
    """Pickle the loaded rows per table, so a probe needs neither pandas, DuckDB nor the CSV files."""
    rows = _loaded_rows()
    blobs = {suite: {name: pickle.dumps(found, protocol=5) for name, found in rows[suite].items()} for suite in _SUITES}
    path.write_bytes(pickle.dumps(blobs, protocol=5))


class _Tables:
    """The worker's tables: built from the loaded rows as setUpClass builds them, then shipped through pickle
    as the pool initializer ships them, one table at a time and once each."""

    def __init__(self, path: Path | None, suite: str, schema: dict) -> None:
        self._path, self._suite = path, suite
        self._loaded: dict[str, bytes] | None = None
        self.schema = pickle.loads(pickle.dumps(schema))
        self._source = schema
        self._shipped: dict[str, Any] = {}

    @property
    def _blobs(self) -> dict[str, bytes]:
        if self._loaded is None:
            if self._path is not None and self._path.exists():
                self._loaded = pickle.loads(self._path.read_bytes())[self._suite]
            else:
                self._loaded = {name: pickle.dumps(found) for name, found in _loaded_rows()[self._suite].items()}
        return self._loaded

    def get(self, name: str) -> Any:
        from sqlglot.executor.table import Table

        if name not in self._shipped:
            built = Table(columns=self._source[name], rows=pickle.loads(self._blobs[name]))
            self._shipped[name] = pickle.loads(pickle.dumps(built))
        return self._shipped[name]

    def ship_all(self) -> None:
        """Build and ship every table, as setUpClass and the pool initializer do."""
        for name in self._blobs:
            self.get(name)


def _execute(tables: _Tables, received: Any) -> Any:
    """The test's mp_execute, in a worker whose initializer received the schema and the tables."""
    from sqlglot import find_tables
    from sqlglot.executor import execute

    chosen = {}
    for table in find_tables(received):
        chosen[table.name] = tables.get(table.name)
    return execute(received, schema=tables.schema, tables=chosen)


def _result(table: Any) -> str:
    received = pickle.loads(pickle.dumps(table))
    return _sha((list(received.columns), [list(row) for row in received.rows]))


def record(suite: str) -> list[dict[str, Any] | None]:
    """Per query: None when the test only ships it, else digests of its DuckDB SQL, plan and result, and their CPU."""
    from sqlglot import transpile

    schema, pairs = _parts(suite)
    tables = _Tables(None, suite, schema)
    entries: list[dict[str, Any] | None] = []
    for meta, sql in pairs:
        received = _received(sql)
        if not meta.get("execute"):
            entries.append(None)
            continue
        start = time.process_time()
        entry: dict[str, Any] = {"transpile": _sha(transpile(sql, write="duckdb")), "plan": _plan(schema, _received(sql))}
        middle = time.process_time()
        entry["result"] = _result(_execute(tables, received))
        entry["check_s"], entry["execute_s"] = round(middle - start, 3), round(time.process_time() - middle, 3)
        entries.append(entry)
    return entries


def replay(suite: str, expected: list, actions: list[int], full: bool, tables_path: Path | None) -> bool:
    """Whether the patched tree leaves the test passing, as on the pinned commit; stops at the first doubt.

    Queries whose action is SKIP run no edited code. Every other query must
    parse and cross pickle, and an executed one must give DuckDB the same SQL.
    Its result must then match the pinned one: taken as given for CHECK when
    its plan digests the same, else by executing it. `full` also rebuilds and
    ships every table, for edits to code setUpClass or the initializer runs.
    """
    try:
        import sqlglot.errors
        import sqlglot.executor
        import sqlglot.executor.python
        import sqlglot.executor.table
        import sqlglot.optimizer
        import sqlglot.planner  # noqa: F401 - the modules the test imports, whose import must not fail
        from sqlglot import transpile

        schema, pairs = _parts(suite)
        if not len(pairs) == len(expected) == len(actions):
            return False
        tables = _Tables(tables_path, suite, schema)
        if full:
            tables.ship_all()
        for (meta, sql), entry, action in zip(pairs, expected, actions):
            if action == SKIP:
                continue
            if bool(meta.get("execute")) != (entry is not None):
                return False
            received = _received(sql)
            if entry is None:
                continue
            if _sha(transpile(sql, write="duckdb")) != entry["transpile"]:
                return False
            if action == CHECK and _plan(schema, _received(sql)) == entry["plan"]:
                continue
            if entry["result"] is None or _result(_execute(tables, received)) != entry["result"]:
                return False
        return True
    except Exception:  # noqa: BLE001 - any failure leaves the test selected
        return False
