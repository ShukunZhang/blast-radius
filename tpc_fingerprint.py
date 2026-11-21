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
from collections.abc import Callable
from typing import Any


_LINKS = ("dependencies", "dependents")


def _expr(value: Any, parent: Any, seen: dict[int, int]) -> Any:
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


def _step(value: Any, seen: dict[int, int]) -> Any:
    """A plan step numbered in a traversal ordered by name and shallow content, so set order is irrelevant."""
    if id(value) in seen:
        return ("step", seen[id(value)])
    seen[id(value)] = len(seen)
    fields = {key: item for key, item in vars(value).items() if key not in _LINKS}
    shallow = _canon(fields, {})

    def order(step: Any) -> tuple[str, str]:
        return (str(step.name), repr(_canon({k: v for k, v in vars(step).items() if k not in _LINKS}, {})))

    dependencies = [_step(step, seen) for step in sorted(value.dependencies, key=order)]
    dependents = sorted(repr(order(step)) for step in value.dependents)
    return ("Step", type(value).__qualname__, shallow, dependencies, dependents)


def _canon(value: Any, seen: dict[int, int]) -> Any:
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


def _worker(suite: str) -> Callable[[Any, dict], Any]:
    """The test's mp_execute, after its setUpClass and with the tables shipped as its pool initializer ships them."""
    import tests.test_executor as module

    if not _LOADED:
        module.TestExecutor.setUpClass()
        _LOADED.append(module)
    tables = module.TestExecutor.tpch_tables if suite == "tpch" else module.TestExecutor.tpcds_tables
    schema = module.TPCH_SCHEMA if suite == "tpch" else module.TPCDS_SCHEMA
    module.initializer(*pickle.loads(pickle.dumps((schema, tables))))
    return module.mp_execute


_LOADED: list[Any] = []


def _result(table: Any) -> str:
    received = pickle.loads(pickle.dumps(table))
    return _sha((list(received.columns), [list(row) for row in received.rows]))


def record(suite: str) -> list[dict[str, str] | None]:
    """Per query: None when the test only ships it, else digests of its DuckDB SQL, its plan and its result."""
    from sqlglot import transpile

    schema, pairs = _parts(suite)
    entries: list[dict[str, str] | None] = []
    worker = _worker(suite)
    for meta, sql in pairs:
        received = _received(sql)
        if not meta.get("execute"):
            entries.append(None)
            continue
        entries.append(
            {
                "transpile": _sha(transpile(sql, write="duckdb")),
                "plan": _plan(schema, _received(sql)),
                "result": _result(worker(received, meta)),
            }
        )
    return entries


def replay(suite: str, expected: list[dict[str, str | None] | None], unreached: bool) -> bool:
    """Whether the patched tree leaves the test passing, as on the pinned commit; stops at the first doubt.

    Every query must parse and cross pickle, and each executed one must give
    DuckDB the same SQL. Its result must then match the pinned one: taken as
    given when `unreached` (no edit reaches code the test runs after planning)
    and the plan digests the same, else by executing it.
    """
    from sqlglot import transpile

    try:
        schema, pairs = _parts(suite)
        if len(pairs) != len(expected):
            return False
        worker = None if unreached else _worker(suite)
        for (meta, sql), entry in zip(pairs, expected):
            received = _received(sql)
            if bool(meta.get("execute")) != (entry is not None):
                return False
            if entry is None:
                continue
            if _sha(transpile(sql, write="duckdb")) != entry["transpile"]:
                return False
            if unreached and _plan(schema, _received(sql)) == entry["plan"]:
                continue
            if entry["result"] is None:
                return False
            worker = worker or _worker(suite)
            if _result(worker(received, meta)) != entry["result"]:
                return False
        return True
    except Exception:  # noqa: BLE001 - any failure leaves the test selected
        return False
