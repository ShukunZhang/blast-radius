"""Build maps/cases.json: per fixture case of each fixture-driven test, the sqlglot lines it runs and its CPU.

Usage: python collect_cases.py SQLGLOT_ROOT OUT_JSON [JOBS]

A fixture-driven test is a test method that calls check_file or one of
tests.helpers' fixture loaders. Each test is recorded in a fresh child under
several hash seeds, whose set orders take different paths, and their lines
are united; one more child times it. case_proof.py reads the map.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SEEDS = ("1", "2", "3")
_CALLS = {"check_file", "load_sql_fixture_pairs", "load_sql_fixtures"}
# The TPC executor tests draw every query before running any; tpc_proof covers them.
_SKIPPED_FILES = {"tests/test_executor.py"}


def fixture_tests(root: Path) -> list[str]:
    """Ids of test methods whose body calls check_file or a fixture loader."""
    found: list[str] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in _SKIPPED_FILES:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test") and _calls_fixtures(item):
                    found.append(f"{relative}::{node.name}::{item.name}")
    return found


def _calls_fixtures(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else ""
            if name in _CALLS:
                return True
    return False


def _child(kind: str, root: Path, test_id: str, seed: str) -> Any:
    command = [sys.executable, "-B", __file__, "--child", kind, str(root), test_id]
    environment = {**os.environ, "PYTHONHASHSEED": seed, "SKIP_INTEGRATION": "0"}
    done = subprocess.run(command, capture_output=True, text=True, env=environment)
    if done.returncode != 0:
        print(f"{test_id} {kind} seed {seed} failed: {done.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
        return None
    return json.loads(done.stdout)


def merge(test_id: str, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The union of one test's recordings, or None when they drew different cases."""
    if any(run["calls"] != runs[0]["calls"] for run in runs):
        print(f"{test_id} draws different cases between seeds", file=sys.stderr)
        return None
    lines: dict[str, dict[str, int]] = {}
    full: dict[str, set[int]] = {}
    for run in runs:
        for path, by_line in run["lines"].items():
            into = lines.setdefault(path, {})
            for line, mask in by_line.items():
                into[line] = into.get(line, 0) | int(mask, 16)
        for path, found in run["full"].items():
            full.setdefault(path, set()).update(found)
    return {
        "calls": runs[0]["calls"],
        "lines": {path: {line: format(mask, "x") for line, mask in by_line.items()} for path, by_line in lines.items()},
        "full": {path: sorted(found) for path, found in full.items()},
        "classes": _union_masks([run["classes"] for run in runs]),
        "args": _union_masks([run["args"] for run in runs]),
        "args_full": sorted(set().union(*(run["args_full"] for run in runs))),
        "args_derived": _union_sites([run["args_derived"] for run in runs]),
    }


def _union_sites(recorded: list[dict[str, list[list[Any]]]]) -> dict[str, list[list[Any]]]:
    sites: dict[str, set[tuple[str, int]]] = {}
    for by_owner in recorded:
        for owner, found in by_owner.items():
            sites.setdefault(owner, set()).update((path, line) for path, line in found)
    return {owner: [list(site) for site in sorted(found)] for owner, found in sites.items()}


def _union_masks(recorded: list[dict[str, str]]) -> dict[str, str]:
    masks: dict[str, int] = {}
    for by_name in recorded:
        for name, mask in by_name.items():
            masks[name] = masks.get(name, 0) | int(mask, 16)
    return {name: format(mask, "x") for name, mask in masks.items()}


def _collect(root: Path, test_id: str) -> dict[str, Any] | None:
    runs = [_child("lines", root, test_id, seed) for seed in SEEDS]
    if any(run is None for run in runs):
        return None
    return merge(test_id, runs)


def main() -> None:
    if sys.argv[1] == "--child":
        sys.path.insert(0, str(HERE))
        import case_record

        record = case_record.record_lines if sys.argv[2] == "lines" else case_record.record_seconds
        sys.stdout.write(json.dumps(record(Path(sys.argv[3]), sys.argv[4])))
        return
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tests = fixture_tests(root)
    with ThreadPoolExecutor(jobs) as pool:
        recorded = dict(zip(tests, pool.map(lambda test_id: _collect(root, test_id), tests)))
    built: dict[str, Any] = {}
    for test_id, entry in recorded.items():
        if entry is None or not entry["calls"]:
            continue
        seconds = _child("seconds", root, test_id, "1")
        if seconds is None or len(seconds["case_s"]) != sum(count for _name, count in entry["calls"]):
            continue
        built[test_id] = {**entry, **seconds}
        print(test_id, len(seconds["case_s"]), seconds["setup_s"], seconds["rest_s"], round(sum(seconds["case_s"]), 2), file=sys.stderr)
    out.write_text(json.dumps(built, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
