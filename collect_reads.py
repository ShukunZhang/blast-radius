"""Record what the tests missing from the coverage map depend on.

Line coverage only sees sqlglot code a test executes in-process. A test that
reads sqlglot sources as data, or runs sqlglot in a child process, leaves no
lines behind. This runs each such test once on the pinned commit under an audit
hook and writes the repository files it opened and whether it started a Python
child that imports sqlglot. Every sqlglot module is imported before the hook
is installed, so module loading is not mistaken for a data read.

    python collect_reads.py SQLGLOT_CHECKOUT tests.json maps/coverage.bin maps/reads.json
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path


def _coverage_nodeids(path: Path) -> set[str]:
    blob = path.read_bytes()
    (count,) = struct.unpack_from("<I", blob, 4)
    offset = 8
    found: set[str] = set()
    for _ in range(count):
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        found.add(blob[offset : offset + length].decode("utf-8"))
        offset += length
    return found


def _eager_modules(repo: Path) -> list[str]:
    """Repository files a fresh `import sqlglot` loads."""
    code = (
        "import json, sys, sqlglot\n"
        "print(json.dumps(sorted(m.__file__ for m in list(sys.modules.values())"
        " if getattr(m, '__file__', None))))"
    )
    env = {**os.environ, "PYTHONPATH": str(repo)}
    output = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, check=True
    ).stdout
    return sorted({_relative(repo, name) for name in json.loads(output)} - {None})


def _relative(repo: Path, name: object) -> str | None:
    try:
        relative = Path(os.fsdecode(name)).resolve().relative_to(repo).as_posix()
    except (TypeError, ValueError, OSError):
        return None
    if relative.startswith("sqlglot/") and relative.endswith(".py"):
        return relative
    if relative.startswith(("sqlglotc/", "tests/fixtures/")) or "/" not in relative:
        return relative
    return None


def _import_everything() -> None:
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        try:
            importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 - optional extras are not dependencies of these tests
            continue


def _run(repo: Path, nodeid: str) -> dict[str, object]:
    file_name, class_name, method = nodeid.split("::")
    reads: set[str] = set()
    spawns_sqlglot = False
    active = False

    def hook(event: str, args: tuple) -> None:
        nonlocal spawns_sqlglot
        if not active:
            return
        if event == "open" and args and args[0] is not None:
            relative = _relative(repo, args[0])
            if relative is not None:
                reads.add(relative)
        elif event == "subprocess.Popen":
            raw = args[1] or []
            command = raw if isinstance(raw, (str, bytes)) else " ".join(str(part) for part in raw)
            command = os.fsdecode(command)
            if "python" in command and "sqlglot" in command:
                spawns_sqlglot = True

    sys.addaudithook(hook)
    module = importlib.import_module(file_name[:-3].replace("/", "."))
    case = getattr(module, class_name)
    active = True
    try:
        suite = unittest.TestSuite([case(method)])
        unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    finally:
        active = False
    return {"reads": sorted(reads), "spawns_sqlglot": spawns_sqlglot}


def main() -> None:
    repo, catalog_path, coverage_path, out = (Path(arg).resolve() for arg in sys.argv[1:5])
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))["tests"]
    covered = _coverage_nodeids(coverage_path)
    missing = [test["id"] for test in catalog if test["id"] not in covered]
    _import_everything()
    tests = {nodeid: _run(repo, nodeid) for nodeid in missing}
    for nodeid, found in tests.items():
        print(f"[reads] {nodeid}: {found}", file=sys.stderr)
    payload = {"eager": _eager_modules(repo), "tests": tests}
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
