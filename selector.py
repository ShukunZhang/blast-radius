"""Select tests for stand-alone patches against the pinned SQLGlot checkout."""
import ast
import bisect
import importlib
import json
import os
import re
import resource
import signal
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from pathlib import Path

from escape_map import changed_keys
from inline_pool import InlinePool
from observe import REPLACEMENTS, identity_site

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
MAX_PROBE_MS = 850
# Cheap tests get the original alarm. A slow optimizer replay is killed and
# kept if the cached jobs do not finish; an executor run is never started.
CHEAP_ALARM_S = 0.75
SLOW_ALARM_S = 1.0
BATCH_PROBE_S = 90.0
OPTIMIZER_REPLAY = "tests/test_optimizer.py"
ORDER_OPS = {"Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"}


def changes(root, diff):
    """Apply ordinary Git unified hunks in memory, validating old context."""
    result = {}
    current = None
    lines = iter(diff.splitlines(keepends=True))
    for line in lines:
        if line.startswith("diff --git "):
            current = None
        elif line.startswith("--- a/"):
            path = line[6:].rstrip("\n")
            if next(lines).rstrip("\n") != "+++ b/" + path:
                raise ValueError("Unsupported diff path")
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or path in result:
                raise ValueError("Unexpected diff path")
            old = (root / path).read_text().splitlines(keepends=True)
            result[path] = current = (old, [], set(), set())
        elif line.startswith("@@"):
            if current is None or not (match := HUNK.match(line)):
                raise ValueError("Unsupported hunk")
            old, hunks, deleted, inserted = current
            start, old_count, _, new_count = match.groups()
            old_count, new_count = int(old_count or 1), int(new_count or 1)
            cursor = int(start) - bool(old_count)
            initial = cursor
            replacement = []
            while old_count or new_count:
                edit = next(lines)
                mode, body = edit[0], edit[1:]
                if mode in " -":
                    if old[cursor] != body:
                        raise ValueError("Original patch context differs")
                    cursor += 1
                    old_count -= 1
                if mode in " +":
                    replacement.append(body)
                    new_count -= 1
                if mode == "-":
                    deleted.add(cursor)
                if mode == "+":
                    inserted.add(cursor)
                if mode not in " +-" or min(old_count, new_count) < 0:
                    raise ValueError("Unsupported patch line")
            hunks.append((initial, cursor, replacement))
        elif line.startswith(("Binary files ", "GIT binary patch", "--- ", "\\ No newline")):
            raise ValueError("Unsupported patch format")
    if not result or any(not h for _, h, _, _ in result.values()):
        raise ValueError("No supported hunks")
    output = {}
    for path, (old, hunks, deleted, inserted) in result.items():
        new = old.copy()
        for start, end, replacement in reversed(hunks):
            new[start:end] = replacement
        output[path] = ("".join(old), "".join(new), deleted, inserted)
    return output


def last_leq(ordered, high):
    return bisect.bisect_right(ordered, high) - 1


def overlaps(ordered, low, high):
    end = last_leq(ordered, high)
    return end >= 0 and ordered[end] >= low


def regions(tree, deleted, inserted):
    """Map old diff coordinates to enclosing executable statements."""
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    intervals = set()
    for line in deleted | inserted:
        owners = [n for n in statements if n.lineno <= line <= n.end_lineno]
        if not owners:
            return None
        owner = min(owners, key=lambda n: (n.end_lineno - n.lineno, -n.col_offset))
        # Header and decorator changes can affect tests which call the definition.
        end = owner.end_lineno
        if isinstance(owner, (ast.If, ast.For, ast.While, ast.With, ast.Match, ast.AsyncFor, ast.AsyncWith)):
            body = owner.cases if isinstance(owner, ast.Match) else owner.body
            end = min(end, max(line, body[0].lineno - 1)) if not isinstance(owner, ast.Match) else end
        intervals.add((owner.lineno, end))
    return intervals


class Impact:
    def __init__(self, tests, baseline, root, escapes):
        self.tests = tests
        self.root = root
        self.references = {}
        self.escapes = escapes
        self.numbers = {test["id"]: test["number"] for test in tests}
        self.all = {t["number"] for t in tests}
        self.cover = defaultdict(list)
        self.readers = defaultdict(set)
        self.unknown = set()
        for test in tests:
            number = test["number"]
            if test["id"] not in baseline["tests"]:
                self.unknown.add(number)
                continue
            executed, reads = baseline["tests"][test["id"]]
            for path, lines in executed.items():
                self.cover[path].append((number, lines))
            for path in reads:
                self.readers[path].add(number)
        self.collection, self.collection_reads = baseline["collection"]
        self.source_sensitive = set()
        checked = {}
        source_pattern = re.compile(r"\b(?:getsource|getlines|co_firstlineno|read_text|traceback)\b")
        for test in tests:
            path = root / test["file"]
            if path not in checked:
                checked[path] = path.is_file() and bool(source_pattern.search(path.read_text()))
            if checked[path]:
                self.source_sensitive.add(test["number"])
        self.direct_config_readers = {
            path for path in {t["file"] for t in tests}
            if (root / path).is_file() and "NUMERIC_ESCAPES" in (root / path).read_text()
        }

    def reference_tests(self, symbols):
        key = tuple(sorted(symbols))
        if key not in self.references:
            name = re.compile(r"\b(?:" + "|".join(map(re.escape, key)) + r")\b")
            matched = set()
            for path, covered in self.cover.items():
                if not path.endswith(".py"):
                    continue
                source = (self.root / path).read_text()
                lines = {n for n, line in enumerate(source.splitlines(), 1) if name.search(line)}
                if lines:
                    matched.update(test for test, executed in covered if lines.intersection(executed))
            self.references[key] = matched
        return self.references[key]

    def candidates(self, edits):
        selected = set(self.unknown)
        for path, (before, after, deleted, inserted) in edits.items():
            selected.update(self.readers[path])
            if path in self.collection_reads:
                return self.all
            if path.startswith("tests/fixtures/"):
                if not self.readers[path]:
                    return self.all
                continue
            if not path.endswith(".py") or not path.startswith(("sqlglot/", "tests/")):
                return self.all
            try:
                old = ast.parse(before, type_comments=True)
                new = ast.parse(after, type_comments=True)
            except SyntaxError:
                return self.all
            if path in self.escapes and len(edits) == 1 and (keys := changed_keys(old, new)) is not None:
                for key in keys:
                    selected.update(self.numbers[node] for node in self.escapes[path].get(key, [])
                                    if node in self.numbers)
                selected.update(n for n in self.reference_tests({"NUMERIC_ESCAPES"})
                                if self.tests[n]["file"] in self.direct_config_readers)
                continue
            if path.startswith("tests/"):
                selected.update(t["number"] for t in self.tests if t["file"] == path)
            affected = regions(old, deleted, inserted)
            if ast.dump(old) == ast.dump(new):
                # Scope source-text/line-number readers conservatively. Ordinary
                # comments do not enter Python's executable syntax tree.
                selected.update(self.source_sensitive)
                continue
            if affected is None:
                return self.all
            declarations = declaration_symbols(old, affected)
            if declarations:
                references = self.reference_tests(declarations)
                parts = Path(path).parts
                dialect = Path(path).stem
                dialect_consumers = self.readers["dialect:" + dialect]
                if parts[1] in {"dialects", "parsers", "generators", "typing"} and dialect_consumers:
                    references = references & dialect_consumers
                    references |= {n for n, _ in self.cover[path]}
                selected.update(references)
            collection = self.collection.get(path, [])
            if not declarations and any(overlaps(collection, lo, hi) for lo, hi in affected):
                return self.all
            for number, executed in self.cover[path]:
                if any(overlaps(executed, lo, hi) for lo, hi in affected):
                    selected.add(number)
        return selected


def _swap_target(previous, current):
    """Return the new comparison operator when that is the only change to the node."""
    if len(previous.ops) != 1 or len(current.ops) != 1:
        return None
    old_op = type(previous.ops[0]).__name__
    new_op = type(current.ops[0]).__name__
    if old_op == new_op:
        return None
    if old_op in ("Is", "IsNot"):
        partner = "Eq" if old_op == "Is" else "NotEq"
        if new_op == partner and identity_site(previous):
            return old_op, new_op
        return None
    if old_op in ORDER_OPS and new_op in ORDER_OPS:
        return old_op, new_op
    return None


def comparison_edits(edits):
    """Patches whose only AST change is one or more comparison-operator swaps."""
    sites = []
    operator_text = re.compile(r"\bis\b|!=|==|[<>]")
    for path, (before, after, removed, _) in edits.items():
        if not path.startswith("sqlglot/") or not path.endswith(".py") or not removed:
            return []
        old_lines = before.split("\n")
        if not any(operator_text.search(old_lines[line - 1]) for line in removed):
            return []
        try:
            old = ast.parse(before, type_comments=True)
            new = ast.parse(after, type_comments=True)
        except SyntaxError:
            return []
        old_nodes = [n for n in ast.walk(old) if isinstance(n, ast.Compare)]
        new_nodes = [n for n in ast.walk(new) if isinstance(n, ast.Compare)]
        if len(old_nodes) != len(new_nodes):
            return []
        for previous, current in zip(old_nodes, new_nodes):
            swap = _swap_target(previous, current)
            if swap is None:
                continue
            old_op, new_op = swap
            sites.append((path, previous.lineno, previous.end_lineno,
                          f"{path}:{previous.lineno}:{previous.col_offset}", old_op, new_op))
            current.ops = previous.ops
        if ast.dump(old) != ast.dump(new):
            return []
    return sites


def _equivalent(value, old_op, new_op):
    """True when a baseline record proves this swap returns the same bool."""
    if value is True:
        partner = REPLACEMENTS.get(old_op)
        return bool(partner) and partner[0] == new_op
    if isinstance(value, str) and value:
        return new_op in value.split(",")
    return False


def exclude_equivalent(candidates, edits, observations, impact):
    if not observations or not candidates:
        return candidates
    sites = comparison_edits(edits)
    if not sites:
        return candidates
    # Startup observations can affect later tests through shared state.
    if any(not _equivalent(observations.get(key, {}).get(""), old_op, new_op)
           for _, _, _, key, old_op, new_op in sites if "" in observations.get(key, {})):
        return candidates
    eligible = candidates - impact.unknown - impact.source_sensitive
    for path in edits:
        eligible.difference_update(impact.readers[path])
    confirmed = set(eligible)
    for path, first, last, key, old_op, new_op in sites:
        evidence = observations.get(key, {})
        if overlaps(impact.collection.get(path, ()), first, last) and not _equivalent(evidence.get(""), old_op, new_op):
            return candidates
        covered = {n for n, lines in impact.cover[path] if overlaps(lines, first, last)}
        matching = {n for n in eligible if _equivalent(evidence.get(impact.tests[n]["id"]), old_op, new_op)}
        if not matching:
            return candidates
        # A test listed by the scope-level filter may never reach this expression.
        # Only waive that site when neither collector saw the line or its operands.
        unobserved = {n for n in eligible - covered if impact.tests[n]["id"] not in evidence}
        confirmed.intersection_update(matching | unobserved)
    return candidates - confirmed


def declaration_symbols(tree, affected):
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            if not any(statement.lineno <= hi and lo <= statement.end_lineno for lo, hi in affected):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(base, ast.Name) and base.id in {"Expression", "Expr"} for base in node.bases):
                names.add(node.name)
            else:
                names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def overlay(base, dest, edits):
    """Link unchanged paths and write patched files into a tiny temp tree."""
    for entry in base.iterdir():
        here = dest / entry.name
        if entry.name in edits:
            here.write_text(edits[entry.name][1])
            continue
        nested = {path.split("/", 1)[1]: change for path, change in edits.items()
                  if path.startswith(entry.name + "/")}
        if nested:
            here.mkdir()
            overlay(entry, here, nested)
        else:
            here.symlink_to(entry, target_is_directory=entry.is_dir())


class ProbeResult(unittest.TestResult):
    limits = {}

    def __init__(self):
        super().__init__()
        self.failed = set()
        self.active = None

    def startTest(self, test):
        self.active = test.id()
        super().startTest(test)
        signal.setitimer(signal.ITIMER_REAL, type(self).limits.get(test.id(), CHEAP_ALARM_S))

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.active = None
        super().stopTest(test)

    def addError(self, test, exc):
        self.failed.add(test.id())

    addFailure = addError

    def addUnexpectedSuccess(self, test):
        self.failed.add(test.id())

    def addSubTest(self, test, subtest, exc):
        if exc:
            self.failed.add(test.id())
            raise unittest.case._ShouldStop


def timeout(*_):
    raise TimeoutError("probe deadline")


def job_impact(edits):
    regions_by_path = {}
    for path, (before, after, deleted, inserted) in edits.items():
        if path.startswith("tests/fixtures/"):
            continue
        if not path.startswith("sqlglot/") or not path.endswith(".py"):
            return None
        old, new = ast.parse(before), ast.parse(after)
        if ast.dump(old) == ast.dump(new):
            continue
        affected = regions(old, deleted, inserted)
        if not affected:
            return None
        # Untracked initializer or declaration effects invalidate all entries.
        def functions_contain(low, high):
            return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.body[0].lineno <= low <= high <= n.end_lineno
                       for n in ast.walk(old))
        if not all(functions_contain(lo, hi) for lo, hi in affected):
            return None
        regions_by_path[path] = affected

    def affected(covered):
        return any(overlaps(covered.get(path, ()), lo, hi)
                   for path, intervals in regions_by_path.items() for lo, hi in intervals)

    return affected


def probe(tests, edits, publish):
    """Run base-suite unittest cases. publish(passed, failed, errors) after each one."""
    InlinePool.impact = job_impact(edits)
    ProbeResult.limits = {}
    signal.signal(signal.SIGALRM, timeout)
    for path in edits:
        if path.startswith("sqlglot/") and path.endswith(".py"):
            module = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
            try:
                importlib.import_module(module)
            except Exception:
                publish(set(), set(), {"*"})
                return
    names = {}
    cases = []
    failed_files = set()
    passed = set()
    failed = set()

    def flush():
        publish(passed, failed, failed_files)

    for test in tests:
        file, *parts = test["id"].split("::")
        module = file.removesuffix(".py").replace("/", ".")
        try:
            imported = importlib.import_module(module)
        except Exception:
            failed_files.add(file)
            continue
        if hasattr(imported, "ProcessPoolExecutor"):
            imported.ProcessPoolExecutor = InlinePool
        if len(parts) != 2:
            continue
        cls = getattr(imported, parts[0], None)
        if not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
            continue
        try:
            case = cls(parts[1])
        except Exception:
            continue
        cases.append(case)
        names[case.id()] = test["number"]
        if test["ms"] > MAX_PROBE_MS:
            ProbeResult.limits[case.id()] = SLOW_ALARM_S

    result = ProbeResult()
    original_stop = result.stopTest

    def stop_and_publish(test):
        original_stop(test)
        number = names.get(test.id())
        if number is None:
            return
        if test.id() in result.failed:
            failed.add(number)
        else:
            passed.add(number)
        flush()

    result.stopTest = stop_and_publish
    unittest.TestSuite(cases).run(result)
    for name in result.failed - names.keys():
        # unittest uses string IDs such as setUpClass (tests.foo.Class).
        target = name.rsplit("(", 1)[-1].removesuffix(")")
        failed.update(number for case, number in names.items() if case.startswith(target + "."))
        passed.difference_update(failed)
    flush()


def _read_probe(path):
    try:
        passed, failed, errors = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return set(), set(), set()
    if not isinstance(passed, list) or not isinstance(failed, list) or not isinstance(errors, list):
        return set(), set(), set()
    return set(passed), set(failed), set(errors)


def verify(root, edits, tests, budget):
    """Replay this commit's targets in one fresh process. A killed run keeps partial passes."""
    with tempfile.TemporaryDirectory(prefix="blast-") as folder:
        folder = Path(folder)
        tree = folder / "checkout"
        tree.mkdir()
        overlay(root, tree, edits)
        result_path = folder / "result.json"

        def publish(passed, failed, errors):
            payload = json.dumps([sorted(passed), sorted(failed), sorted(errors)])
            temporary = result_path.with_suffix(".part")
            temporary.write_text(payload)
            temporary.replace(result_path)

        pid = os.fork()
        if pid == 0:
            try:
                with open(os.devnull, "w") as silence:
                    os.dup2(silence.fileno(), 1)
                    os.dup2(silence.fileno(), 2)
                os.setsid()
                os.chdir(tree)
                sys.path.insert(0, str(tree))
                probe(tests, edits, publish)
                os._exit(0)
            except BaseException:
                os._exit(1)
        deadline = time.monotonic() + budget
        while not os.waitpid(pid, os.WNOHANG)[0]:
            if time.monotonic() >= deadline:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                break
            time.sleep(0.005)
        if result_path.exists():
            return _read_probe(result_path)
        return set(), set(), set()


def split(chosen, durations):
    if not chosen:
        return []
    jobs = sorted(chosen, key=lambda number: (-durations[number], number))
    total = sum(durations[j] for j in jobs)
    longest = durations[jobs[0]]
    best_cost, best = total + 2000, [jobs]
    for k in range(2, min(8, len(jobs)) + 1):
        fee = 2000 * k
        if fee + longest >= best_cost:
            break
        if fee + (total + k - 1) // k >= best_cost:
            continue
        groups = [[job] for job in jobs[:k]]
        loads = [durations[job] for job in jobs[:k]]
        for job in jobs[k:]:
            i = loads.index(min(loads))
            loads[i] += durations[job]
            groups[i].append(job)
        cost = max(loads) + fee
        if cost < best_cost:
            best_cost, best = cost, groups
    return best


def schedule_cost(groups, durations):
    return max((sum(durations[j] for j in g) for g in groups), default=0) + 2000 * len(groups)


def tests_to_replay(candidates, tests, durations, cache_replay):
    """Cheap tests, or the slow optimizer test that sets the schedule.

    Tests over 850 ms are kept when a shorter replay cannot change the cost.
    The exception is an optimizer test that itself sets that cost: its pooled
    SQL calls come from the baseline cache unless they touched the patch, so
    a pass can be shown inside the one-second alarm. Executor and parser
    monsters are not replayed; running them costs about as much CPU as it saves.
    """
    by_number = {test["number"]: test for test in tests}
    expensive = set()
    cheap = []
    slow = []
    for number in candidates:
        test = by_number[number]
        if test["ms"] <= MAX_PROBE_MS:
            cheap.append(test)
        else:
            expensive.add(number)
            if cache_replay and test["file"] == OPTIMIZER_REPLAY:
                slow.append(test)
    base = schedule_cost(split(candidates, durations), durations)
    alarm_ms = int(SLOW_ALARM_S * 1000)
    worth = []
    for test in slow:
        saved = base - schedule_cost(split(candidates - {test["number"]}, durations), durations)
        if saved > alarm_ms:
            worth.append(test)
    anchored = bool(expensive) and base == schedule_cost(split(expensive, durations), durations)
    if anchored:
        worth.sort(key=lambda test: -test["ms"])
        return worth
    cheap.sort(key=lambda test: test["number"])
    return cheap


def cpu_time():
    return sum(sum(resource.getrusage(who)[:2])
               for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def main():
    data = Path(os.environ.get("BLAST_DATA", "/data"))
    root = Path(os.environ.get("BLAST_ROOT", "/sqlglot")).resolve()
    tests = json.loads((data / "tests.json").read_text())["tests"]
    for test in tests:
        test["ms"] = round(test["seconds"] * 1000)
    durations = {test["number"]: test["ms"] for test in tests}
    map_path = Path(os.environ.get("BLAST_MAP", "/app/coverage.json"))
    job_path = Path(os.environ.get("BLAST_JOBS", str(map_path) + ".jobs"))
    if job_path.is_file():
        InlinePool.baseline_jobs = json.loads(job_path.read_text())
    observed_path = Path(os.environ.get("BLAST_OBSERVED", "/app/observed.json"))
    observations = json.loads(observed_path.read_text()) if observed_path.exists() else {}
    observed = observations.get("comparisons", {})
    impact = Impact(tests, json.loads(map_path.read_text()), root,
                    observations.get("escapes", {}))
    full_schedule = split(impact.all, durations)
    sys.dont_write_bytecode = True
    importlib.import_module("pandas")
    importlib.import_module("duckdb")
    probe_budget = BATCH_PROBE_S
    for patch in sorted((data / "commits").glob("*.patch")):
        started = cpu_time() if os.environ.get("BLAST_STATS") else 0
        candidates = set()
        confirmed = set()
        edits = {}
        try:
            edits = changes(root, patch.read_text())
            raw = impact.candidates(edits)
            candidates = exclude_equivalent(raw, edits, observed, impact)
            confirmed = raw - candidates
            one_fixture_owner = len(candidates) == 1 and all(
                path.startswith("tests/fixtures/") for path in edits
            )
            # A fixture's own reader is the assertion. Replaying it spends CPU
            # to rediscover a failure. Other commits replay every test whose
            # result can change the schedule.
            cache_replay = job_impact(edits) is not None
            targets = [] if one_fixture_owner or probe_budget < 0.2 else tests_to_replay(
                candidates, tests, durations, cache_replay
            )
            if targets:
                slow_count = sum(test["ms"] > MAX_PROBE_MS for test in targets)
                allowance = min(probe_budget, 3.0 + SLOW_ALARM_S * slow_count)
                began = time.monotonic()
                passed, failed, errors = verify(root, edits, targets, allowance)
                probe_budget -= time.monotonic() - began
                if "*" in errors:
                    chosen = set(impact.all)
                else:
                    confirmed.update(passed - failed)
                    confirmed.difference_update(
                        test["number"] for test in tests if test["file"] in errors
                    )
                    chosen = set(candidates) - confirmed
                    chosen.update(
                        test["number"] for test in tests if test["file"] in errors
                    )
            else:
                chosen = set(candidates)
        except (OSError, ValueError, IndexError, StopIteration):
            chosen = impact.all
            confirmed = set()
        if os.environ.get("BLAST_STATS"):
            print(f"{patch.stem}: {len(chosen)} selected (from {len(candidates)}) CPU {cpu_time()-started:.4f}", file=sys.stderr)
        workers = full_schedule if chosen == impact.all else split(chosen, durations)
        print(json.dumps({"commit": patch.stem, "workers": workers},
                         separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
