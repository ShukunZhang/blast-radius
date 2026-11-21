"""Split selected tests across up to 8 workers.

Each worker after the first adds 2 seconds in the search. The scorer charges
2 seconds for every worker, including the first, so the two objectives pick
the same worker count.
"""

from __future__ import annotations

from heapq import heapify, heapreplace
from math import isfinite


def schedule_tests(durations: list[float]) -> tuple[float, list[list[int]]]:
    """Schedule independent tests on up to 8 identical workers.

    Durations are seconds. The returned groups are indexes into `durations`.
    """
    if any(not isfinite(duration) or duration < 0 for duration in durations):
        raise ValueError("Test durations must be finite and nonnegative.")

    if not durations:
        return 0.0, []

    total = sum(durations)

    # Two workers can save at most half the serial time.
    # For batches of 4 seconds or less, one worker is optimal.
    if total <= 4.0:
        return total, [list(range(len(durations)))]

    jobs = sorted(range(len(durations)), key=durations.__getitem__, reverse=True)
    longest = durations[jobs[0]]

    best_time = total
    best_groups = [jobs]

    for workers in range(2, min(8, len(jobs)) + 1):
        penalty = 2.0 * (workers - 1)

        # This bound can only increase as more workers are added.
        if longest + penalty >= best_time:
            break

        # Even perfectly balanced loads must meet this bound.
        if total / workers + penalty >= best_time:
            continue

        # Give each worker one test, then assign the rest by current load.
        groups = [[job] for job in jobs[:workers]]
        heap = [(durations[job], worker) for worker, job in enumerate(jobs[:workers])]
        heapify(heap)

        makespan = longest
        for job in jobs[workers:]:
            load, worker = heap[0]
            new_load = load + durations[job]
            groups[worker].append(job)
            heapreplace(heap, (new_load, worker))
            makespan = max(makespan, new_load)

        predicted_time = makespan + penalty
        if predicted_time < best_time:
            best_time = predicted_time
            best_groups = groups

    return best_time, best_groups


def pack(selected: set[int], seconds: dict[int, float]) -> list[list[int]]:
    """Return worker lists of test numbers. Empty when nothing is selected."""
    if not selected:
        return []
    ordered = sorted(selected, key=lambda number: (-seconds.get(number, 0.0), number))
    _predicted, groups = schedule_tests([seconds.get(number, 0.0) for number in ordered])
    return [[ordered[index] for index in group] for group in groups]
