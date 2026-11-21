"""Indexes of the pinned sqlglot tree that every run needs, built once when the image is built.

Patches are applied in memory, so these depend only on the checkout at
`SQLGLOT_ROOT`. The build is not charged for CPU; a run that finds no cache,
or one made for another checkout, builds them itself.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from analyze import build_fixture_index, build_import_graph
from class_users import test_words
from edits import enum_names


def build(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "fixtures": build_fixture_index(root),
        "imports": build_import_graph(root),
        "enums": enum_names(root),
        "test_words": test_words(root),
    }


def load(path: Path, root: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if isinstance(state, dict) and state.get("root") == str(root):
            return state
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        pass
    return build(root)


def write(path: Path, root: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(build(root), handle, protocol=pickle.HIGHEST_PROTOCOL)
