"""Tests that can observe an import-time edit inside a sqlglot class body.

`maps/classes.json`, built by `collect_classes.py`, records for each test every
class in the MRO of the parsers, generators, tokenizers, dialects and type
annotators it used. A class attribute is only seen through such an object, so
the tests that used the edited class or a subclass bound what can fail. When
every read of the attribute is a line inside an ordinary method, the tests
that executed one of those lines narrow it further. An entry of a generator's
`TRANSFORMS` is read through the cached dispatch table, so the recorded
dispatch lookups of its expression class stand in for that read. A table a
metaclass derives from the attribute onto the class it creates (a trie from a
mapping) is read wherever the derived attribute is.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path

from analyze import assigned_names, functions_and_classes, module_name, statement_at, tightest
from class_reads import (
    at_class_creation,
    chain_tail,
    constant_truth,
    copied_owners,
    derived_attribute,
    dict_entry_key,
    dynamic_access,
    ever_executed,
    in_lambda_body,
    in_place,
    keys_only,
    memoized,
    name_use,
    scopes,
    statement_on,
)

Coverage = dict[str, dict[int, set[int]]]
Mentions = Callable[[str], "list[tuple[str, int]] | None"]
# `collect_classes.py` records the lookups into the table this function builds.
_DISPATCH_SITE = ("sqlglot/generator.py", "_build_dispatch")
# Tables derived from tables while classes are created, followed before giving up.
_MAX_DERIVATIONS = 8


class ClassUsers:
    def __init__(self, path: Path, numbers: dict[str, int], root: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        classes: list[str] = payload.get("classes", [])
        nodeids = list(payload.get("tests", {}))
        self.root = root
        self.numbers = numbers
        self.test_words: dict[str, list[str]] | None = None
        self.users: dict[str, set[int]] = {}
        for nodeid, indexes in payload.get("tests", {}).items():
            number = numbers.get(nodeid)
            if number is None:
                continue
            for index in indexes:
                self.users.setdefault(classes[index], set()).add(number)
        self.mro = {classes[int(key)]: [classes[i] for i in chain] for key, chain in payload.get("mro", {}).items()}
        self._nodeids = nodeids
        self._dispatch = {classes[int(owner)]: names for owner, names in payload.get("dispatch", {}).items()}
        self._tables = {classes[int(owner)]: attrs for owner, attrs in payload.get("tables", {}).items()}
        self._tracked = {classes[int(owner)]: set(attrs) for owner, attrs in payload.get("tracked", {}).items()}
        self._trees: dict[str, ast.AST | None] = {}

    def _numbers(self, positions: list[int]) -> set[int]:
        found = (self.numbers.get(self._nodeids[p]) for p in positions)
        return {number for number in found if number is not None}

    def _dispatched(self, sources: set[str], entries: set[str]) -> set[int]:
        """Tests whose generator, built on one of `sources`, looked up one of the expression classes `entries`."""
        selected: set[int] = set()
        for owner, names in self._dispatch.items():
            if sources.intersection(self.mro.get(owner, [owner])):
                for entry in entries:
                    selected |= self._numbers(names.get(entry, []))
        return selected

    def _keyed(self, sources: set[str], name: str, entries: set[str]) -> set[int] | None:
        """Tests that looked up one of `entries`, or read the whole table, in `name` of one of `sources`.

        None unless every source's table was tracked.
        """
        selected: set[int] = set()
        for source in sources:
            if name not in self._tracked.get(source, ()):
                return None
            table = self._tables.get(source, {}).get(name, {})
            for entry in entries | {"*"}:
                selected |= self._numbers(table.get(entry, []))
        return selected

    def select(
        self,
        path: str,
        tree: ast.AST,
        line: int,
        coverage: Coverage,
        mentions: Mentions,
        original: bool = True,
        renamed_keys: set[str] | None = None,
    ) -> set[int] | None:
        """Tests for an edit at `line` of a class body; None when no recorded class owns it.

        `original` says `tree` is the code the tests ran, so its value is what
        they read. `renamed_keys` holds the old and new names of dict keys the
        edit renamed; None when it may have added or removed entries.
        """
        key = class_key(path, tree, line)
        if key is None or key not in self.users:
            return None
        observers = set(self.users[key])
        statement = statement_at(tree, line)
        names = assigned_names(statement)
        if len(names) != 1:
            return observers
        observers |= self._test_mentions(names[0])
        truthy = constant_truth(statement) if original else None
        entry = dict_entry_key(statement, line) if renamed_keys is not None else None
        entries = {entry} | (renamed_keys or set()) if entry is not None else set()
        values_only = bool(entries) and not renamed_keys
        proof = self._readers(names[0], key, coverage, mentions, truthy, bool(entries), values_only)
        if proof is None:
            return observers
        readers, copiers, via_dispatch, derived = proof
        for copier in copiers:
            if copier not in self.users:
                return observers | self._all_users()
            observers |= self.users[copier]
        if entries and not derived:
            sources = {key} | copiers
            if via_dispatch:
                readers |= self._dispatched(sources, entries)
            else:
                keyed = self._keyed(sources, names[0], entries)
                if keyed is not None:
                    readers = keyed
        return (readers & observers) | self._test_mentions(names[0])

    def _test_mentions(self, name: str) -> set[int]:
        """Tests whose own method names `name`; they can read the attribute without using the class."""
        if self.test_words is None:
            self.test_words = test_words(self.root)
        found = (self.numbers.get(nodeid) for nodeid in self.test_words.get(name, ()))
        return {number for number in found if number is not None}

    def _all_users(self) -> set[int]:
        return set().union(*self.users.values()) if self.users else set()

    def _may_descend(self, simple: str | None, sources: set[str]) -> bool:
        """Whether a class named `simple` can be one of `sources` or inherit from one."""
        if simple is None:
            return True
        named = [other for other in self.mro if other.rsplit(":", 1)[-1].split(".")[-1] == simple]
        return not named or any(sources.intersection(self.mro[other]) for other in named)

    def _copiers(
        self, name: str, key: str, sites: list[tuple[str, ast.AST, int]], values_only: bool
    ) -> set[str] | None:
        """Classes whose body copies `name` from the edited class, a subclass, or another copier.

        None when a class body derives something else from it, unless that only
        reads keys and the edit changed only values.
        """
        copiers: set[str] = set()
        pending = list(sites)
        changed = True
        while changed:
            changed = False
            for site in list(pending):
                relative, tree, lineno = site
                if not any(self._may_descend(owner, {key} | copiers) for owner in copied_owners(tree, lineno, name)):
                    continue
                pending.remove(site)
                copier = class_key(relative, tree, lineno, scopes)
                if copier is None or assigned_names(statement_on(tree, lineno)) != [name]:
                    if values_only and keys_only(tree, lineno, name):
                        continue
                    return None
                copiers.add(copier)
                changed = True
        return copiers

    def _readers(
        self,
        name: str,
        key: str,
        coverage: Coverage,
        mentions: Mentions,
        truthy: bool | None,
        entry: bool,
        values_only: bool,
        derivations: tuple[tuple[str, int, int], ...] = (),
    ) -> tuple[set[int], set[str], bool, bool] | None:
        """Tests executing a method line that loads `name`, classes whose body copies the edited
        class's value, whether the generator's dispatch table is one of the reads, and whether a
        table derived from `name` while classes are created was followed.

        None when a read cannot be traced to such a line. With the truth of the
        value the tests read, a read that only lets a short-circuit chain go on
        counts the tests that reached the chain's last operand. When the edit
        changed only dict values, reads at class creation that use only the keys
        cannot see it. A read at class creation that sets `klass.X` on the class
        created stands for the readers of `X`; `derivations` holds the statements
        so far, whose own reads see the value before it is set.
        """
        found = mentions(name)
        if found is None:
            return None
        readers: set[int] = set()
        class_sites: list[tuple[str, ast.AST, int]] = []
        via_dispatch = False
        derived: dict[str, list[tuple[str, int, int]]] = {}
        for relative, lineno in found:
            if any(relative == path and start <= lineno <= end for path, start, end in derivations):
                continue
            tree = self._tree(relative)
            if tree is None:
                return None
            use = name_use(tree, lineno, name)
            if use is None:
                if dynamic_access(tree, lineno, name):
                    return None
                continue
            if use == "store":
                continue
            functions, _classes = scopes(tree)
            function = tightest(functions, lineno)
            if function is None and in_lambda_body(tree, lineno):
                readers |= coverage.get(relative, {}).get(lineno, set())
                continue
            if function is None:
                class_sites.append((relative, tree, lineno))
                continue
            statement = statement_on(tree, lineno)
            if at_class_creation(tree, function) and not in_place(statement, name):
                if values_only and keys_only(tree, lineno, name):
                    continue
                target = derived_attribute(function, statement, name)
                if target is None or statement is None:
                    return None
                span = (relative, statement.lineno, statement.end_lineno or statement.lineno)  # type: ignore[attr-defined]
                derived.setdefault(target, []).append(span)
                continue
            if memoized(tree, function):
                if entry and (relative, function.name) == _DISPATCH_SITE:  # type: ignore[attr-defined]
                    via_dispatch = True
                    continue
                return None
            owners = coverage.get(relative, {}).get(lineno, set())
            if not owners and not getattr(coverage, "complete", False) and not ever_executed(coverage, relative, function):
                return None
            if truthy is not None:
                last = chain_tail(tree, lineno, name, truthy)
                if last is not None:
                    owners = coverage.get(relative, {}).get(last, set())
            readers |= owners
        copiers = self._copiers(name, key, class_sites, values_only)
        if copiers is None:
            return None
        for target, spans in derived.items():
            if len(derivations) + len(spans) > _MAX_DERIVATIONS:
                return None
            proof = self._readers(target, key, coverage, mentions, None, False, False, (*derivations, *spans))
            if proof is None:
                return None
            readers |= proof[0]
            copiers |= proof[1]
            via_dispatch |= proof[2]
        return readers, copiers, via_dispatch, bool(derived)

    def _tree(self, relative: str) -> ast.AST | None:
        if relative not in self._trees:
            try:
                self._trees[relative] = ast.parse((self.root / relative).read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                self._trees[relative] = None
        return self._trees[relative]


def test_words(root: Path) -> dict[str, list[str]]:
    """Every name or attribute a test method's own code uses, with the ids of the tests using it."""
    words: dict[str, list[str]] = {}
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or not item.name.startswith("test"):
                    continue
                nodeid = f"{relative}::{node.name}::{item.name}"
                seen = set()
                for child in ast.walk(item):
                    word = child.attr if isinstance(child, ast.Attribute) else getattr(child, "id", None)
                    if isinstance(word, str) and word not in seen:
                        seen.add(word)
                        words.setdefault(word, []).append(nodeid)
    return words


def class_key(
    path: str,
    tree: ast.AST,
    line: int,
    index: Callable[[ast.AST], tuple[list[ast.AST], list[ast.AST]]] = functions_and_classes,
) -> str | None:
    """`module:Outer.Inner` of the class bodies around `line`, outside any function."""
    functions, classes = index(tree)
    if tightest(functions, line) is not None:
        return None
    around = [node for node in classes if node.lineno <= line <= (node.end_lineno or node.lineno)]
    if not around:
        return None
    around.sort(key=lambda node: node.lineno)
    return f"{module_name(path)}:{'.'.join(node.name for node in around)}"  # type: ignore[attr-defined]
