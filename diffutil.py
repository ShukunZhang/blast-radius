"""Parse and apply unified diffs from the blast-radius input."""

from __future__ import annotations

from dataclasses import dataclass, field


class DiffApplyError(Exception):
    """The patch does not match the file it claims to modify."""


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FileDiff:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str | None:
        return self.new_path or self.old_path


def _strip_ab(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse_patch(text: str) -> list[FileDiff]:
    """Split one unified diff into per-file hunks."""
    lines = text.splitlines()
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            current = FileDiff(old_path=None, new_path=None)
            files.append(current)
            hunk = None
            parts = line.split()
            if len(parts) >= 4:
                current.old_path = _strip_ab(parts[2])
                current.new_path = _strip_ab(parts[3])
        elif current is not None and line.startswith("--- "):
            path = line[4:].split("\t", 1)[0].strip()
            current.old_path = None if path == "/dev/null" else _strip_ab(path)
        elif current is not None and line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0].strip()
            current.new_path = None if path == "/dev/null" else _strip_ab(path)
        elif current is not None and line.startswith("@@"):
            hunk = _parse_hunk_header(line)
            current.hunks.append(hunk)
        elif current is not None and line.startswith("Binary files"):
            current.binary = True
        elif hunk is not None and line.startswith("\\"):
            pass
        elif hunk is not None and line[:1] in {" ", "+", "-"}:
            hunk.lines.append((line[:1], line[1:]))
        index += 1
    return files


def _parse_hunk_header(line: str) -> Hunk:
    # @@ -l,c +l,c @@ optional text
    body = line.split("@@")[1].strip()
    old_spec, new_spec = body.split(" ", 1)[0], body.split(" ", 1)[1].split(" ", 1)[0]
    old_start, old_count = _parse_range(old_spec[1:])
    new_start, new_count = _parse_range(new_spec[1:])
    return Hunk(old_start, old_count, new_start, new_count)


def _parse_range(spec: str) -> tuple[int, int]:
    if "," in spec:
        start_text, count_text = spec.split(",", 1)
        return int(start_text), int(count_text)
    return int(spec), 1


def apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """Return the file after applying hunks. Raises DiffApplyError on mismatch."""
    original_lines = original.splitlines()
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        start = max(hunk.old_start - 1, 0)
        if start < cursor:
            raise DiffApplyError("overlapping hunk")
        output.extend(original_lines[cursor:start])
        cursor = start
        for kind, text in hunk.lines:
            if kind == " ":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    raise DiffApplyError(f"context mismatch at line {cursor + 1}")
                output.append(original_lines[cursor])
                cursor += 1
            elif kind == "-":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    raise DiffApplyError(f"delete mismatch at line {cursor + 1}")
                cursor += 1
            else:
                output.append(text)
    output.extend(original_lines[cursor:])
    joined = "\n".join(output)
    if original.endswith("\n") or original == "":
        if output:
            joined += "\n"
    return joined


def removed_lines(hunks: list[Hunk]) -> list[tuple[int, str]]:
    """Old file line numbers and text for deleted or replaced lines."""
    found: list[tuple[int, str]] = []
    for hunk in hunks:
        old_line = hunk.old_start
        for kind, text in hunk.lines:
            if kind == " ":
                old_line += 1
            elif kind == "-":
                found.append((old_line, text))
                old_line += 1
    return found


def added_lines(hunks: list[Hunk]) -> list[tuple[int, str]]:
    """New file line numbers and text for inserted or replaced lines."""
    found: list[tuple[int, str]] = []
    for hunk in hunks:
        new_line = hunk.new_start
        for kind, text in hunk.lines:
            if kind == " ":
                new_line += 1
            elif kind == "+":
                found.append((new_line, text))
                new_line += 1
    return found
