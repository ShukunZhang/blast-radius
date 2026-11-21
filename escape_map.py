"""Record tests that fetch values from baseline numeric escape configuration."""
import sys
from pathlib import Path

ROOT = Path.cwd()
hits = set()
complete = False


def install():
    from sqlglot.tokenizer_core import TokenizerCore
    from sqlglot.tokens import Tokenizer

    original = TokenizerCore.__init__
    owners = {}
    ready = {}

    def locate(mapping):
        pending = [Tokenizer]
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            value = cls.__dict__.get("NUMERIC_ESCAPES")
            if isinstance(value, dict) and value:
                module = sys.modules.get(cls.__module__)
                filename = Path(getattr(module, "__file__", "")).resolve()
                if filename.is_relative_to(ROOT):
                    owners.setdefault(id(value), set()).add(str(filename.relative_to(ROOT)))
        return owners.get(id(mapping), set())

    class Traced(dict):
        def __init__(self, source, paths):
            super().__init__(source)
            self.paths = paths

        def get(self, key, *default):
            for path in self.paths:
                hits.add((path, key))
            return super().get(key, *default)

    def init(self, *args, **kwargs):
        mapping = kwargs.get("numeric_escapes")
        if mapping:
            pointer = id(mapping)
            if pointer not in ready:
                ready[pointer] = Traced(mapping, locate(mapping))
            kwargs["numeric_escapes"] = ready[pointer]
        return original(self, *args, **kwargs)

    TokenizerCore.__init__ = init


def changed_keys(old, new):
    """Recognize a value-only edit to the numeric escape config dictionary."""
    import ast

    old_declarations = [n for n in ast.walk(old) if isinstance(n, ast.Assign)
                        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "NUMERIC_ESCAPES"]
    new_declarations = [n for n in ast.walk(new) if isinstance(n, ast.Assign)
                        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "NUMERIC_ESCAPES"]
    if len(old_declarations) != 1 or len(new_declarations) != 1:
        return None
    left, right = old_declarations[0], new_declarations[0]
    try:
        before, after = ast.literal_eval(left.value), ast.literal_eval(right.value)
    except (ValueError, TypeError):
        return None
    if not isinstance(before, dict) or not isinstance(after, dict) or before.keys() != after.keys():
        return None
    if not all(isinstance(k, str) and isinstance(v, tuple) and len(v) == 4
               and all(type(n) is int for n in v) for mapping in (before, after)
               for k, v in mapping.items()):
        return None
    restore = right.value
    right.value = left.value
    unchanged = ast.dump(old) == ast.dump(new)
    right.value = restore
    return {key for key in before if before[key] != after[key]} if unchanged else None
