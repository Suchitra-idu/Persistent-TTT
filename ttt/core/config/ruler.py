"""RulerConfig — the RULER sweep's knobs. Frozen (D3).

Kept out of `cli.Resolved`: the prepare phase needs no model, and a Modal
entrypoint can only take one flat string per parameter, so this gets its own
flag string (`ruler_flags`) the same way `flags` carries `cli.Resolved`'s.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ttt.core.config import resolve as resolve_mod
from ttt.core.types import COLD_CARRY, FRESH

DEFAULT_TASKS = ("niah_single",)
DEFAULT_LENGTHS = (4096,)
DEFAULT_REGIMES = (FRESH, COLD_CARRY)

BASE = "base"
INSTRUCT = "instruct"
PROMPT_STYLES = (BASE, INSTRUCT)

_LIST_FIELDS = ("tasks", "context_lengths", "regimes")
_INT_FIELDS = ("num_samples", "max_new_tokens", "seed")
_STR_FIELDS = ("prompt_style",)


@dataclass(frozen=True)
class RulerConfig:
    tasks: tuple[str, ...] = DEFAULT_TASKS
    context_lengths: tuple[int, ...] = DEFAULT_LENGTHS
    num_samples: int = 20
    max_new_tokens: int = 64
    seed: int = 0
    regimes: tuple[str, ...] = DEFAULT_REGIMES
    # "base": prompt + answer_prefix, a raw completion cue. "instruct": through
    # the tokenizer's chat template instead — wired but unused until we have
    # an instruct checkpoint to point this at.
    prompt_style: str = BASE

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("RulerConfig needs at least one task")
        if not self.context_lengths or any(n <= 0 for n in self.context_lengths):
            raise ValueError(
                f"context_lengths must all be > 0, got {self.context_lengths}"
            )
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not self.regimes or any(r not in (FRESH, COLD_CARRY) for r in self.regimes):
            raise ValueError(
                f"regimes must be a subset of ({FRESH!r}, {COLD_CARRY!r}), "
                f"got {self.regimes}"
            )
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"prompt_style must be one of {PROMPT_STYLES}, got {self.prompt_style!r}"
            )
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "context_lengths", tuple(self.context_lengths))
        object.__setattr__(self, "regimes", tuple(self.regimes))

    def key(self, task: str, length: int) -> str:
        """The prepared-example filename stem: one JSONL per task x length."""
        return f"{task}_{length}_n{self.num_samples}_s{self.seed}"


def parse_ruler_flags(text: str) -> RulerConfig:
    """`"tasks=niah_single:vt,context_lengths=4096:8192,num_samples=20"` -> RulerConfig.

    Colon-separates a list field's items; everything else is `cli.parse_flags`'s
    `name=value` comma-joined shape, because Modal only carries flat strings.
    """
    given: dict[str, object] = {}
    for part in text.split(","):
        pair = part.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"expected name=value, got {pair!r}")
        name, _, value = pair.partition("=")
        given[name.strip()] = _coerce(name.strip(), value.strip())
    return resolve_mod.merge(RulerConfig(), given)


def _coerce(name: str, value: str):
    if name in _LIST_FIELDS:
        items = tuple(item.strip() for item in value.split(":") if item.strip())
        return tuple(int(item) for item in items) if name == "context_lengths" else items
    if name in _INT_FIELDS:
        return int(value)
    if name in _STR_FIELDS:
        return value
    known = {f.name for f in dataclasses.fields(RulerConfig)}
    raise ValueError(f"unknown ruler flag {name!r}; known flags are {sorted(known)}")
