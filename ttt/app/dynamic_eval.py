"""Dynamic evaluation on the relatedness cases: the baseline for the carry.

Instead of carrying a fast-weight state, the model takes gradient steps on the
prime and reads the target with the adapted weights (Krause et al. 2018),
restored after every case. The arms, cases and estimates are the relatedness
suite's, so the two gradients compare case for case. Settings are chosen on
cases disjoint from the ones reported, once for the same-document arm and once
for the related arm: aggressive adaptation wins the first and hurts the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from ttt.app import evidence
from ttt.app.relatedness_eval import LEVELS, RELATED, RESET, SELF, Case, RelatedRow, make_row
from ttt.core import stats
from ttt.ports.compute import Compute


class Adapter(Protocol):
    def adapt(self, token_ids: Sequence[int], *, optimizer: str, lr: float, steps: int) -> None:
        """Gradient steps on next-token loss over `token_ids`, in place."""

    def restore(self) -> None:
        """Back to the weights the adapter was built with."""


@dataclass(frozen=True)
class Setting:
    optimizer: str
    lr: float
    steps: int

    @property
    def label(self) -> str:
        return f"{self.optimizer} lr={self.lr:g} steps={self.steps}"

    @property
    def text(self) -> str:
        return f"{self.optimizer}:{self.lr:g}:{self.steps}"

    @classmethod
    def parse(cls, text: str) -> "Setting":
        optimizer, lr, steps = text.split(":")
        return cls(optimizer, float(lr), int(steps))


def parse_settings(text: str) -> tuple[Setting, ...]:
    """`adam:1e-05:10,adam:3e-05:10`; empty is none."""
    return tuple(Setting.parse(t) for t in text.split(",") if t.strip())


def parse_objectives(text: str) -> dict[str, Setting]:
    """`self=adam:3e-05:10;related=adam:1e-05:10`; empty is none."""
    pairs = (t.split("=") for t in text.split(";") if t.strip())
    return {objective: Setting.parse(setting) for objective, setting in pairs}


@dataclass(frozen=True)
class TuneRow:
    optimizer: str
    lr: float
    steps: int
    arm: str
    source: str
    doc_idx: int
    n_tokens: int
    ppl: float


OBJECTIVES = (SELF, RELATED)
STEPS = (3, 10, 30)
# Brackets the optimum of two coarser searches: Adam diverged from 3e-4 up, and plain
# SGD stayed far below Adam at every step size tried (results/lora-only-matched).
LRS = {"adam": (3e-6, 1e-5, 3e-5, 1e-4)}


def grid(lrs: Mapping[str, Sequence[float]] = LRS, steps: Sequence[int] = STEPS) -> tuple[Setting, ...]:
    return tuple(Setting(o, lr, n) for o, values in lrs.items() for lr in values for n in steps)


def _adapted_loss(case: Case, level: str, compute: Compute, adapter: Adapter, setting: Setting) -> float:
    adapter.adapt(case.primes[level], optimizer=setting.optimizer, lr=setting.lr, steps=setting.steps)
    try:
        return compute.eval_loss(case.target)
    finally:
        adapter.restore()


def tune(
    *, cases: Sequence[Case], compute: Compute, adapter: Adapter, settings: Sequence[Setting]
) -> tuple[TuneRow, ...]:
    """Every setting on every objective arm of the tuning cases."""
    rows: list[TuneRow] = []
    for case in cases:
        reset = compute.eval_loss(case.target)
        for setting in settings:
            scored = [(RESET, reset)] + [
                (arm, _adapted_loss(case, arm, compute, adapter, setting)) for arm in OBJECTIVES
            ]
            for arm, loss in scored:
                r = make_row(arm, case, "", loss)
                rows.append(TuneRow(setting.optimizer, setting.lr, setting.steps, r.arm, r.source,
                                    r.doc_idx, r.n_tokens, r.ppl))
    return tuple(rows)


def gain(rows: Sequence[TuneRow], setting: Setting, arm: str) -> float:
    picked = [r for r in rows if (r.optimizer, r.lr, r.steps) == (setting.optimizer, setting.lr, setting.steps)]
    reset = evidence.per_doc(picked, where=lambda r: r.arm == RESET)
    adapted = evidence.per_doc(picked, where=lambda r: r.arm == arm)
    keys = sorted(set(reset) & set(adapted))
    return stats.ppl_gap([reset[k] for k in keys], [adapted[k] for k in keys])


def settings_in(rows: Sequence[TuneRow]) -> list[Setting]:
    return [Setting(*key) for key in sorted({(r.optimizer, r.lr, r.steps) for r in rows})]


def best(rows: Sequence[TuneRow], arm: str = SELF) -> Setting:
    """The setting with the largest pooled gain on `arm`; the first on ties."""
    candidates = settings_in(rows)
    return max(candidates, key=lambda s: (gain(rows, s, arm), -candidates.index(s)))


def measure(
    *, cases: Sequence[Case], compute: Compute, adapter: Adapter, setting: Setting
) -> tuple[RelatedRow, ...]:
    rows: list[RelatedRow] = []
    for case in cases:
        rows.append(make_row(RESET, case, "", compute.eval_loss(case.target)))
        for level in LEVELS:
            loss = _adapted_loss(case, level, compute, adapter, setting)
            rows.append(make_row(level, case, case.prime_titles[level], loss))
    return tuple(rows)
