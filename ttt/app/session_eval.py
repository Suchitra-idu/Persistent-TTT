"""Per-item perplexity with the carry persisting across items.

The reset ladder is the whole point: `reset_between_items` isolates within-item
adaptation, `reset_between_docs` matches training's per-doc start, and neither
lets the carry compound across the eval unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.app.data_pipeline import Doc
from ttt.core import metrics, schedule
from ttt.core.types import Carry
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import CARRY, FastWeights


@dataclass(frozen=True)
class ItemRow:
    doc_idx: int
    source: str
    position: int
    start: int
    end: int
    ppl: float
    state_ratio: float

    @property
    def n_tokens(self) -> int:
        return self.end - self.start

    @property
    def nll(self) -> float:
        return math.log(self.ppl)


def session_perplexity(
    *,
    docs: Sequence[Doc],
    compute: Compute,
    fast_weights: FastWeights,
    n_slices: int = 1,
    evolve: bool = True,
    reset_between_items: bool = False,
    reset_between_docs: bool = True,
    carries: Mapping[str, Carry] | None = None,
    force_source: str = "",
) -> tuple[ItemRow, ...]:
    """`force_source` installs that source's carrier whatever the doc is — the
    swap test: a source-specific benefit should degrade under a mismatch."""
    fast_weights.set_mode(evolve=evolve, stream=False, session=True)
    seeds = carries or {}

    rows: list[ItemRow] = []
    position = 0
    for index, doc in enumerate(docs):
        if index == 0 or reset_between_docs:
            _restart(fast_weights, seeds, force_source or doc.source)
        for start, end in schedule.equal_token_slices(doc.n_tokens, n_slices):
            loss = compute.eval_loss(doc.token_ids[start:end])
            fast_weights.advance_carry()
            rows.append(
                ItemRow(
                    doc_idx=doc.index,
                    source=doc.source,
                    position=position,
                    start=start,
                    end=end,
                    ppl=metrics.perplexity(loss),
                    state_ratio=fast_weights.state_ratio(family=CARRY),
                )
            )
            position += 1
            if reset_between_items:
                _restart(fast_weights, seeds, force_source or doc.source)
    return tuple(rows)


def _restart(fast_weights, seeds: Mapping[str, Carry], source: str) -> None:
    fast_weights.reset_carry()
    seed = seeds.get(source)
    if seed is not None:
        fast_weights.install(seed)
