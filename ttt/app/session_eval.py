"""Per-item perplexity with the carry persisting across items.

The reset ladder is the whole point: `reset_between_items` isolates within-item
adaptation, `reset_between_docs` matches training's per-doc start, and neither
lets the carry compound across the eval unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ttt.app.data_pipeline import Doc
from ttt.core import metrics, schedule
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights


@dataclass(frozen=True)
class ItemRow:
    doc_idx: int
    source: str
    position: int
    start: int
    end: int
    ppl: float
    state_ratio: float
    gate_mean: float = 1.0  # fraction of the TTT term reaching the output; 1.0 = ungated

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
) -> tuple[ItemRow, ...]:
    fast_weights.set_mode(evolve=evolve, session=True)

    rows: list[ItemRow] = []
    position = 0
    for index, doc in enumerate(docs):
        if index == 0 or reset_between_docs:
            fast_weights.reset_carry()
        for start, end in schedule.equal_token_slices(doc.n_tokens, n_slices):
            loss = compute.eval_loss(doc.token_ids[start:end])
            fast_weights.advance_carry()
            gate_mean = fast_weights.gate_mean()
            rows.append(
                ItemRow(
                    doc_idx=doc.index,
                    source=doc.source,
                    position=position,
                    start=start,
                    end=end,
                    ppl=metrics.perplexity(loss),
                    state_ratio=fast_weights.state_ratio(),
                    gate_mean=1.0 if gate_mean is None else gate_mean,
                )
            )
            position += 1
            if reset_between_items:
                fast_weights.reset_carry()
    return tuple(rows)
