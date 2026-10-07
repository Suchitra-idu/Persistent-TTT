"""Each document replayed `n_repeats` times with a carry that is never reset.

Repeat 0 is the cold pass, so it is every later repeat's paired reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.app import session_eval
from ttt.app.data_pipeline import Doc
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights


@dataclass(frozen=True)
class RepeatItem:
    source: str
    doc_idx: int
    repeat: int
    n_tokens: int
    ppl: float
    state_ratio: float
    gate_mean: float


def measure(
    *,
    docs: Sequence[Doc],
    compute: Compute,
    fast_weights: FastWeights,
    n_repeats: int,
) -> tuple[RepeatItem, ...]:
    if n_repeats < 2:
        raise ValueError(f"n_repeats must be >= 2 to have a repeat to compare, got {n_repeats}")
    rows: list[RepeatItem] = []
    for doc in docs:
        items = session_eval.session_perplexity(
            docs=(doc,) * n_repeats,
            compute=compute,
            fast_weights=fast_weights,
            n_slices=1,
            reset_between_items=False,
            reset_between_docs=False,
        )
        rows.extend(
            RepeatItem(
                source=item.source,
                doc_idx=item.doc_idx,
                repeat=item.position,
                n_tokens=item.n_tokens,
                ppl=item.ppl,
                state_ratio=item.state_ratio,
                gate_mean=item.gate_mean,
            )
            for item in items
        )
    return tuple(rows)
