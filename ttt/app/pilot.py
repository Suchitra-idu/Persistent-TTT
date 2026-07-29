"""Per-document NLL against position in a synthetic session, under three regimes.

Cold against persist answers whether the carry compounds at all; seeded against
persist answers whether the trained seed is an offset or a different shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.app import session_eval
from ttt.app.data_pipeline import Doc
from ttt.core.types import Carry
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights

COLD = "cold"
PERSIST = "persist"
SEEDED = "seeded"
REGIMES = (COLD, PERSIST, SEEDED)


@dataclass(frozen=True)
class PilotRow:
    regime: str
    position: int
    doc_idx: int
    source: str
    n_tokens: int
    ppl: float
    nll: float
    state_ratio: float


def run(
    *,
    docs: Sequence[Doc],
    compute: Compute,
    fast_weights: FastWeights,
    seed: Carry | None = None,
) -> tuple[PilotRow, ...]:
    """The same document sequence three times, so position is the only variable."""
    rows = _regime(COLD, docs, compute, fast_weights, reset=True, carries=None)
    rows += _regime(PERSIST, docs, compute, fast_weights, reset=False, carries=None)
    if seed is not None:
        rows += _regime(
            SEEDED,
            docs,
            compute,
            fast_weights,
            reset=False,
            carries={doc.source: seed for doc in docs},
        )
    return rows


def _regime(
    regime: str, docs, compute, fast_weights, *, reset: bool, carries
) -> tuple[PilotRow, ...]:
    measured = session_eval.session_perplexity(
        docs=docs,
        compute=compute,
        fast_weights=fast_weights,
        n_slices=1,
        reset_between_docs=reset,
        carries=carries,
    )
    return tuple(
        PilotRow(
            regime=regime,
            position=row.position,
            doc_idx=row.doc_idx,
            source=row.source,
            n_tokens=row.n_tokens,
            ppl=row.ppl,
            nll=row.nll,
            state_ratio=row.state_ratio,
        )
        for row in measured
    )
