"""The carry across a document boundary, separated from the carry across slices.

One chain of same-source documents, three arms. `doc_reset` against `persist`
is the cross-document effect; `slice_reset` against `doc_reset` is the
within-document one. A chain's first document is identical under `persist`
and `doc_reset`, so the cross-document gap must skip it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.app import session_eval
from ttt.app.data_pipeline import Doc
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights

PERSIST = "persist"
DOC_RESET = "doc_reset"
SLICE_RESET = "slice_reset"

# (arm, reset between documents, reset between slices)
ARMS = (
    (PERSIST, False, False),
    (DOC_RESET, True, False),
    (SLICE_RESET, True, True),
)


@dataclass(frozen=True)
class ArmRow:
    arm: str
    source: str
    doc_idx: int
    chain_position: int
    slice_index: int
    n_tokens: int
    ppl: float
    state_ratio: float
    gate_mean: float


def measure(
    *,
    chains: Mapping[str, Sequence[Doc]],
    compute: Compute,
    fast_weights: FastWeights,
    n_slices: int,
) -> tuple[ArmRow, ...]:
    rows: list[ArmRow] = []
    for source in sorted(chains):
        docs = tuple(chains[source])
        for arm, reset_docs, reset_slices in ARMS:
            items = session_eval.session_perplexity(
                docs=docs,
                compute=compute,
                fast_weights=fast_weights,
                n_slices=n_slices,
                reset_between_items=reset_slices,
                reset_between_docs=reset_docs,
            )
            rows.extend(_tag(arm, docs, items))
    return tuple(rows)


def _tag(arm: str, docs: Sequence[Doc], items) -> list[ArmRow]:
    chain_position = {doc.index: position for position, doc in enumerate(docs)}
    seen: dict[int, int] = {}
    tagged = []
    for item in items:
        slice_index = seen.get(item.doc_idx, 0)
        seen[item.doc_idx] = slice_index + 1
        tagged.append(
            ArmRow(
                arm=arm,
                source=item.source,
                doc_idx=item.doc_idx,
                chain_position=chain_position[item.doc_idx],
                slice_index=slice_index,
                n_tokens=item.n_tokens,
                ppl=item.ppl,
                state_ratio=item.state_ratio,
                gate_mean=item.gate_mean,
            )
        )
    return tagged
