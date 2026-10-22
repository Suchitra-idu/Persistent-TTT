"""The carried state against the text it stands in for.

Each slice after the first is scored three ways on exactly the same tokens:
alone with the state reset, alone with the state carried from earlier slices,
and with the previous slice placed before it in attention. The carry's share
of the context's gain is how much of that text a constant-size state recovers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.app import evidence
from ttt.app.data_pipeline import Doc
from ttt.core import metrics, schedule, stats
from ttt.core.stats import Estimate, Interval
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.rng import Rng

RESET = "reset"
CARRIED = "carried"
CONTEXT = "context"
ARMS = (RESET, CARRIED, CONTEXT)

CARRY_GAIN = "carry_gain"
CONTEXT_GAIN = "context_gain"
CONTEXT_OVER_CARRY = "context_over_carry"


@dataclass(frozen=True)
class ContextRow:
    arm: str
    source: str
    doc_idx: int
    slice_index: int
    n_tokens: int
    ppl: float


def measure(
    *, docs: Sequence[Doc], compute: Compute, fast_weights: FastWeights, n_slices: int
) -> tuple[ContextRow, ...]:
    """The first token of a slice is never scored: alone it has nothing to be
    predicted from, so the context arm skips it too and every arm scores the same tokens."""
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    rows: list[ContextRow] = []
    for doc in docs:
        bounds = schedule.equal_token_slices(doc.n_tokens, n_slices)
        pieces = [tuple(doc.token_ids[start:end]) for start, end in bounds]
        fast_weights.reset_carry()
        for index, piece in enumerate(pieces):
            loss = compute.eval_loss(piece)
            fast_weights.advance_carry()
            if index:
                rows.append(_row(CARRIED, doc, index, piece, loss))
        for index in range(1, len(pieces)):
            fast_weights.reset_carry()
            rows.append(_row(RESET, doc, index, pieces[index], compute.eval_loss(pieces[index])))
            fast_weights.reset_carry()
            before = pieces[index - 1]
            loss = compute.eval_loss(before + pieces[index], score_from=len(before) + 1)
            rows.append(_row(CONTEXT, doc, index, pieces[index], loss))
    fast_weights.reset_carry()
    return tuple(rows)


def _row(arm: str, doc: Doc, index: int, piece: Sequence[int], loss: float) -> ContextRow:
    return ContextRow(
        arm=arm, source=doc.source, doc_idx=doc.index, slice_index=index,
        n_tokens=max(len(piece) - 1, 1), ppl=metrics.perplexity(loss),
    )


def estimates(rows: Sequence[ContextRow], *, rng: Rng, n_boot: int) -> tuple[Estimate, ...]:
    def arm(name: str):
        return evidence.per_doc(rows, where=lambda r: r.arm == name)

    kw = dict(rng=rng, n_boot=n_boot)
    return (
        evidence.paired_estimates(CARRY_GAIN, arm(RESET), arm(CARRIED), **kw)
        + evidence.paired_estimates(CONTEXT_GAIN, arm(RESET), arm(CONTEXT), **kw)
        + evidence.paired_estimates(CONTEXT_OVER_CARRY, arm(CARRIED), arm(CONTEXT), **kw)
    )


def carry_share(rows: Sequence[ContextRow], *, rng: Rng, n_boot: int) -> Interval:
    """The carry's gain over the context's gain, pooled over documents, with
    both resampled on the same documents."""
    def arm(name: str):
        return evidence.per_doc(rows, where=lambda r: r.arm == name)

    reset, carried, context = arm(RESET), arm(CARRIED), arm(CONTEXT)
    keys = sorted(set(reset) & set(carried) & set(context))
    r = [reset[k] for k in keys]
    c = [carried[k] for k in keys]
    x = [context[k] for k in keys]

    def share(indices: Sequence[int]) -> float:
        pick = lambda values: [values[i] for i in indices]
        context_gain = stats.ppl_gap(pick(r), pick(x))
        return stats.ppl_gap(pick(r), pick(c)) / context_gain if context_gain else float("nan")

    return stats.bootstrap(share, len(keys), evidence.resamples(len(keys), n_boot, rng))
