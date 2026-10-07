"""A carry built up at runtime on one source, read on another source's document.

No trained seed is involved: every session starts from zero. If the
cross-document effect is source-specific, same-source priming should beat
other-source priming on the same target document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.app import session_eval
from ttt.app.data_pipeline import Doc
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights

UNPRIMED = ""


@dataclass(frozen=True)
class SwapRow:
    target_source: str
    prime_source: str
    doc_idx: int
    n_tokens: int
    ppl: float
    state_ratio: float


def measure(
    *,
    by_source: Mapping[str, Sequence[Doc]],
    compute: Compute,
    fast_weights: FastWeights,
    n_primes: int,
    n_targets: int,
) -> tuple[SwapRow, ...]:
    """A (prime, target) pair short of `n_primes` documents is skipped, not
    run with fewer: unequal priming would confound the comparison."""
    if n_primes < 1:
        raise ValueError(f"n_primes must be >= 1, got {n_primes}")
    rows: list[SwapRow] = []
    for target in sorted(by_source):
        for doc in tuple(by_source[target])[:n_targets]:
            rows.append(_read(target, UNPRIMED, (), doc, compute, fast_weights))
            for prime in sorted(by_source):
                primes = [d for d in by_source[prime] if d.index != doc.index][:n_primes]
                if len(primes) == n_primes:
                    rows.append(_read(target, prime, primes, doc, compute, fast_weights))
    return tuple(rows)


def _read(target, prime, primes, doc, compute, fast_weights) -> SwapRow:
    items = session_eval.session_perplexity(
        docs=(*primes, doc),
        compute=compute,
        fast_weights=fast_weights,
        n_slices=1,
        reset_between_docs=False,
    )
    last = items[-1]
    return SwapRow(
        target_source=target,
        prime_source=prime,
        doc_idx=doc.index,
        n_tokens=last.n_tokens,
        ppl=last.ppl,
        state_ratio=last.state_ratio,
    )
