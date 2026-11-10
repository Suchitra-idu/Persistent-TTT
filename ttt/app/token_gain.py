"""Per-token losses on the relatedness cases, so a gain can be located in the target.

Same protocol and arms as `relatedness_eval.measure`; only the loss is kept per
token instead of averaged, so the mean of a row's losses is that arm's loss there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ttt.app.relatedness_eval import GENERIC, IN_CONTEXT, LEVELS, RELATED, RESET, Case
from ttt.ports.compute import Compute
from ttt.core.types import Carry
from ttt.ports.fast_weights import FastWeights


@dataclass(frozen=True)
class TokenRow:
    arm: str
    doc_idx: int
    target_title: str
    losses: tuple[float, ...]


def measure(
    *,
    cases: Sequence[Case],
    compute: Compute,
    fast_weights: FastWeights,
    generic: Carry | None = None,
    in_context: bool = False,
) -> tuple[TokenRow, ...]:
    """The in-context row drops the target's first token, so every row scores the same positions."""
    fast_weights.set_mode(evolve=True, session=True)
    rows: list[TokenRow] = []
    for case in cases:
        fast_weights.reset_carry()
        rows.append(_row(RESET, case, compute.eval_token_losses(case.target)))
        for level in LEVELS:
            fast_weights.reset_carry()
            compute.eval_loss(case.primes[level])
            fast_weights.advance_carry()
            rows.append(_row(level, case, compute.eval_token_losses(case.target)))
        if generic is not None:
            fast_weights.reset_carry()
            fast_weights.install(generic)
            rows.append(_row(GENERIC, case, compute.eval_token_losses(case.target)))
        if in_context:
            fast_weights.reset_carry()
            prime = case.primes[RELATED]
            losses = compute.eval_token_losses(prime + case.target, score_from=len(prime))
            rows.append(_row(IN_CONTEXT, case, losses[1:]))
    fast_weights.reset_carry()
    return tuple(rows)


def _row(arm: str, case: Case, losses: Sequence[float]) -> TokenRow:
    return TokenRow(arm=arm, doc_idx=case.index, target_title=case.target_title, losses=tuple(losses))
