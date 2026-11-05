"""One target scored after a reset carry and after a carry primed on one text, per token."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights

BLOCK = 1024
FRONT = 32


@dataclass(frozen=True)
class TokenView:
    """`reset[i]` and `carry[i]` are the losses on target token i + 1, in nats."""

    reset: tuple[float, ...]
    carry: tuple[float, ...]

    @property
    def gains(self) -> tuple[float, ...]:
        return tuple(r - c for r, c in zip(self.reset, self.carry))

    def summary(self) -> dict:
        n = len(self.reset)
        gains = self.gains
        total = sum(gains)
        mean_reset, mean_carry = sum(self.reset) / n, sum(self.carry) / n
        return {
            "tokens": n,
            "ppl_reset": math.exp(mean_reset),
            "ppl_carry": math.exp(mean_carry),
            "ppl_change_pct": 100 * (1 - math.exp(mean_carry - mean_reset)),
            "gain_per_token": total / n,
            "front_gain_per_token": sum(gains[: FRONT - 1]) / max(1, min(n, FRONT - 1)),
            "rest_gain_per_token": sum(gains[FRONT - 1 :]) / max(1, n - FRONT + 1),
        }


def measure(
    *, prime: Sequence[int], target: Sequence[int], compute: Compute, fast_weights: FastWeights
) -> TokenView:
    """The relatedness protocol: the prime is read and folded into the carry, never attended."""
    if len(target) < 2:
        raise ValueError("the target needs at least two tokens")
    if not prime:
        raise ValueError("the carry needs a text to read")
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    try:
        fast_weights.reset_carry()
        reset = compute.eval_token_losses(target)
        compute.eval_loss(prime)
        fast_weights.advance_carry()
        carry = compute.eval_token_losses(target)
    finally:
        fast_weights.reset_carry()
    return TokenView(reset=tuple(reset), carry=tuple(carry))
