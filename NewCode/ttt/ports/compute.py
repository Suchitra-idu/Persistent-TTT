"""Compute — loss, backward, and the optimizer step (D6).

Split from FastWeights so the whole train loop, carry lifecycle included, runs
with no model loaded. `loss` retains a graph that the next `backward` consumes;
a nonfinite loss is Ring 4's to act on, not this port's.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from ttt.core.types import GradStats

LORA = "lora"
WDOWN = "wdown"
NEW = "new"
GROUPS = (LORA, WDOWN, NEW)


@runtime_checkable
class Compute(Protocol):
    def loss(self, token_ids: Sequence[int]) -> float: ...

    def backward(self, *, scale: float) -> None: ...

    def zero_grad(self) -> None: ...

    def clip_and_step(
        self, *, max_grad_norm: float, learning_rates: Mapping[str, float]
    ) -> GradStats:
        """Norms are measured before clipping. `learning_rates` is keyed by
        GROUPS and applied first, so the schedule stays in Ring 0."""

    def eval_loss(self, token_ids: Sequence[int]) -> float:
        """No graph, and accumulated gradients left untouched."""

    def parameter_counts(self) -> Mapping[str, int]: ...
