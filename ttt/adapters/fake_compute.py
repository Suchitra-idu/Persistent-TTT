"""FakeCompute — scripted losses. The train loop's whole test surface.

Losses cycle, and a scripted `nan` is how the nonfinite-loss guard gets tested
without a model that can diverge. `total_norm` does the same for the
nonfinite-*gradient* guard — a finite loss whose backward pass still didn't
produce a usable gradient.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ttt.core.types import GradStats
from ttt.ports.compute import GROUPS


class FakeCompute:
    def __init__(
        self,
        *,
        losses: Sequence[float] = (1.0,),
        grad_norms: Mapping[str, float] | None = None,
        total_norm: float = 1.0,
        fast_weights=None,
        parameter_counts: Mapping[str, int] | None = None,
    ) -> None:
        if not losses:
            raise ValueError("FakeCompute needs at least one scripted loss")
        self._losses = tuple(float(v) for v in losses)
        self._grad_norms = dict(grad_norms or {name: 1.0 for name in GROUPS})
        self._total_norm = total_norm
        self._fast_weights = fast_weights
        self._parameter_counts = dict(parameter_counts or {name: 1 for name in GROUPS})

        self.forwards: list[tuple[int, ...]] = []
        self.eval_forwards: list[tuple[int, ...]] = []
        self.eval_lora_flags: list[bool] = []
        self.backwards: list[float] = []
        self.steps: list[dict[str, float]] = []
        self.clipped_at: list[float] = []
        self.zero_grads = 0
        self._pending: float | None = None

    def loss(self, token_ids: Sequence[int]) -> float:
        ids = tuple(token_ids)
        self.forwards.append(ids)
        if self._fast_weights is not None:
            self._fast_weights.stage(len(ids))
        self._pending = self._losses[(len(self.forwards) - 1) % len(self._losses)]
        return self._pending

    def backward(self, *, scale: float) -> None:
        if self._pending is None:
            raise RuntimeError(
                "backward without a preceding loss; the graph it would consume "
                "does not exist"
            )
        self._pending = None
        self.backwards.append(scale)

    def zero_grad(self) -> None:
        self.zero_grads += 1

    def clip_and_step(
        self, *, max_grad_norm: float, learning_rates: Mapping[str, float]
    ) -> GradStats:
        self.clipped_at.append(max_grad_norm)
        if math.isfinite(self._total_norm):
            self.steps.append(dict(learning_rates))
        return GradStats(total_norm=self._total_norm, **self._grad_norms)

    def eval_loss(self, token_ids: Sequence[int], *, lora: bool = True) -> float:
        ids = tuple(token_ids)
        self.eval_forwards.append(ids)
        self.eval_lora_flags.append(lora)
        if self._fast_weights is not None:
            self._fast_weights.stage(len(ids))
        return self._losses[(len(self.eval_forwards) - 1) % len(self._losses)]

    def parameter_counts(self) -> Mapping[str, int]:
        return dict(self._parameter_counts)
