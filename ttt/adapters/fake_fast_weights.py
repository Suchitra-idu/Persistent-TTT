"""FakeFastWeights — the carry lifecycle as scalars, so Ring 4 needs no model.

A 1x1 delta per layer, advanced by the same EMA as the real thing. `stage` is
what a forward does on the real adapter; FakeCompute calls it at the same point.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ttt.core import carry as carry_math
from ttt.core.types import Carry


class FakeFastWeights:
    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        chunk_size: int = 4,
        decay: float = 0.9,
        eta: float = 1.0,
        delta_per_token: float = 1.0,
    ) -> None:
        self._layer_indices = tuple(sorted(int(i) for i in layer_indices))
        self.chunk_size = chunk_size
        self.decay = decay
        self.eta = eta
        self.delta_per_token = delta_per_token

        self.evolve = True
        self.session_mode = False
        self.events: list[str] = []

        self._carried: dict[int, torch.Tensor] = {}
        self._staged: dict[int, torch.Tensor] | None = None

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self._layer_indices

    def stage(self, n_tokens: int) -> None:
        """One forward over `n_tokens`."""
        if not self.evolve:
            return
        if self.session_mode:
            delta = torch.tensor([[self.delta_per_token * n_tokens]])
            self._staged = {i: delta.clone() for i in self._layer_indices}

    def set_mode(self, *, evolve: bool, session: bool) -> None:
        self.evolve, self.session_mode = evolve, session
        self.events.append(f"mode(evolve={evolve},session={session})")

    def reset_carry(self) -> None:
        self._carried = {}
        self._staged = None
        self.events.append("reset_carry")

    def advance_carry(self) -> None:
        self.events.append("advance_carry")
        if self._staged is None:
            return
        for index, delta in self._staged.items():
            self._carried[index] = carry_math.advance(
                self._carried.get(index), delta, decay=self.decay
            )
        self._staged = None

    def detach_carry(self) -> None:
        self._carried = {i: t.detach() for i, t in self._carried.items()}
        self.events.append("detach_carry")

    def snapshot(self, *, to_cpu: bool = False) -> Carry:
        return Carry(deltas={i: t.clone() for i, t in self._carried.items()})

    def install(self, carry: Carry) -> None:
        unknown = sorted(set(carry.deltas) - set(self._layer_indices))
        if unknown:
            raise KeyError(
                f"carry has layer indices {unknown} this model does not have; "
                f"it has {list(self._layer_indices)}"
            )
        self._carried = {i: t.clone() for i, t in carry.deltas.items()}
        self.events.append("install")

    def state_ratio(self) -> float:
        ratios = {
            i: carry_math.state_ratio(self._carried.get(i), 1.0, eta=self.eta)
            for i in self._layer_indices
        }
        return carry_math.mean_state_ratio(ratios)

    def gate_mean(self) -> float | None:
        return 0.5
