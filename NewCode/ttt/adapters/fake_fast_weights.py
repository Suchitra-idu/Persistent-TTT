"""FakeFastWeights — the carry lifecycle as scalars, so Ring 4 needs no model.

A 1x1 delta per layer, advanced by the same EMA as the real thing. `stage` is
what a forward does on the real adapter; FakeCompute and FakeGeneration call it
for the same reason and at the same point.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ttt.core import carry as carry_math
from ttt.core.types import Carry
from ttt.ports.fast_weights import CARRY, FAMILIES, STREAM


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
        self.stream_mode = False
        self.session_mode = False
        self.events: list[str] = []

        self._carried: dict[int, torch.Tensor] = {}
        self._staged: dict[int, torch.Tensor] | None = None
        self._stream: dict[int, torch.Tensor] = {}
        self._pending = 0

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self._layer_indices

    def stage(self, n_tokens: int) -> None:
        """One forward over `n_tokens`, on whichever family the mode selects."""
        if not self.evolve:
            return
        if self.stream_mode:
            self._stream_tokens(n_tokens)
            return
        if self.session_mode:
            delta = torch.tensor([[self.delta_per_token * n_tokens]])
            self._staged = {i: delta.clone() for i in self._layer_indices}

    def _stream_tokens(self, n_tokens: int) -> None:
        self._pending += n_tokens
        commits, self._pending = divmod(self._pending, self.chunk_size)
        if not commits:
            return
        amount = self.delta_per_token * self.chunk_size * commits
        for index in self._layer_indices:
            current = self._stream.get(index, torch.zeros(1, 1))
            self._stream[index] = current + amount

    def set_mode(self, *, evolve: bool, stream: bool, session: bool) -> None:
        self.evolve, self.stream_mode, self.session_mode = evolve, stream, session
        self.events.append(f"mode(evolve={evolve},stream={stream},session={session})")

    def reset_stream(self) -> None:
        self._stream = {}
        self._pending = 0
        self.events.append("reset_stream")

    def reset_v_context(self) -> None:
        self.events.append("reset_v_context")

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

    def state_ratio(self, *, family: str) -> float:
        if family not in FAMILIES:
            raise ValueError(
                f"unknown state family {family!r}; expected {CARRY!r} or {STREAM!r}"
            )
        states = self._carried if family == CARRY else self._stream
        ratios = {
            i: carry_math.state_ratio(states.get(i), 1.0, eta=self.eta)
            for i in self._layer_indices
        }
        return carry_math.mean_state_ratio(ratios)

    def stream_progress(self) -> tuple[int, int]:
        return self._pending, self.chunk_size

    def gate_stats(self) -> tuple[float, float] | None:
        return (0.5, 0.0)
