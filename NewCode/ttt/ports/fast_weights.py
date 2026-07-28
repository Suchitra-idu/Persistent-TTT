"""FastWeights — the carry lifecycle, both state families behind one port (D6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ttt.core.types import Carry

CARRY = "carry"
STREAM = "stream"
FAMILIES = (CARRY, STREAM)


@runtime_checkable
class FastWeights(Protocol):
    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Base-model indices of the TTT layers, ascending."""

    def set_mode(self, *, evolve: bool, stream: bool, session: bool) -> None:
        """evolve freezes or thaws; stream selects the generation path."""

    def reset_stream(self) -> None: ...

    def reset_v_context(self) -> None:
        """The soft turn boundary: conv left-context only, delta survives."""

    def reset_carry(self) -> None: ...

    def advance_carry(self) -> None:
        """Promote the staged item delta. Once per item, after backward."""

    def snapshot(self, *, to_cpu: bool = False) -> Carry: ...

    def install(self, carry: Carry) -> None: ...

    def state_ratio(self, *, family: str) -> float:
        """Mean ||eta*S||_F / ||W0||_F over layers. Zero when nothing is staged."""

    def stream_progress(self) -> tuple[int, int]:
        """(pending tokens, chunk size) — disambiguates a zero state_ratio."""

    def gate_stats(self) -> tuple[float, float] | None: ...
