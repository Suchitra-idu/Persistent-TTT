"""FastWeights — the carry lifecycle behind one port (D6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ttt.core.types import Carry

@runtime_checkable
class FastWeights(Protocol):
    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Base-model indices of the TTT layers, ascending."""

    def set_mode(self, *, evolve: bool, session: bool) -> None:
        """evolve freezes or thaws the fast weights; session keeps the carry."""

    def reset_carry(self) -> None: ...

    def advance_carry(self) -> None:
        """Promote the staged item delta. Once per item, after backward."""

    def detach_carry(self) -> None:
        """Cut the carry from the graph. Required after an optimizer step."""

    def snapshot(self, *, to_cpu: bool = False) -> Carry: ...

    def install(self, carry: Carry) -> None: ...

    def state_ratio(self) -> float:
        """Mean ||eta*S||_F / ||W0||_F over layers. Zero when nothing is staged."""

    def gate_stats(self) -> tuple[float, float] | None: ...
