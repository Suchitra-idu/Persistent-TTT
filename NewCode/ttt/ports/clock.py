"""Clock — wall time and monotonic time, injected so tests never sleep."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

STAMP_LENGTH = 9


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float:
        """Seconds, non-decreasing. Unrelated to wall time."""

    def stamp(self) -> str:
        """`MMDD-HHMM`, the run-name suffix. Exactly STAMP_LENGTH characters."""
