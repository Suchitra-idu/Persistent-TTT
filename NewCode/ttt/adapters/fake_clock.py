"""FakeClock — time moves only when a test says so."""

from __future__ import annotations


class FakeClock:
    def __init__(self, *, start: float = 0.0, stamp: str = "0101-0000") -> None:
        self._now = float(start)
        self._stamp = stamp

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError(f"a clock does not go backwards, got {seconds}")
        self._now += float(seconds)

    def monotonic(self) -> float:
        return self._now

    def stamp(self) -> str:
        return self._stamp
