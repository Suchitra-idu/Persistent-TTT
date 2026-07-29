"""SystemClock — the real one. The only place the process reads a clock."""

from __future__ import annotations

import time


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def stamp(self) -> str:
        return time.strftime("%m%d-%H%M")
