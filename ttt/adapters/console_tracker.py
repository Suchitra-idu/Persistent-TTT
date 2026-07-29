"""ConsoleTracker — the degraded path when wandb is unavailable."""

from __future__ import annotations

from typing import Mapping

from ttt.ports.tracker import MICRO_STEP, TRAIN_STEP


class ConsoleTracker:
    def __init__(self, *, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.every = every
        self._seen = 0

    def log(self, metrics: Mapping[str, float]) -> None:
        self._seen += 1
        if self._seen % self.every:
            return
        print(_line(metrics))

    def finish(self) -> None:
        print("tracker finished")


def _line(metrics: Mapping[str, float]) -> str:
    axis = next((k for k in (TRAIN_STEP, MICRO_STEP) if k in metrics), None)
    head = "" if axis is None else f"[{axis} {int(metrics[axis])}] "
    body = " ".join(
        f"{key}={_format(value)}"
        for key, value in sorted(metrics.items())
        if key != axis
    )
    return head + body


def _format(value: float) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)
