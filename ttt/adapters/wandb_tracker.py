"""WandbTracker — production telemetry. A wandb outage degrades to console.

Only wandb's own exceptions are swallowed; AttributeError and TypeError are
programmer bugs and must still surface.
"""

from __future__ import annotations

import sys
from typing import Mapping

from ttt.adapters.console_tracker import ConsoleTracker
from ttt.ports.tracker import MICRO_STEP, TRAIN_STEP

_AGGREGATE_NAMESPACES = ("train/*", "grad/*", "eval/*", "eval_seed/*")


class WandbTracker:
    def __init__(self, run, *, fallback: ConsoleTracker | None = None) -> None:
        self._run = run
        self._fallback = fallback or ConsoleTracker()
        self._warned = False
        self._define_axes()

    @classmethod
    def start(
        cls, *, project: str, run_name: str, job_type: str, config: Mapping
    ) -> "WandbTracker":
        import wandb

        run = wandb.init(
            project=project, name=run_name, job_type=job_type, config=dict(config)
        )
        return cls(run)

    def _define_axes(self) -> None:
        self._run.define_metric(TRAIN_STEP)
        self._run.define_metric(MICRO_STEP)
        self._run.define_metric("micro/*", step_metric=MICRO_STEP)
        for namespace in _AGGREGATE_NAMESPACES:
            self._run.define_metric(namespace, step_metric=TRAIN_STEP)

    def log(self, metrics: Mapping[str, float]) -> None:
        try:
            self._run.log(dict(metrics))
        except _wandb_errors() as exc:
            self._warn(exc)
            self._fallback.log(metrics)

    def finish(self) -> None:
        try:
            self._run.finish()
        except _wandb_errors() as exc:
            self._warn(exc)
            self._fallback.finish()

    def _warn(self, exc: BaseException) -> None:
        """Once, not every call: the fallback already runs from here on, and
        a warning per step would just be the silence it replaces, louder."""
        if self._warned:
            return
        self._warned = True
        print(
            f"wandb logging failed ({type(exc).__name__}: {exc}); falling back "
            "to console for the rest of this run",
            file=sys.stderr,
        )


def _wandb_errors() -> tuple[type[BaseException], ...]:
    try:
        import wandb

        return (wandb.Error,)
    except ImportError:
        return (RuntimeError,)
