"""Tracker — metric logging. Telemetry never crashes or stalls a run.

The two x-axes survive D12's cut from 25 keys to 13: aggregates step on
TRAIN_STEP, per-item signals on MICRO_STEP. `alert` does not survive.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

TRAIN_STEP = "train/step"
MICRO_STEP = "micro/step"

BUDGET = frozenset(
    {
        MICRO_STEP,
        "micro/doc_loss",
        "micro/state_ratio_mean",
        TRAIN_STEP,
        "train/loss",
        "train/lr_lora",
        "train/lr_wdown",
        "train/lr_new",
        "train/grad_clip_ratio",
        "grad/lora",
        "grad/wdown",
        "grad/new",
        "eval/carry_ppl",
        "eval/carry_off_ppl",
        "eval/lora_only_ppl",
        "eval/fresh_ppl",
        "eval/gap_lora",
        "eval/gap_within",
        "eval/gap_between",
        "eval/gap_total",
        "eval/state_ratio_final",
        "eval_seed/carry_ppl",
        "eval_seed/carry_off_ppl",
        "eval_seed/gap_between",
        "eval_seed/gap_seed",
        "eval_seed/n_seeded_docs",
    }
)


@runtime_checkable
class Tracker(Protocol):
    def log(self, metrics: Mapping[str, float]) -> None:
        """Must not raise: a telemetry outage is not a training failure."""

    def finish(self) -> None: ...


def outside_budget(metrics: Mapping[str, float]) -> tuple[str, ...]:
    """Keys D12 cut. Ring 4 asserts this is empty; adapters do not police it."""
    return tuple(sorted(key for key in metrics if key not in BUDGET))
