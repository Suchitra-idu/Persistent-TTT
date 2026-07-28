"""The session carry: how an item's delta joins it, and how big it gets."""

from __future__ import annotations

from typing import Mapping

import torch


def advance(
    carried: torch.Tensor | None, item_delta: torch.Tensor, *, decay: float
) -> torch.Tensor:
    """S <- decay * S + delta, starting at delta on the first item.

    decay=1.0 is a pure sum (unbounded); decay=0.0 keeps only the last item.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"carried_decay must be in [0, 1], got {decay}")
    if carried is None:
        return item_delta
    return decay * carried + item_delta


def steady_state_scale(decay: float) -> float:
    """1/(1-decay): what a constant per-item delta converges to under `advance`."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(
            f"a steady state exists only for decay in [0, 1), got {decay}"
        )
    return 1.0 / (1.0 - decay)


def state_ratio(state: torch.Tensor | None, w0_norm: float, *, eta: float) -> float:
    """||eta*S||_F / ||W0||_F for one layer. No state means zero."""
    if w0_norm <= 0.0:
        raise ValueError(f"||W0||_F must be positive, got {w0_norm}")
    if state is None:
        return 0.0
    return float((eta * state).norm(p="fro")) / w0_norm


def mean_state_ratio(ratios: Mapping[int, float]) -> float:
    if not ratios:
        return 0.0
    return sum(ratios.values()) / len(ratios)
