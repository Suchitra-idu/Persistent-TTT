"""Ring 1 — the TTT mechanism over core.ttt_math. One implementation, so a
module rather than a registry (D5)."""

from ttt.extensions.mechanism.inplace_ttt import (
    InPlaceTTTMLP,
    RecomputeAsForward,
    StreamState,
    checkpoint_context_fn,
    iter_ttt_modules,
    pretrained_mlps,
)

__all__ = [
    "InPlaceTTTMLP",
    "RecomputeAsForward",
    "StreamState",
    "checkpoint_context_fn",
    "iter_ttt_modules",
    "pretrained_mlps",
]
