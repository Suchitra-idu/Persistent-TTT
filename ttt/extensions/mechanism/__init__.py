"""Ring 1 — the TTT mechanism over core.ttt_math. One implementation, so a
module rather than a registry (D5)."""

from ttt.extensions.mechanism.inplace_ttt import (
    InPlaceTTTMLP,
    StreamState,
    iter_ttt_modules,
)

__all__ = ["InPlaceTTTMLP", "StreamState", "iter_ttt_modules"]
