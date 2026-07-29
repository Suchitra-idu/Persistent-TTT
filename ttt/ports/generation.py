"""Generation — next-token logits and the KV cache. The token loop is Ring 4's."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class Generation(Protocol):
    def prefill(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Logits of the token after the prompt, shaped [1, vocab_size]."""

    def step(self, token_id: int) -> torch.Tensor:
        """Raises without a preceding `prefill` — there would be no cache."""

    def reset_cache(self) -> None:
        """Attention history only; the fast weight is FastWeights' to reset."""
