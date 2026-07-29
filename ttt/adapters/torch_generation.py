"""TorchGeneration — prefill, then one token at a time, over a KV cache.

`reset_cache` drops attention history only. The fast weight is the other
port's; that separation is what makes D14's `cross_turn` switch meaningful.
"""

from __future__ import annotations

from typing import Sequence

import torch


class TorchGeneration:
    def __init__(self, model: torch.nn.Module, *, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self._cache = None

    @torch.no_grad()
    def prefill(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not len(token_ids):
            raise ValueError("cannot prefill on an empty prompt")
        return self._forward(list(token_ids), cache=None)

    @torch.no_grad()
    def step(self, token_id: int) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("step before prefill; there is no cache to extend")
        return self._forward([int(token_id)], cache=self._cache)

    def reset_cache(self) -> None:
        self._cache = None

    def _forward(self, token_ids: list[int], cache) -> torch.Tensor:
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        self._cache = out.past_key_values
        return out.logits[:, -1, :]
