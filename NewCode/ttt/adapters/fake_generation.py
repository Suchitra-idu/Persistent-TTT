"""FakeGeneration — a scripted token stream, returned as logits.

The script is what the app will sample at temperature 0; a peaked logits row
rather than a returned id keeps the sampling policy in Ring 4 where the A/B
test can vary it.
"""

from __future__ import annotations

from typing import Sequence

import torch

PEAK = 10.0


class FakeGeneration:
    def __init__(
        self,
        script: Sequence[int],
        *,
        vocab_size: int = 32,
        fast_weights=None,
    ) -> None:
        if not script:
            raise ValueError("FakeGeneration needs at least one scripted token")
        bad = [t for t in script if not 0 <= t < vocab_size]
        if bad:
            raise ValueError(f"scripted tokens {bad} outside vocab of {vocab_size}")
        self.script = tuple(int(t) for t in script)
        self.vocab_size = vocab_size
        self._fast_weights = fast_weights

        self.prompts: list[tuple[int, ...]] = []
        self.stepped: list[int] = []
        self.cache_resets = 0
        self._emitted = 0
        self._primed = False

    def prefill(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not len(token_ids):
            raise ValueError("cannot prefill on an empty prompt")
        self.prompts.append(tuple(token_ids))
        self._primed = True
        self._observe(len(token_ids))
        return self._logits()

    def step(self, token_id: int) -> torch.Tensor:
        if not self._primed:
            raise RuntimeError("step before prefill; there is no cache to extend")
        self.stepped.append(int(token_id))
        self._observe(1)
        return self._logits()

    def reset_cache(self) -> None:
        self._primed = False
        self.cache_resets += 1

    def _observe(self, n_tokens: int) -> None:
        if self._fast_weights is not None:
            self._fast_weights.stage(n_tokens)

    def _logits(self) -> torch.Tensor:
        token = self.script[self._emitted % len(self.script)]
        self._emitted += 1
        row = torch.zeros(1, self.vocab_size)
        row[0, token] = PEAK
        return row
