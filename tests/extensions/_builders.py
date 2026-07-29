"""Data builders for the Ring 1 suites. Deterministic per explicit seed."""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.ttt import TTTConfig
from ttt.extensions.mechanism import InPlaceTTTMLP

FP64 = torch.float64


class FakeRng:
    """The Rng port's shape, backed by a private `random.Random`."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def integers(self, low: int, high: int) -> int:
        """Half-open [low, high), like numpy's."""
        return self._rng.randrange(low, high)

    def permutation(self, n: int) -> list[int]:
        order = list(range(n))
        self._rng.shuffle(order)
        return order


def doc_lengths(n: int, *, seed: int, low: int = 1, high: int = 12_000) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(low, high) for _ in range(n)]


def sources_for(lengths, *, seed: int, labels=("a", "b", "c")) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(labels) for _ in lengths]


def labelled_row(spec: DatasetSpec, source: str) -> dict[str, Any]:
    if spec.constant_source is not None:
        return {spec.text_column: "x"}
    return {
        spec.text_column: "x",
        spec.source_meta_column: {spec.source_meta_key: source},
    }


def a_source_of(spec: DatasetSpec) -> str:
    return spec.constant_source or spec.include_sources[0]


def generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


class FakeMLP(nn.Module):
    """A Qwen3 gated MLP's shape, with weights that owe nothing to global RNG."""

    def __init__(self, hidden_size: int, d_ff: int, *, seed: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.up_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, hidden_size, bias=False)
        self.act_fn = nn.functional.silu
        with torch.no_grad():
            for offset, layer in enumerate(
                (self.gate_proj, self.up_proj, self.down_proj)
            ):
                layer.weight.copy_(
                    torch.randn(
                        layer.weight.shape, generator=generator(seed + offset)
                    )
                    * 0.1
                )


def ttt_module(
    *,
    hidden_size: int = 6,
    d_ff: int = 8,
    seed: int = 0,
    w_target_scale: float = 0.0,
    conv_scale: float = 0.0,
    dtype: torch.dtype = torch.float32,
    **cfg_overrides,
) -> InPlaceTTTMLP:
    """conv_scale > 0 gives the depthwise conv a real kernel. At init it is a
    single identity tap, which makes every left-context bug invisible."""
    cfg = TTTConfig(**cfg_overrides)
    module = InPlaceTTTMLP(FakeMLP(hidden_size, d_ff, seed=seed), hidden_size, cfg)
    with torch.no_grad():
        module.w_target.copy_(
            torch.randn(
                (hidden_size, hidden_size), generator=generator(seed + 500)
            )
            * w_target_scale
        )
        module.target_conv.weight.add_(
            torch.randn(
                module.target_conv.weight.shape, generator=generator(seed + 900)
            )
            * conv_scale
        )
    return module.to(dtype=dtype)


def hidden_states(
    n_tokens: int,
    *,
    hidden_size: int = 6,
    batch: int = 1,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(
        batch, n_tokens, hidden_size, generator=generator(seed), dtype=dtype
    )
