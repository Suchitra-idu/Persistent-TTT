"""Builders for the Rings 2/3 suites. Deterministic; nothing reads global RNG."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
import torch.nn as nn

from ttt.adapters.model_builder import patch_model_with_ttt, unfreeze_ttt_params
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.ttt import TTTConfig

HIDDEN = 4
D_FF = 6
LAYERS = 4
VOCAB = 16
LAYER_INDICES = (1, 3)

TTT_TENSOR_SUFFIXES = (
    "v_source_norm.weight",
    "target_conv.weight",
    "w_target",
    "output_gate.weight",
    "output_gate.bias",
    "down_proj.weight",
)


def generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def fill(module: nn.Module, *, seed: int, scale: float = 0.1) -> None:
    with torch.no_grad():
        for offset, parameter in enumerate(module.parameters()):
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator(seed + offset)) * scale
            )


@dataclass
class TinyConfig:
    hidden_size: int = HIDDEN
    num_hidden_layers: int = LAYERS
    vocab_size: int = VOCAB


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.up_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int) -> None:
        super().__init__()
        self.mlp = TinyMLP(hidden_size, d_ff)


class TinyBody(nn.Module):
    def __init__(self, config: TinyConfig, d_ff: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TinyLayer(config.hidden_size, d_ff) for _ in range(config.num_hidden_layers)
        )


@dataclass
class TinyOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: int | None = None


class TinyCausalLM(nn.Module):
    """A causal LM with no attention: position t's logits depend on token t alone.

    That is enough to exercise every adapter — a real loss, real gradients, a
    cache whose only content is how many tokens it has seen — and it keeps the
    Ring 3 suites at CPU-millisecond speed with no transformers install.
    """

    def __init__(self, *, d_ff: int = D_FF, config: TinyConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyConfig()
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.model = TinyBody(self.config, d_ff)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: int | None = None,
        use_cache: bool = False,
    ) -> TinyOutput:
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = hidden + layer.mlp(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        cache = (past_key_values or 0) + input_ids.shape[1] if use_cache else None
        return TinyOutput(logits=logits, loss=loss, past_key_values=cache)


def tiny_model(
    *,
    seed: int = 0,
    layer_indices: Sequence[int] = LAYER_INDICES,
    **cfg_overrides: Any,
) -> tuple[TinyCausalLM, TTTConfig]:
    """Patched, frozen, then TTT-unfrozen — the same order build_model uses."""
    model = TinyCausalLM()
    fill(model, seed=seed)
    cfg = TTTConfig(layer_indices=tuple(layer_indices), **cfg_overrides)
    patch_model_with_ttt(model, cfg)
    # w_target is zero-init by design, which would make every delta zero and
    # every state-ratio assertion below vacuous.
    with torch.no_grad():
        for index in cfg.layer_indices:
            target = model.model.layers[index].mlp.w_target
            target.copy_(
                torch.randn(target.shape, generator=generator(seed + 700 + index)) * 0.3
            )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    unfreeze_ttt_params(model, cfg)
    return model, cfg


def token_ids(n: int, *, seed: int = 3, vocab: int = VOCAB) -> list[int]:
    drawn = torch.randint(0, vocab, (n,), generator=generator(seed))
    return [int(t) for t in drawn]


FIXTURE_SPEC = DatasetSpec(
    name="ports-fixture",
    source="_fixture",
    source_meta_column="meta",
    source_meta_key="set_name",
    include_sources=("alpha", "beta"),
)

UNLABELLED_SPEC = DatasetSpec(
    name="ports-unlabelled",
    source="_fixture",
    source_meta_column="meta",
    source_meta_key="set_name",
)

CONSTANT_SPEC = DatasetSpec(
    name="ports-constant",
    source="_fixture",
    constant_source="only",
)


def spec_rows(sources: Sequence[str], *, spec: DatasetSpec = FIXTURE_SPEC) -> list[dict]:
    return [
        {spec.text_column: f"doc {i}", spec.source_meta_column: {spec.source_meta_key: s}}
        for i, s in enumerate(sources)
    ]


def saved_to_disk(directory, spec: DatasetSpec, rows: Sequence[dict]) -> DatasetSpec:
    import datasets

    datasets.Dataset.from_list(list(rows)).save_to_disk(str(directory))
    return replace(spec, source=str(directory))


def table_rows(n: int = 5) -> list[dict]:
    return [{"text": f"doc {i}", "source": "alpha", "n": i} for i in range(n)]
