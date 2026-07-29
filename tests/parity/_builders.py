"""One old module and one new module holding bit-identical fp64 weights.

Every parameter is filled from a seeded generator before the copy. `w_target` is
zero-init on both sides, and a comparison against an untouched one would agree
perfectly while proving nothing — the TTT term would be exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import inplace_ttt as old_mechanism
import torch
import ttt_config
import ttt_wiring

from ttt.core.config.ttt import TTTConfig
from ttt.extensions.mechanism import InPlaceTTTMLP

HIDDEN = 6
D_FF = 8
LAYERS = 4
VOCAB = 16
LAYER_INDICES = (1, 3)

# Shared by both TTTConfigs. v_source and gate_reg_weight are old-only (D11).
COMMON = dict(
    chunk_size=4,
    eta=7e-2,
    normalize_delta_by_chunk=True,
    conv_kernel_size=3,
    v_bidirectional=False,
    output_gate=True,
    output_gate_bias_init=-2.0,
    clip_enabled=True,
    clip_tau=5.0,
    clip_at_inference_only=False,
    carried_decay=0.9,
)


def generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def fill(module: torch.nn.Module, *, seed: int, scale: float = 0.5) -> None:
    """Every parameter, including the ones both sides zero-init."""
    with torch.no_grad():
        for offset, parameter in enumerate(module.parameters()):
            parameter.copy_(
                torch.randn(
                    parameter.shape, generator=generator(seed + offset), dtype=torch.float64
                )
                * scale
            )


class Mlp(torch.nn.Module):
    def __init__(self, hidden: int = HIDDEN, d_ff: int = D_FF) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, d_ff, bias=False)
        self.up_proj = torch.nn.Linear(hidden, d_ff, bias=False)
        self.down_proj = torch.nn.Linear(d_ff, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu


# Every field both TTTConfigs still have. v_source and gate_reg_weight are
# old-only (D11); layer_indices is passed separately.
SHARED_FIELDS = tuple(COMMON)


def as_old(new_cfg: TTTConfig) -> Any:
    """The old config that means the same thing. Derived rather than written out
    twice, so the two sides cannot drift apart in a fixture."""
    return ttt_config.TTTConfig(
        layer_indices=new_cfg.layer_indices,
        v_source="hidden_state",
        **{field: getattr(new_cfg, field) for field in SHARED_FIELDS},
    )


def configs(**overrides) -> tuple[Any, TTTConfig]:
    new = TTTConfig(layer_indices=LAYER_INDICES, **{**COMMON, **overrides})
    return as_old(new), new


@dataclass
class Pair:
    old: Any
    new: InPlaceTTTMLP
    old_cfg: Any
    new_cfg: TTTConfig

    def both(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.old(hidden_states), self.new(hidden_states)

    def set_mode(self, *, evolve: bool, stream: bool, session: bool) -> None:
        self.old.ttt_evolve, self.old.stateful = evolve, stream
        self.old.session_mode = session
        self.new.evolve, self.new.stateful = evolve, stream
        self.new.session_mode = session

    def advance_carry(self) -> None:
        if self.old._next_carried is not None:
            self.old.carried_delta = self.old._next_carried
            self.old._next_carried = None
        self.new.advance_carry()

    def reset_carry(self) -> None:
        self.old.carried_delta = self.old._next_carried = None
        self.new.reset_carry()

    def reset_stream(self) -> None:
        self.old.reset_stream_state()
        self.new.reset_stream()

    def carries(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.old.carried_delta, self.new.carried

    def stream_states(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.old.state.delta, self.new.stream.delta


def pair(*, seed: int = 0, **overrides) -> Pair:
    """Both modules in fp64, weights copied old -> new after a seeded fill."""
    old_cfg, new_cfg = configs(**overrides)
    tap = old_mechanism.EmbeddingTap(old_cfg.conv_kernel_size)

    old = old_mechanism.InPlaceTTTMLP(Mlp(), HIDDEN, old_cfg, tap).double()
    new = InPlaceTTTMLP(Mlp(), HIDDEN, new_cfg).double()
    fill(old, seed=seed)
    new.load_state_dict(old.state_dict())

    old.eval()
    new.eval()
    return Pair(old=old, new=new, old_cfg=old_cfg, new_cfg=new_cfg)


def hidden(n_tokens: int, *, seed: int = 11, batch: int = 1) -> torch.Tensor:
    return torch.randn(
        batch, n_tokens, HIDDEN, generator=generator(seed), dtype=torch.float64
    )


def old_model(**overrides):
    """A TinyCausalLM patched by the *old* patcher, for checkpoint compatibility.

    Built from the TTTConfig *defaults* so it matches `ports.tiny_model()` —
    a checkpoint only crosses between trees when the shapes agree.
    """
    from tests.ports import _builders as ports

    old_cfg = as_old(TTTConfig(layer_indices=ports.LAYER_INDICES, **overrides))
    model = ports.TinyCausalLM()
    ports.fill(model, seed=0)
    old_mechanism.patch_model_with_ttt(model, old_cfg)
    for index in old_cfg.layer_indices:
        ports.fill(model.model.layers[index].mlp, seed=100 + index)
    ttt_wiring.unfreeze_ttt_params(model, old_cfg)
    return model, old_cfg
