"""TorchFastWeights — the FastWeights port over the patched InPlaceTTTMLPs.

Keys come from the module path, so a `Carry` is base-model-layer-indexed at
the only point where the two numbering schemes could ever diverge (D14 defect 2).
"""

from __future__ import annotations

import re

import torch

from ttt.core import carry as carry_math
from ttt.core.config.ttt import TTTConfig
from ttt.core.types import Carry
from ttt.extensions.mechanism import InPlaceTTTMLP, iter_ttt_modules
from ttt.ports.fast_weights import CARRY, FAMILIES, STREAM

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


class TorchFastWeights:
    def __init__(self, model: torch.nn.Module, cfg: TTTConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._modules = _by_layer_index(model)
        if not self._modules:
            raise ValueError(
                "model has no InPlaceTTTMLP layers; patch it before building "
                "the FastWeights adapter"
            )

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._modules))

    def set_mode(self, *, evolve: bool, stream: bool, session: bool) -> None:
        for module in self._modules.values():
            module.evolve = evolve
            module.stateful = stream
            module.session_mode = session

    def reset_stream(self) -> None:
        for module in self._modules.values():
            module.reset_stream()

    def reset_v_context(self) -> None:
        for module in self._modules.values():
            module.reset_v_context()

    def reset_carry(self) -> None:
        for module in self._modules.values():
            module.reset_carry()

    def advance_carry(self) -> None:
        for module in self._modules.values():
            module.advance_carry()

    def snapshot(self, *, to_cpu: bool = False) -> Carry:
        deltas = {}
        for index, module in self._modules.items():
            if module.carried is None:
                continue
            staged = module.carried.detach().float()
            deltas[index] = (staged.cpu() if to_cpu else staged).clone()
        return Carry(deltas=deltas)

    def install(self, carry: Carry) -> None:
        unknown = sorted(set(carry.deltas) - set(self._modules))
        if unknown:
            raise KeyError(
                f"carry has layer indices {unknown} this model does not have; "
                f"it has {list(self.layer_indices)}"
            )
        for index, delta in carry.deltas.items():
            module = self._modules[index]
            target = module.down_proj.weight
            if tuple(delta.shape[-2:]) != tuple(target.shape):
                raise ValueError(
                    f"carry for layer {index} has shape {tuple(delta.shape)}, "
                    f"incompatible with W0 {tuple(target.shape)}"
                )
            module.carried = delta.to(device=target.device).float()

    def state_ratio(self, *, family: str) -> float:
        _check_family(family)
        ratios = {
            index: carry_math.state_ratio(
                _state_of(module, family),
                float(module.down_proj.weight.detach().float().norm(p="fro")),
                eta=self.cfg.eta,
            )
            for index, module in self._modules.items()
        }
        return carry_math.mean_state_ratio(ratios)

    def stream_progress(self) -> tuple[int, int]:
        first = self._modules[self.layer_indices[0]]
        return first.stream.pending_tokens, self.cfg.chunk_size

    def gate_stats(self) -> tuple[float, float] | None:
        means = [
            (m.gate_mean, m.gate_std)
            for m in self._modules.values()
            if m.gate_mean is not None
        ]
        if not means:
            return None
        return (
            sum(mean for mean, _ in means) / len(means),
            sum(std for _, std in means) / len(means),
        )


def _by_layer_index(model: torch.nn.Module) -> dict[int, InPlaceTTTMLP]:
    found: dict[int, InPlaceTTTMLP] = {}
    patched = set(id(m) for m in iter_ttt_modules(model))
    for name, module in model.named_modules():
        if id(module) not in patched:
            continue
        matches = _LAYER_RE.findall(name)
        if not matches:
            raise ValueError(
                f"TTT module at {name!r} has no `layers.<i>.` in its path, so "
                "its base-model layer index cannot be recovered"
            )
        found[int(matches[-1])] = module
    return found


def _state_of(module: InPlaceTTTMLP, family: str) -> torch.Tensor | None:
    return module.carried if family == CARRY else module.stream.delta


def _check_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(
            f"unknown state family {family!r}; expected {CARRY!r} or {STREAM!r}"
        )
