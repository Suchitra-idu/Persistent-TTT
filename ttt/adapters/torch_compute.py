"""TorchCompute — loss, backward and AdamW over the three parameter groups.

Group membership comes from `core.naming`, and an unclassified trainable is a
hard error: silently dropping one would train it at no learning rate at all.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ttt.core import naming
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig
from ttt.core.types import GradStats
from ttt.ports.compute import GROUPS


class TorchCompute:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        ttt_cfg: TTTConfig,
        train_cfg: TrainConfig,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.device = device
        self._groups = _group_parameters(model, ttt_cfg)
        self._optimizer = torch.optim.AdamW(
            [
                {
                    "params": self._groups[name],
                    "lr": lr,
                    "weight_decay": (
                        train_cfg.weight_decay_lora
                        if name == "lora"
                        else train_cfg.weight_decay_full
                    ),
                }
                for name, lr in (
                    ("lora", train_cfg.lr_lora),
                    ("wdown", train_cfg.lr_wdown),
                    ("new", train_cfg.lr_new_modules),
                )
            ]
        )
        self._pending: torch.Tensor | None = None

    def _ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)

    def loss(self, token_ids: Sequence[int]) -> float:
        ids = self._ids(token_ids)
        self._pending = self.model(input_ids=ids, labels=ids).loss
        return float(self._pending.detach())

    def backward(self, *, scale: float) -> None:
        if self._pending is None:
            raise RuntimeError(
                "backward without a preceding loss; the graph it would consume "
                "does not exist"
            )
        pending, self._pending = self._pending, None
        (pending * scale).backward()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad(set_to_none=True)

    def clip_and_step(
        self, *, max_grad_norm: float, learning_rates: Mapping[str, float]
    ) -> GradStats:
        norms = {name: _grad_norm(params) for name, params in self._groups.items()}
        total = torch.nn.utils.clip_grad_norm_(
            [p for params in self._groups.values() for p in params], max_grad_norm
        )
        for group, name in zip(self._optimizer.param_groups, GROUPS):
            group["lr"] = learning_rates[name]
        self._optimizer.step()
        return GradStats(total_norm=float(total), **norms)

    @torch.no_grad()
    def eval_loss(self, token_ids: Sequence[int]) -> float:
        ids = self._ids(token_ids)
        return float(self.model(input_ids=ids, labels=ids).loss)

    def parameter_counts(self) -> Mapping[str, int]:
        return {
            name: sum(p.numel() for p in params)
            for name, params in self._groups.items()
        }


def _group_parameters(
    model: torch.nn.Module, cfg: TTTConfig
) -> dict[str, list[torch.nn.Parameter]]:
    ttt_down = naming.ttt_down_suffixes(cfg.layer_indices or ())
    groups: dict[str, list[torch.nn.Parameter]] = {name: [] for name in GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = naming.classify_param(name, ttt_down)
        if group is None:
            raise RuntimeError(f"unclassified trainable parameter: {name}")
        groups[group].append(parameter)
    return groups


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(
        torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads))
    )
