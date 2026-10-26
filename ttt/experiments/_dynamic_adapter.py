"""Dynamic evaluation on a loaded model's LoRA weights, for `dynamic_eval`.

Updates are made on fp32 master copies and written back in the model's dtype:
in bf16 a step smaller than about 0.4 percent of a weight would round away.
"""

from __future__ import annotations

from typing import Sequence

import torch

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


class LoraAdapter:
    def __init__(self, model: torch.nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.params = [p for name, p in model.named_parameters() if "lora_" in name]
        if not self.params:
            raise ValueError("model has no LoRA parameters to adapt")
        self._original = [p.detach().clone() for p in self.params]

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.params)

    def adapt(self, token_ids: Sequence[int], *, optimizer: str, lr: float, steps: int) -> None:
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
        ids = torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)
        master = [p.detach().float().clone() for p in self.params]
        moments = [(torch.zeros_like(m), torch.zeros_like(m)) for m in master]
        was_training = self.model.training
        self.model.eval()
        for p in self.params:
            p.requires_grad_(True)
        try:
            for step in range(1, steps + 1):
                with torch.enable_grad():
                    loss = self.model(input_ids=ids, labels=ids).loss
                    grads = torch.autograd.grad(loss, self.params)
                with torch.no_grad():
                    for m, g, (m1, m2), p in zip(master, grads, moments, self.params):
                        g = g.float()
                        if optimizer == "sgd":
                            m.sub_(lr * g)
                        else:
                            m1.mul_(ADAM_BETAS[0]).add_(g, alpha=1 - ADAM_BETAS[0])
                            m2.mul_(ADAM_BETAS[1]).addcmul_(g, g, value=1 - ADAM_BETAS[1])
                            m_hat = m1 / (1 - ADAM_BETAS[0] ** step)
                            v_hat = m2 / (1 - ADAM_BETAS[1] ** step)
                            m.sub_(lr * m_hat / (v_hat.sqrt() + ADAM_EPS))
                        p.copy_(m.to(p.dtype))
        finally:
            for p in self.params:
                p.requires_grad_(False)
            self.model.train(was_training)

    def restore(self) -> None:
        with torch.no_grad():
            for p, original in zip(self.params, self._original):
                p.copy_(original)
