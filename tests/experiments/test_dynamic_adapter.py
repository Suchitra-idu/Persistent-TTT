from __future__ import annotations

import pytest
import torch

from ttt.experiments._dynamic_adapter import LoraAdapter

IDS = (1, 2, 3, 4, 5, 6)


class TinyLm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(8, 4)
        self.lora_A = torch.nn.Parameter(torch.randn(4, 4, generator=gen) * 0.1)
        self.head = torch.nn.Linear(4, 8)
        for p in (*self.embed.parameters(), *self.head.parameters()):
            p.requires_grad_(False)
        self.lora_A.requires_grad_(False)

    def forward(self, input_ids, labels):
        logits = self.head(self.embed(input_ids) @ (torch.eye(4) + self.lora_A))
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:])
        return type("Out", (), {"loss": loss})()


def adapter():
    return LoraAdapter(TinyLm(), device="cpu")


def loss_of(a):
    with torch.no_grad():
        ids = torch.tensor([IDS])
        return float(a.model(input_ids=ids, labels=ids).loss)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_adapting_lowers_the_loss_on_the_adapted_text(optimizer):
    a = adapter()
    before = loss_of(a)

    a.adapt(IDS, optimizer=optimizer, lr=0.05, steps=3)

    assert loss_of(a) < before


def test_restore_returns_the_exact_original_weights():
    a = adapter()
    original = a.params[0].detach().clone()
    a.adapt(IDS, optimizer="adam", lr=0.05, steps=3)

    a.restore()

    assert torch.equal(a.params[0], original)


def test_only_lora_parameters_move():
    a = adapter()
    head = a.model.head.weight.detach().clone()

    a.adapt(IDS, optimizer="sgd", lr=0.5, steps=1)

    assert torch.equal(a.model.head.weight, head)


def test_adapting_leaves_the_weights_frozen_and_the_mode_unchanged():
    a = adapter()
    a.model.train()

    a.adapt(IDS, optimizer="sgd", lr=0.1, steps=1)

    assert not a.params[0].requires_grad and a.model.training


def test_an_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer must be"):
        adapter().adapt(IDS, optimizer="rmsprop", lr=0.1, steps=1)


def test_a_model_without_lora_is_rejected():
    with pytest.raises(ValueError, match="no LoRA parameters"):
        LoraAdapter(torch.nn.Linear(2, 2), device="cpu")
