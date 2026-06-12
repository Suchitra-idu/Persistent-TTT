"""
Shared fixtures. Tiny dimensions, CPU, fp64, no Modal, no GPU, no model
downloads. The whole suite runs in seconds.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from inplace_ttt import EmbeddingTap, InPlaceTTTMLP
from ttt_config import TTTConfig

torch.set_default_dtype(torch.float64)

D, DFF, C, KERNEL = 8, 16, 4, 3


@pytest.fixture
def cfg():
    return TTTConfig(layer_indices=(0,), chunk_size=C, eta=0.05,
                     conv_kernel_size=KERNEL,
                     normalize_delta_by_chunk=True)


@pytest.fixture
def module_factory(cfg):
    """Returns (ttt_module, plain_mlp, tap). randomize=True makes the
    targets nonzero so updates actually happen; False keeps the exact
    zero-init that guarantees identity with the base MLP."""

    def make(randomize: bool = True, seed: int = 0, config: TTTConfig = None):
        torch.manual_seed(seed)
        mlp = types.SimpleNamespace(
            gate_proj=torch.nn.Linear(D, DFF, bias=False),
            up_proj=torch.nn.Linear(D, DFF, bias=False),
            down_proj=torch.nn.Linear(DFF, D, bias=False),
            act_fn=torch.nn.SiLU(),
        )
        c = config or cfg
        tap = EmbeddingTap(c.conv_kernel_size)
        m = InPlaceTTTMLP(mlp, D, c, tap)
        if randomize:
            torch.nn.init.normal_(m.w_target, std=0.5)
            torch.nn.init.normal_(m.target_conv.weight, std=0.5)
        m.eval()
        return m, mlp, tap

    return make


def scan(m, tap, x0):
    """Run one sequence through the stateless scan path."""
    tap.current = x0.unsqueeze(0)
    with torch.no_grad():
        return m(x0.unsqueeze(0))[0]
