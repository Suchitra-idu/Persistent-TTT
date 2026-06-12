"""
Wiring tests. These guard the boundaries between our code and
PEFT/transformers, where the failure mode is silent (LoRA landing on a
fast weight, a parameter training that shouldn't, a checkpoint loading
into the wrong tensors) rather than an exception.
"""

import re

import pytest
import torch
import torch.nn as nn

from inplace_ttt import (
    build_lora_target_regex, build_param_groups, load_ttt_state_dict,
    save_ttt_state_dict, strip_peft_prefix, ttt_down_suffixes,
)
from ttt_config import TTTConfig

CFG = TTTConfig(layer_indices=(5, 11))
NUM_LAYERS = 12


def fullmatch(name):
    return re.fullmatch(build_lora_target_regex(NUM_LAYERS, CFG), name)


# ------------------------------------------------------------- LoRA regex --
def test_regex_targets_attention_and_gate_up_everywhere():
    for layer in (0, 5, 11):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert fullmatch(f"model.layers.{layer}.self_attn.{proj}")
        for proj in ("gate_proj", "up_proj"):
            assert fullmatch(f"model.layers.{layer}.mlp.{proj}")


def test_regex_excludes_down_proj_on_ttt_layers():
    """THE critical property. LoRA on a TTT layer's down_proj would
    silently decouple the fast weight from part of the slow weight."""
    assert fullmatch("model.layers.5.mlp.down_proj") is None
    assert fullmatch("model.layers.11.mlp.down_proj") is None


def test_regex_includes_down_proj_on_non_ttt_layers():
    for layer in (0, 4, 6, 10):
        assert fullmatch(f"model.layers.{layer}.mlp.down_proj")


def test_regex_layer_index_no_prefix_collision():
    """Layer 1 must not be excluded just because layer 11 is a TTT
    layer (regex alternation '1|11' style bugs)."""
    cfg = TTTConfig(layer_indices=(11,))
    pat = build_lora_target_regex(12, cfg)
    assert re.fullmatch(pat, "model.layers.1.mlp.down_proj")
    assert re.fullmatch(pat, "model.layers.11.mlp.down_proj") is None


def test_regex_ignores_embeddings_and_head():
    assert fullmatch("model.embed_tokens") is None
    assert fullmatch("lm_head") is None


# ------------------------------------------------------------ param groups --
class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 2, bias=False)


class FakeModel(nn.Module):
    """Reproduces the parameter NAMES build_param_groups dispatches on."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(12)])
        self.lora_A = nn.Linear(2, 2, bias=False)       # 'lora_' in name
        self.w_target = nn.Parameter(torch.zeros(2, 2))
        self.target_conv = nn.Conv1d(2, 2, 2, bias=False)


def _freeze_except(model, cfg):
    suffixes = ttt_down_suffixes(cfg)
    for name, p in model.named_parameters():
        keep = ("lora_" in name or "w_target" in name
                or "target_conv" in name
                or any(name.endswith(s) for s in suffixes))
        p.requires_grad_(keep)


def test_param_groups_classify_and_count():
    model = FakeModel()
    _freeze_except(model, CFG)
    groups, named = build_param_groups(
        model, CFG, lr_lora=1e-4, lr_wdown=2e-5, lr_new=2e-4,
        wd_full=0.1, wd_lora=0.0,
    )
    assert len(named["lora"]) == 1
    assert len(named["wdown"]) == 2          # layers 5 and 11
    assert len(named["new"]) == 2            # w_target + conv
    assert groups[0]["lr"] == 1e-4 and groups[0]["weight_decay"] == 0.0
    assert groups[1]["lr"] == 2e-5 and groups[2]["lr"] == 2e-4


def test_param_groups_reject_unknown_trainable():
    model = FakeModel()
    _freeze_except(model, CFG)
    model.layers[3].mlp.down_proj.weight.requires_grad_(True)  # not a TTT layer
    with pytest.raises(RuntimeError, match="Unclassified"):
        build_param_groups(model, CFG, 1e-4, 2e-5, 2e-4, 0.1, 0.0)


# -------------------------------------------------------- state dict I/O --
def test_strip_peft_prefix():
    assert strip_peft_prefix("base_model.model.layers.5.x") == "layers.5.x"
    assert strip_peft_prefix("layers.5.x") == "layers.5.x"


def test_save_load_roundtrip_across_peft_wrapping(tmp_path):
    """Save from a bare model, load into a PEFT-prefixed one. Values
    must land in the right tensors; LoRA-only params must be ignored."""
    src = FakeModel()
    for p in src.parameters():
        nn.init.normal_(p)
    path = str(tmp_path / "ttt.pt")
    save_ttt_state_dict(src, path, CFG)

    wrapper = nn.Module()                     # names become base_model.model.*
    wrapper.base_model = nn.Module()
    wrapper.base_model.model = FakeModel()
    dst = wrapper.base_model.model
    load_ttt_state_dict(wrapper, path)

    assert torch.equal(dst.w_target, src.w_target)
    assert torch.equal(dst.target_conv.weight, src.target_conv.weight)
    for i in (5, 11):
        assert torch.equal(dst.layers[i].mlp.down_proj.weight,
                           src.layers[i].mlp.down_proj.weight)
    for i in (0, 3):                          # non-TTT layers untouched
        assert not torch.equal(dst.layers[i].mlp.down_proj.weight,
                               src.layers[i].mlp.down_proj.weight)


def test_load_raises_on_layer_mismatch(tmp_path):
    src = FakeModel()
    path = str(tmp_path / "ttt.pt")
    save_ttt_state_dict(src, path, CFG)

    class Tiny(nn.Module):                    # lacks layers 5/11 entirely
        def __init__(self):
            super().__init__()
            self.w_target = nn.Parameter(torch.zeros(2, 2))

    with pytest.raises(RuntimeError, match="mismatch"):
        load_ttt_state_dict(Tiny(), path)
