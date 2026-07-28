"""The framework seam: base model + TTT patch + LoRA, assembled in that order.

`patch_model_with_ttt` runs before PEFT so the adapters wrap gate/up *inside*
each InPlaceTTTMLP, and `core.naming.lora_target_regex` keeps them off the
down_proj of a TTT layer, which is the fast weight.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ttt.core import naming
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig, derive_layer_indices
from ttt.extensions.mechanism import InPlaceTTTMLP


def patch_model_with_ttt(model: torch.nn.Module, cfg: TTTConfig) -> TTTConfig:
    """Replace the MLP on cfg.layer_indices. Returns the config with indices filled."""
    if cfg.layer_indices is None:
        cfg = replace(
            cfg, layer_indices=derive_layer_indices(model.config.num_hidden_layers)
        )
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    for index in cfg.layer_indices:
        layer = model.model.layers[index]
        device = layer.mlp.down_proj.weight.device
        layer.mlp = InPlaceTTTMLP(layer.mlp, hidden, cfg).to(device=device, dtype=dtype)
    return cfg


def unfreeze_ttt_params(model: torch.nn.Module, cfg: TTTConfig) -> int:
    """PEFT freezes everything non-LoRA; re-enable the TTT trainables."""
    ttt_down = naming.ttt_down_suffixes(cfg.layer_indices or ())
    unfrozen = 0
    for name, parameter in model.named_parameters():
        if naming.is_frozen_ttt_param(name):
            continue
        if naming.classify_param(name, ttt_down) in (naming.GROUP_NEW, naming.GROUP_WDOWN):
            parameter.requires_grad_(True)
            unfrozen += 1
    return unfrozen


def build_model(
    base_model: str,
    *,
    ttt_cfg: TTTConfig,
    train_cfg: TrainConfig,
    adapter_path: str | None = None,
    trainable: bool = True,
    attn_implementation: str = "sdpa",
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="cuda",
    )
    num_layers = model.config.num_hidden_layers
    ttt_cfg = patch_model_with_ttt(model, ttt_cfg)
    model = _wrap_with_lora(
        model,
        num_layers=num_layers,
        ttt_cfg=ttt_cfg,
        train_cfg=train_cfg,
        adapter_path=adapter_path,
        trainable=trainable,
    )
    if trainable:
        unfreeze_ttt_params(model, ttt_cfg)
    return model, ttt_cfg


def _wrap_with_lora(
    model, *, num_layers, ttt_cfg, train_cfg, adapter_path, trainable
):
    if adapter_path:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)

    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=naming.lora_target_regex(
                num_layers, ttt_cfg.layer_indices or ()
            ),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
