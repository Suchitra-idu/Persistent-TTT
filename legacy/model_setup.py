"""Model assembly shared by the training and inference apps."""

import os

import torch

from inplace_ttt import patch_model_with_ttt
from ttt_wiring import build_lora_config, load_ttt_state_dict, unfreeze_ttt_params
from ttt_config import BASE_MODEL, TTT_CFG, TRAIN_CFG, derive_ttt_layer_indices


def build_model(adapter_path: str | None = None,
                ttt_ckpt_path: str | None = None,
                trainable: bool = True,
                attn_impl: str = "flash_attention_2"):
    """Build base + TTT + LoRA. attn_impl='sdpa' lets inference skip flash-attn."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="cuda",
    )

    num_layers = model.config.num_hidden_layers
    if TTT_CFG.layer_indices is None:
        TTT_CFG.layer_indices = derive_ttt_layer_indices(num_layers)

    # TTT patch must precede LoRA so PEFT wraps gate/up projections inside InPlaceTTTMLP.
    patch_model_with_ttt(model, TTT_CFG)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=trainable
        )
    else:
        from peft import get_peft_model
        lora_cfg = build_lora_config(
            num_layers, TTT_CFG,
            r=TRAIN_CFG.lora_r, alpha=TRAIN_CFG.lora_alpha,
            dropout=TRAIN_CFG.lora_dropout,
        )
        model = get_peft_model(model, lora_cfg)

    # PEFT froze everything non-LoRA; re-enable the TTT trainables.
    if trainable:
        unfreeze_ttt_params(model, TTT_CFG)

    if ttt_ckpt_path and os.path.exists(ttt_ckpt_path):
        load_ttt_state_dict(model, ttt_ckpt_path)

    return model, tokenizer
