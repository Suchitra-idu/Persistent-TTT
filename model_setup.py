"""
Model assembly shared by the training and inference apps.

One function builds the full stack in the one order that is correct:

    base Qwen3 (bf16) -> patch TTT layers -> wrap with LoRA
    -> unfreeze TTT trainables -> (optionally) load a checkpoint

Keeping this in one place means train and inference can never assemble
the model differently (DRY), which would silently corrupt evaluation.

Model-size invariant: NUM_LAYERS and the TTT layer schedule are derived
from model.config.num_hidden_layers right after the HF load, so swapping
BASE_MODEL (e.g. 8B -> 0.6B via TTT_MODEL_SIZE) needs no edits here.
"""

import os

import torch

from inplace_ttt import (
    build_lora_config,
    load_ttt_state_dict,
    patch_model_with_ttt,
    unfreeze_ttt_params,
)
from ttt_config import BASE_MODEL, TTT_CFG, TRAIN_CFG, derive_ttt_layer_indices


def build_model(adapter_path: str | None = None,
                ttt_ckpt_path: str | None = None,
                trainable: bool = True,
                attn_impl: str = "flash_attention_2"):
    """Build base + TTT + LoRA. Pass adapter_path / ttt_ckpt_path to
    resume or to load for inference. trainable=False skips grad setup.
    attn_impl="sdpa" lets inference skip the flash-attn dependency."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="cuda",
    )

    # Derive the TTT layer schedule from the actual loaded model. Populated
    # once, idempotent across train/infer (re-deriving with the same
    # num_layers yields the same tuple). Skip if a caller has already
    # supplied an explicit schedule (e.g. tests, ablations).
    num_layers = model.config.num_hidden_layers
    if TTT_CFG.layer_indices is None:
        TTT_CFG.layer_indices = derive_ttt_layer_indices(num_layers)

    # 1. TTT patch must precede LoRA so PEFT sees and wraps the gate/up
    #    projections living inside InPlaceTTTMLP.
    patch_model_with_ttt(model, TTT_CFG)

    # 2. LoRA everywhere except down_proj on TTT layers (regex-enforced).
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

    # 3. PEFT froze everything non-LoRA; re-enable the TTT trainables.
    if trainable:
        unfreeze_ttt_params(model, TTT_CFG)

    # 4. Restore W_down / W_target / Conv1D from a checkpoint if given.
    if ttt_ckpt_path and os.path.exists(ttt_ckpt_path):
        load_ttt_state_dict(model, ttt_ckpt_path)

    return model, tokenizer
