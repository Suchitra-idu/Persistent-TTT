"""LoRA config, parameter grouping, and checkpointing for TTT-patched models."""

import torch

from ttt_config import TTTConfig

_PEFT_PREFIX = "base_model.model."
TTT_PARAM_MARKERS = ("target_conv", "w_target", "output_gate", "v_source_norm")
# v_source_norm gamma frozen at 1.0; kept in markers so it lands in checkpoints.
TTT_FROZEN_MARKERS = ("v_source_norm",)


def ttt_down_suffixes(cfg: TTTConfig) -> set:
    return {f"layers.{i}.mlp.down_proj.weight" for i in cfg.layer_indices}


def strip_peft_prefix(name: str) -> str:
    return name[len(_PEFT_PREFIX):] if name.startswith(_PEFT_PREFIX) else name


def build_lora_target_regex(num_layers: int, cfg: TTTConfig) -> str:
    """LoRA targets attention + gate/up everywhere, plus down_proj ONLY on non-TTT layers.
    Silent failure mode is LoRA landing on a fast weight, so this is unit-testable."""
    non_ttt = [str(i) for i in range(num_layers) if i not in cfg.layer_indices]
    return (
        r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"
        r"|.*\.layers\.(" + "|".join(non_ttt) + r")\.mlp\.down_proj"
    )


def build_lora_config(num_layers: int, cfg: TTTConfig,
                      r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=build_lora_target_regex(num_layers, cfg),
        bias="none", task_type="CAUSAL_LM",
    )


def _classify_ttt_param(name: str, ttt_down: set) -> str | None:
    if "lora_" in name:
        return "lora"
    if any(marker in name for marker in TTT_PARAM_MARKERS):
        return "new"
    if any(name.endswith(suffix) for suffix in ttt_down):
        return "wdown"
    return None


def unfreeze_ttt_params(model, cfg: TTTConfig):
    """PEFT freezes everything non-LoRA; re-enable grads for TTT trainables.
    TTT_FROZEN_MARKERS stay frozen by design."""
    ttt_down = ttt_down_suffixes(cfg)
    for name, p in model.named_parameters():
        if any(m in name for m in TTT_FROZEN_MARKERS):
            continue
        if _classify_ttt_param(name, ttt_down) in ("new", "wdown"):
            p.requires_grad_(True)


def build_param_groups(model, cfg: TTTConfig, lr_lora: float,
                       lr_wdown: float, lr_new: float,
                       wd_full: float, wd_lora: float):
    ttt_down = ttt_down_suffixes(cfg)
    groups = {"lora": [], "wdown": [], "new": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = _classify_ttt_param(name, ttt_down)
        if group is None:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
        groups[group].append(p)
    optim_groups = [
        {"params": groups["lora"], "lr": lr_lora, "weight_decay": wd_lora},
        {"params": groups["wdown"], "lr": lr_wdown, "weight_decay": wd_full},
        {"params": groups["new"], "lr": lr_new, "weight_decay": wd_full},
    ]
    return optim_groups, groups


# Checkpoint keys stored relative to BASE model so loading works with or without PEFT wrap.
def save_ttt_state_dict(model, path: str, cfg: TTTConfig):
    ttt_down = ttt_down_suffixes(cfg)
    out = {}
    for name, p in model.named_parameters():
        key = strip_peft_prefix(name)
        if any(m in key for m in TTT_PARAM_MARKERS) or \
           any(key.endswith(s) for s in ttt_down):
            out[key] = p.detach().cpu()
    torch.save(out, path)


def load_ttt_state_dict(model, path: str):
    saved = torch.load(path, map_location="cpu")
    by_suffix = dict(saved)
    loaded = 0
    for name, p in model.named_parameters():
        key = strip_peft_prefix(name)
        if key in by_suffix:
            p.data.copy_(by_suffix[key].to(p.device, p.dtype))
            loaded += 1
    if loaded != len(saved):
        raise RuntimeError(
            f"TTT checkpoint mismatch, saved {len(saved)} tensors, "
            f"loaded {loaded}. Check layer indices match the checkpoint."
        )
