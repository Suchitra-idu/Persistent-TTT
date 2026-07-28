"""Parameter-name logic: LoRA targeting, optimizer grouping, checkpoint keys."""

from __future__ import annotations

from typing import Iterable

_PEFT_PREFIX = "base_model.model."

TTT_PARAM_MARKERS = ("target_conv", "w_target", "output_gate", "v_source_norm")

# Frozen but still checkpointed, hence its presence in both tuples.
TTT_FROZEN_MARKERS = ("v_source_norm",)

GROUP_LORA = "lora"
GROUP_WDOWN = "wdown"
GROUP_NEW = "new"
PARAM_GROUPS = (GROUP_LORA, GROUP_WDOWN, GROUP_NEW)


def strip_peft_prefix(name: str) -> str:
    """Make a key base-model-relative, so it loads with or without a PEFT wrap."""
    return name[len(_PEFT_PREFIX):] if name.startswith(_PEFT_PREFIX) else name


def ttt_down_suffixes(layer_indices: Iterable[int]) -> frozenset[str]:
    return frozenset(f"layers.{i}.mlp.down_proj.weight" for i in layer_indices)


def lora_target_regex(num_layers: int, layer_indices: Iterable[int]) -> str:
    """Attention + gate/up everywhere, down_proj on non-TTT layers only.

    A TTT layer's down_proj is W0. An adapter on it would be a second thing
    editing the fast weight, with no error to say so.
    """
    ttt = set(layer_indices)
    non_ttt = [str(i) for i in range(num_layers) if i not in ttt]
    return (
        r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"
        r"|.*\.layers\.(" + "|".join(non_ttt) + r")\.mlp\.down_proj"
    )


def classify_param(name: str, ttt_down: frozenset[str]) -> str | None:
    """Optimizer group, or None. A *trainable* None means the caller should raise."""
    if "lora_" in name:
        return GROUP_LORA
    if any(marker in name for marker in TTT_PARAM_MARKERS):
        return GROUP_NEW
    if any(name.endswith(suffix) for suffix in ttt_down):
        return GROUP_WDOWN
    return None


def is_frozen_ttt_param(name: str) -> bool:
    return any(marker in name for marker in TTT_FROZEN_MARKERS)


def is_checkpoint_key(name: str, ttt_down: frozenset[str]) -> bool:
    """Whether a name belongs in ttt_params.pt: TTT params plus the fast weights."""
    key = strip_peft_prefix(name)
    return any(marker in key for marker in TTT_PARAM_MARKERS) or any(
        key.endswith(suffix) for suffix in ttt_down
    )
