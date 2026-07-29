"""Parameter naming: what LoRA may touch, and which group a tensor trains in."""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from ttt.core import naming

LAYERS = 28
TTT_LAYERS = tuple(range(1, LAYERS, 2))
TTT_DOWN = naming.ttt_down_suffixes(TTT_LAYERS)


def _matches(name: str) -> bool:
    return re.fullmatch(naming.lora_target_regex(LAYERS, TTT_LAYERS), name) is not None


@given(layer=st.sampled_from(TTT_LAYERS))
def test_lora_never_targets_a_ttt_layers_down_projection(layer):
    assert not _matches(f"model.layers.{layer}.mlp.down_proj")


@given(layer=st.sampled_from([i for i in range(LAYERS) if i not in TTT_LAYERS]))
def test_lora_targets_down_projection_on_ordinary_layers(layer):
    assert _matches(f"model.layers.{layer}.mlp.down_proj")


@given(
    layer=st.integers(min_value=0, max_value=LAYERS - 1),
    projection=st.sampled_from(["q_proj", "k_proj", "v_proj", "o_proj"]),
)
def test_lora_targets_attention_on_every_layer(layer, projection):
    assert _matches(f"model.layers.{layer}.self_attn.{projection}")


@given(
    layer=st.integers(min_value=0, max_value=LAYERS - 1),
    projection=st.sampled_from(["gate_proj", "up_proj"]),
)
def test_lora_targets_gate_and_up_on_every_layer(layer, projection):
    """Both are called as modules in the TTT forward, so LoRA is included in z."""
    assert _matches(f"model.layers.{layer}.mlp.{projection}")


def test_the_regex_is_stable_for_a_given_schedule():
    assert naming.lora_target_regex(LAYERS, TTT_LAYERS) == naming.lora_target_regex(
        LAYERS, TTT_LAYERS
    )


def test_a_lora_parameter_lands_in_the_lora_group():
    assert naming.classify_param(
        "model.layers.0.self_attn.q_proj.lora_A.default.weight", TTT_DOWN
    ) == naming.GROUP_LORA


def test_a_ttt_layers_down_projection_is_a_fast_weight():
    assert naming.classify_param(
        "model.layers.1.mlp.down_proj.weight", TTT_DOWN
    ) == naming.GROUP_WDOWN


def test_an_ordinary_layers_down_projection_is_not_a_fast_weight():
    assert naming.classify_param("model.layers.0.mlp.down_proj.weight", TTT_DOWN) is None


@given(marker=st.sampled_from(naming.TTT_PARAM_MARKERS))
def test_every_ttt_module_parameter_lands_in_the_new_group(marker):
    assert naming.classify_param(
        f"model.layers.1.mlp.{marker}.weight", TTT_DOWN
    ) == naming.GROUP_NEW


def test_an_unrelated_parameter_classifies_as_nothing():
    """None is the useful answer: a trainable None means something leaked."""
    assert naming.classify_param("model.embed_tokens.weight", TTT_DOWN) is None


def test_lora_wins_over_the_ttt_markers():
    """A LoRA adapter on a gate_proj must not be mistaken for a TTT parameter."""
    assert naming.classify_param(
        "model.layers.1.mlp.output_gate.lora_A.weight", TTT_DOWN
    ) == naming.GROUP_LORA


def test_the_frozen_norm_is_recognised_as_frozen():
    assert naming.is_frozen_ttt_param("model.layers.1.mlp.v_source_norm.weight")


def test_the_frozen_norm_is_still_a_new_group_parameter():
    """Frozen but checkpointed: it must be saved, it must not get gradients."""
    name = "model.layers.1.mlp.v_source_norm.weight"

    assert naming.classify_param(name, TTT_DOWN) == naming.GROUP_NEW
    assert naming.is_checkpoint_key(name, TTT_DOWN)


def test_the_peft_prefix_is_stripped_so_keys_are_base_relative():
    assert naming.strip_peft_prefix(
        "base_model.model.model.layers.1.mlp.w_target"
    ) == "model.layers.1.mlp.w_target"


def test_stripping_leaves_an_unwrapped_name_alone():
    assert naming.strip_peft_prefix("model.layers.1.mlp.w_target") == (
        "model.layers.1.mlp.w_target"
    )


def test_fast_weights_are_checkpointed():
    assert naming.is_checkpoint_key("model.layers.1.mlp.down_proj.weight", TTT_DOWN)


def test_ordinary_weights_are_not_checkpointed():
    """They are unchanged base-model weights, or covered by the LoRA adapter."""
    assert not naming.is_checkpoint_key(
        "model.layers.0.mlp.down_proj.weight", TTT_DOWN
    )


def test_checkpoint_keys_are_recognised_through_a_peft_wrap():
    assert naming.is_checkpoint_key(
        "base_model.model.model.layers.1.mlp.w_target", TTT_DOWN
    )


def test_down_suffixes_cover_exactly_the_ttt_layers():
    assert len(TTT_DOWN) == len(TTT_LAYERS)
