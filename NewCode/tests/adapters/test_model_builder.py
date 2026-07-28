from __future__ import annotations

from tests.ports import _builders
from ttt.adapters.model_builder import patch_model_with_ttt, unfreeze_ttt_params
from ttt.core import naming
from ttt.core.config.ttt import TTTConfig, derive_layer_indices
from ttt.extensions.mechanism import InPlaceTTTMLP

# target_conv, w_target, output_gate weight+bias, down_proj; v_source_norm stays frozen.
THAWED_PER_LAYER = 5


def mlp_at(model, index):
    return model.model.layers[index].mlp


def test_it_replaces_the_mlp_on_every_named_layer():
    model = _builders.TinyCausalLM()

    patch_model_with_ttt(model, TTTConfig(layer_indices=(1, 3)))

    assert all(isinstance(mlp_at(model, i), InPlaceTTTMLP) for i in (1, 3))


def test_it_leaves_the_other_layers_alone():
    model = _builders.TinyCausalLM()

    patch_model_with_ttt(model, TTTConfig(layer_indices=(1, 3)))

    assert not any(isinstance(mlp_at(model, i), InPlaceTTTMLP) for i in (0, 2))


def test_the_original_projections_are_reused_not_recreated():
    model = _builders.TinyCausalLM()
    original = model.model.layers[1].mlp.down_proj

    patch_model_with_ttt(model, TTTConfig(layer_indices=(1,)))

    assert mlp_at(model, 1).down_proj is original


def test_layer_indices_are_derived_from_depth_when_unset():
    model = _builders.TinyCausalLM()

    cfg = patch_model_with_ttt(model, TTTConfig())

    assert cfg.layer_indices == derive_layer_indices(model.config.num_hidden_layers)


def test_the_returned_config_is_what_the_model_was_patched_with():
    model = _builders.TinyCausalLM()

    cfg = patch_model_with_ttt(model, TTTConfig())

    assert all(isinstance(mlp_at(model, i), InPlaceTTTMLP) for i in cfg.layer_indices)


def test_a_patched_model_still_runs_a_forward():
    import torch

    model, _ = _builders.tiny_model()

    out = model(input_ids=torch.tensor([_builders.token_ids(8)]))

    assert out.logits.shape[-1] == _builders.VOCAB


def test_unfreezing_thaws_the_new_modules():
    model, cfg = _builders.tiny_model()

    thawed = {
        name for name, p in model.named_parameters() if p.requires_grad
    }

    assert any("w_target" in name for name in thawed)


def test_unfreezing_thaws_the_fast_weight():
    model, cfg = _builders.tiny_model()
    ttt_down = naming.ttt_down_suffixes(cfg.layer_indices)

    thawed = {name for name, p in model.named_parameters() if p.requires_grad}

    assert any(name.endswith(suffix) for name in thawed for suffix in ttt_down)


def test_the_normalisation_gain_stays_frozen():
    model, _ = _builders.tiny_model()

    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}

    assert any("v_source_norm" in name for name in frozen)


def test_a_non_ttt_down_projection_stays_frozen():
    model, _ = _builders.tiny_model(layer_indices=(1, 3))

    assert not model.model.layers[0].mlp.down_proj.weight.requires_grad


def test_every_thawed_parameter_is_classifiable():
    model, cfg = _builders.tiny_model()
    ttt_down = naming.ttt_down_suffixes(cfg.layer_indices)

    unclassified = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and naming.classify_param(name, ttt_down) is None
    ]

    assert unclassified == []


def test_unfreezing_reports_how_many_it_thawed():
    model = _builders.TinyCausalLM()
    cfg = TTTConfig(layer_indices=(1, 3))
    patch_model_with_ttt(model, cfg)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    assert unfreeze_ttt_params(model, cfg) == len(cfg.layer_indices) * THAWED_PER_LAYER
