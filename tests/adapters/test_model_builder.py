from __future__ import annotations

import pytest

from tests.ports import _builders
from ttt.adapters.model_builder import (
    patch_model_with_ttt,
    prepare_for_training,
    unfreeze_ttt_params,
)
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


class _SpyModel:
    """Records what `prepare_for_training` asks of a model. `TinyCausalLM` has no
    checkpointing API, and the real one needs a download."""

    def __init__(self):
        self.calls = []
        self.checkpoint_kwargs = None
        self.config = type("Config", (), {"use_cache": True})()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.calls.append("gradient_checkpointing_enable")
        self.checkpoint_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        self.calls.append("enable_input_require_grads")

    def train(self):
        self.calls.append("train")


class TestPrepareForTraining:
    def test_it_turns_on_gradient_checkpointing(self):
        model = _SpyModel()

        prepare_for_training(model)

        assert "gradient_checkpointing_enable" in model.calls

    def test_it_asks_for_the_non_reentrant_implementation(self):
        model = _SpyModel()

        prepare_for_training(model)

        assert model.checkpoint_kwargs == {"use_reentrant": False}

    def test_it_enables_input_grads(self):
        """Belt and braces under `use_reentrant=False`; what makes the reentrant
        path safe if anyone switches back."""
        model = _SpyModel()

        prepare_for_training(model)

        assert "enable_input_require_grads" in model.calls

    def test_it_turns_the_kv_cache_off(self):
        model = _SpyModel()

        prepare_for_training(model)

        assert model.config.use_cache is False

    def test_it_puts_the_model_in_train_mode(self):
        """`from_pretrained` returns an eval-mode model, which silently disables
        LoRA dropout and flips `self.training` inside the mechanism."""
        model = _SpyModel()

        prepare_for_training(model)

        assert "train" in model.calls

    def test_input_grads_are_enabled_after_checkpointing_not_before(self):
        model = _SpyModel()

        prepare_for_training(model)

        assert model.calls.index("gradient_checkpointing_enable") < model.calls.index(
            "enable_input_require_grads"
        )


@pytest.mark.integration
class TestPrepareForTrainingOnARealModel:
    """A spy proves the calls happen; this proves transformers accepts them."""

    @pytest.fixture(scope="class")
    def prepared(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        from ttt.core.config.train import TrainConfig

        model = AutoModelForCausalLM.from_pretrained(
            "peft-internal-testing/tiny-dummy-qwen2"
        )
        depth = model.config.num_hidden_layers
        # Under the default chunk_size the 8-token probe below would hit
        # `_scan_forward`'s early return and never exercise the TTT path.
        cfg = patch_model_with_ttt(
            model, TTTConfig(layer_indices=(0,), chunk_size=4)
        )
        train_cfg = TrainConfig()
        model = get_peft_model(
            model,
            LoraConfig(
                r=train_cfg.lora_r,
                lora_alpha=train_cfg.lora_alpha,
                target_modules=naming.lora_target_regex(depth, cfg.layer_indices),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        unfreeze_ttt_params(model, cfg)
        prepare_for_training(model)
        return model

    def test_the_model_reports_gradient_checkpointing(self, prepared):
        assert prepared.is_gradient_checkpointing

    def test_the_model_is_in_train_mode(self, prepared):
        assert prepared.training

    def test_the_kv_cache_is_off(self, prepared):
        assert prepared.config.use_cache is False

    def test_a_backward_still_reaches_the_fast_weight(self, prepared):
        """Checkpointing must not sever the graph to the fast weight — a broken
        segment yields a None gradient rather than an error."""
        import torch

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        prepared(input_ids=ids, labels=ids).loss.backward()

        target = next(
            p for n, p in prepared.named_parameters() if n.endswith("w_target")
        )
        assert target.grad is not None and torch.isfinite(target.grad).all()
