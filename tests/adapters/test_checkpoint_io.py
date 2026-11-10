from __future__ import annotations

import pytest
import torch

from tests.ports import _builders
from ttt.adapters import checkpoint_io
from ttt.adapters.in_memory_storage import InMemoryStorage

PARAMS = "run/ckpt-100/" + checkpoint_io.TTT_PARAMS


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def model_and_cfg():
    return _builders.tiny_model()


def test_it_saves_every_ttt_tensor_of_every_ttt_layer(storage, model_and_cfg):
    model, cfg = model_and_cfg

    saved = checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )

    assert saved == len(cfg.layer_indices) * len(_builders.TTT_TENSOR_SUFFIXES)


def test_a_parameter_round_trips(storage, model_and_cfg):
    model, cfg = model_and_cfg
    checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )
    target = model.model.layers[cfg.layer_indices[0]].mlp.w_target
    original = target.detach().clone()
    with torch.no_grad():
        target.zero_()

    checkpoint_io.load_ttt_params(model, storage, PARAMS)

    assert torch.equal(target, original)


def test_a_checkpoint_for_other_layers_is_rejected(storage, model_and_cfg):
    model, cfg = model_and_cfg
    checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )
    other, _ = _builders.tiny_model(layer_indices=(0, 2))

    with pytest.raises(RuntimeError, match="the model does not"):
        checkpoint_io.load_ttt_params(other, storage, PARAMS)


def test_a_shape_change_is_reported_before_anything_is_copied(storage, model_and_cfg):
    model, cfg = model_and_cfg
    checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )
    wider, _ = _builders.tiny_model(seed=99, conv_kernel_size=cfg.conv_kernel_size + 1)
    original = wider.model.layers[cfg.layer_indices[0]].mlp.w_target.detach().clone()

    with pytest.raises(RuntimeError, match="rebuild the checkpoint"):
        checkpoint_io.load_ttt_params(wider, storage, PARAMS)

    assert torch.equal(wider.model.layers[cfg.layer_indices[0]].mlp.w_target, original)


def test_it_saves_the_fast_weight_alongside_the_new_modules(storage, model_and_cfg):
    model, cfg = model_and_cfg
    checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )
    layer = cfg.layer_indices[0]

    keys = checkpoint_io.keys_in(storage, PARAMS)

    assert set(keys) >= {
        f"model.layers.{layer}.mlp.{suffix}"
        for suffix in _builders.TTT_TENSOR_SUFFIXES
    }


def test_it_saves_nothing_that_is_not_a_ttt_key(storage, model_and_cfg):
    model, cfg = model_and_cfg
    checkpoint_io.save_ttt_params(
        model, storage, PARAMS, layer_indices=cfg.layer_indices
    )

    keys = checkpoint_io.keys_in(storage, PARAMS)

    assert not any("lm_head" in key or "embed" in key for key in keys)
