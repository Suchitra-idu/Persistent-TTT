from __future__ import annotations

import pytest
import torch

from tests.ports import _builders
from ttt.adapters import checkpoint_io
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.torch_fast_weights import TorchFastWeights
from ttt.core.types import Carry

PARAMS = "run/ckpt-100/" + checkpoint_io.TTT_PARAMS
CARRIES = "run/ckpt-100/" + checkpoint_io.CARRIES


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


def test_an_absent_carry_file_is_an_empty_seed(storage):
    assert checkpoint_io.load_carries(storage, CARRIES) == ({}, {})


def test_carries_round_trip_per_source(storage, model_and_cfg):
    model, cfg = model_and_cfg
    fast_weights = TorchFastWeights(model, cfg)
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    model(input_ids=torch.tensor([_builders.token_ids(12)]))
    fast_weights.advance_carry()

    checkpoint_io.save_carries(
        {"alpha": fast_weights.snapshot(to_cpu=True)}, storage, CARRIES
    )
    loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

    fast_weights.reset_carry()
    fast_weights.install(loaded["alpha"])

    assert fast_weights.state_ratio(family="carry") > 0.0


def test_carry_keys_stay_base_model_layer_indices(storage, model_and_cfg):
    _, cfg = model_and_cfg
    carry = Carry(deltas={i: torch.ones(1, 1) for i in cfg.layer_indices})

    checkpoint_io.save_carries({"alpha": carry}, storage, CARRIES)
    loaded, _ = checkpoint_io.load_carries(storage, CARRIES)

    assert loaded["alpha"].layer_indices == cfg.layer_indices


def test_update_counts_survive_the_round_trip(storage):
    checkpoint_io.save_carries(
        {"alpha": Carry.empty()}, storage, CARRIES, n_updates={"alpha": 7}
    )

    _, meta = checkpoint_io.load_carries(storage, CARRIES)

    assert meta["n_updates"] == {"alpha": 7}


def test_metadata_survives_the_round_trip(storage):
    checkpoint_io.save_carries(
        {}, storage, CARRIES, meta={"step": 400, "strategy": "everlasting"}
    )

    _, meta = checkpoint_io.load_carries(storage, CARRIES)

    assert meta["strategy"] == "everlasting"
