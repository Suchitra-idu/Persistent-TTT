"""The frozen configs themselves: validation, and the D9/D11 invariants."""

from __future__ import annotations

import dataclasses

import pytest

from ttt.core.config import presets
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig, derive_layer_indices

SLIM = DatasetSpec(
    name="slimpajama-6b",
    source="DKYoon/SlimPajama-6B",
    source_meta_column="meta",
    source_meta_key="redpajama_set_name",
    include_sources=("RedPajamaC4", "RedPajamaBook"),
    default_source_preset="slim-research",
)


@pytest.mark.parametrize(
    ("config", "field", "value"),
    [
        (TTTConfig(), "chunk_size", 64),
        (TrainConfig(), "num_epochs", 5),
        (SLIM, "holdout_last_n", 10),
    ],
)
def test_configs_cannot_be_mutated_in_place(config, field, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field, value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_size": 0}, "chunk_size must be >= 1"),
        ({"conv_kernel_size": 0}, "conv_kernel_size must be >= 1"),
        ({"clip_tau": 0.0}, "clip_tau must be > 0"),
        ({"carried_decay": 1.5}, r"carried_decay must be in \[0, 1\]"),
        ({"update_rule": "gradient"}, "update_rule must be"),
        ({"truncate_every": 0}, "truncate_every must be >= 1"),
        ({"carry_gate": True, "update_rule": "delta_chunk"}, "carry_gate is implemented"),
        ({"layer_indices": (1, 1, 3)}, "duplicates"),
        ({"layer_indices": (-1,)}, "non-negative"),
    ],
)
def test_ttt_config_rejects_impossible_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        TTTConfig(**overrides)


def test_clipping_off_means_no_tau_reaches_the_scan():
    assert TTTConfig(clip_enabled=False).effective_clip_tau is None


def test_clipping_on_passes_tau_to_the_scan():
    assert TTTConfig(clip_enabled=True, clip_tau=4.0).effective_clip_tau == 4.0


def test_layer_indices_are_every_stride_th_layer():
    assert derive_layer_indices(28, stride=2, start=1) == tuple(range(1, 28, 2))


def test_layer_indices_never_exceed_the_models_depth():
    assert max(derive_layer_indices(28)) < 28


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"micro_batch_size": 4}, "fixed at 1"),
        ({"grad_accum_steps": 0}, "grad_accum_steps must be >= 1"),
        ({"max_steps": -1}, "max_steps must be >= 0"),
        ({"min_doc_tokens": 0}, "min_doc_tokens must be >= 1"),
        ({"max_seq_len": 512}, "below min_doc_tokens"),
        ({"warmup_ratio": 2.0}, r"warmup_ratio must be in \[0, 1\]"),
        ({"max_grad_norm": 0.0}, "max_grad_norm must be > 0"),
        ({"eval_n_slices": 0}, "eval_n_slices must be >= 1"),
        ({"carry_window": 0}, "carry_window must be >= 1"),
        ({"strategy": ""}, "must name a registered session strategy"),
    ],
)
def test_train_config_rejects_impossible_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**overrides)


def test_a_spec_must_label_every_row_somehow():
    with pytest.raises(ValueError, match="must name source_meta_column"):
        DatasetSpec(name="unlabelled", source="somewhere")


def test_a_struct_meta_column_yields_the_label():
    assert SLIM.source_of({"meta": {"redpajama_set_name": "RedPajamaC4"}}) == (
        "RedPajamaC4"
    )


def test_a_json_string_meta_column_yields_the_same_label():
    """Parquet shards and the Hub disagree about which type this column is."""
    assert SLIM.source_of({"meta": '{"redpajama_set_name": "RedPajamaC4"}'}) == (
        "RedPajamaC4"
    )


def test_an_unlabellable_row_is_an_error_not_an_empty_string():
    with pytest.raises(ValueError, match="no usable 'meta' field"):
        SLIM.source_of({"text": "no meta here"})


def test_an_empty_label_is_an_error():
    with pytest.raises(ValueError, match="every row must carry a source label"):
        SLIM.source_of({"meta": {"redpajama_set_name": ""}})


def test_the_holdout_is_the_newest_rows():
    spec = DatasetSpec(
        name="s", source="x", source_meta_column="meta", source_meta_key="set", holdout_last_n=5000
    )

    assert spec.holdout_boundary(1_000_000) == 995_000


def test_a_pool_smaller_than_the_holdout_leaves_no_training_rows():
    spec = DatasetSpec(name="s", source="x", source_meta_column="meta", source_meta_key="set", holdout_last_n=100)

    assert spec.holdout_boundary(50) == 0


def test_a_preset_is_a_plain_mapping_of_weights():
    assert presets.get_preset("slim-research")["RedPajamaC4"] == 25


def test_the_paper_preset_is_the_studied_source_mix():
    assert dict(presets.get_preset("slim-research")) == {
        "RedPajamaC4": 25, "RedPajamaGithub": 20, "RedPajamaBook": 15,
        "RedPajamaArXiv": 20, "RedPajamaWikipedia": 10, "RedPajamaStackExchange": 10,
    }
