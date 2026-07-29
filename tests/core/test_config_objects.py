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


def test_the_cut_fields_are_gone():
    """v_source and gate_reg_weight died with the embedding tap (D11)."""
    fields = {f.name for f in dataclasses.fields(TTTConfig)}

    assert "v_source" not in fields and "gate_reg_weight" not in fields


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_size": 0}, "chunk_size must be >= 1"),
        ({"conv_kernel_size": 0}, "conv_kernel_size must be >= 1"),
        ({"clip_tau": 0.0}, "clip_tau must be > 0"),
        ({"carried_decay": 1.5}, r"carried_decay must be in \[0, 1\]"),
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


def test_a_wider_stride_gives_fewer_ttt_layers():
    assert len(derive_layer_indices(28, stride=4)) < len(
        derive_layer_indices(28, stride=2)
    )


def test_layer_indices_never_exceed_the_models_depth():
    assert max(derive_layer_indices(28)) < 28


def test_the_session_mode_booleans_are_gone():
    """Four modes became one strategy name resolved against a registry (D4)."""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}

    assert not fields & {
        "single_paper_sessions",
        "hybrid_sessions",
        "everlasting_carry",
    }
    assert "strategy" in fields


def test_the_cut_regimes_knobs_are_gone():
    fields = {f.name for f in dataclasses.fields(TrainConfig)}

    assert not {f for f in fields if f.startswith(("slice_", "session_papers"))}


def test_the_strategy_owns_its_own_knobs():
    """hybrid_* moved to the plugin, so deleting it deletes its knobs."""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}

    assert not {f for f in fields if f.startswith("hybrid_")}


def test_paper_vocabulary_is_gone_from_the_eval_knobs():
    fields = {f.name for f in dataclasses.fields(TrainConfig)}

    assert "eval_n_docs" in fields and "eval_n_papers" not in fields


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"micro_batch_size": 4}, "fixed at 1"),
        ({"grad_accum_steps": 0}, "grad_accum_steps must be >= 1"),
        ({"min_doc_tokens": 0}, "min_doc_tokens must be >= 1"),
        ({"max_seq_len": 512}, "below min_doc_tokens"),
        ({"warmup_ratio": 2.0}, r"warmup_ratio must be in \[0, 1\]"),
        ({"max_grad_norm": 0.0}, "max_grad_norm must be > 0"),
        ({"eval_n_slices": 0}, "eval_n_slices must be >= 1"),
        ({"strategy": ""}, "must name a registered session strategy"),
    ],
)
def test_train_config_rejects_impossible_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**overrides)


def test_a_spec_must_label_every_row_somehow():
    with pytest.raises(ValueError, match="must label every row exactly one way"):
        DatasetSpec(name="unlabelled", source="somewhere")


def test_a_spec_may_not_label_rows_two_ways_at_once():
    with pytest.raises(ValueError, match="exactly one way"):
        DatasetSpec(
            name="both",
            source="somewhere",
            source_meta_column="meta",
            source_meta_key="set",
            constant_source="fixed",
        )


def test_the_tokens_est_column_is_gone():
    """Token counts are always estimated now (D9)."""
    fields = {f.name for f in dataclasses.fields(DatasetSpec)}

    assert "tokens_est_column" not in fields


def test_a_struct_meta_column_yields_the_label():
    assert SLIM.source_of({"meta": {"redpajama_set_name": "RedPajamaC4"}}) == (
        "RedPajamaC4"
    )


def test_a_json_string_meta_column_yields_the_same_label():
    """Parquet shards and the Hub disagree about which type this column is."""
    assert SLIM.source_of({"meta": '{"redpajama_set_name": "RedPajamaC4"}'}) == (
        "RedPajamaC4"
    )


def test_a_single_source_spec_stamps_a_constant_on_every_row():
    spec = DatasetSpec(name="fixture", source="local", constant_source="synthetic")

    assert spec.source_of({}) == "synthetic"


def test_an_unlabellable_row_is_an_error_not_an_empty_string():
    with pytest.raises(ValueError, match="no usable 'meta' field"):
        SLIM.source_of({"text": "no meta here"})


def test_an_empty_label_is_an_error():
    with pytest.raises(ValueError, match="every row must carry a source label"):
        SLIM.source_of({"meta": {"redpajama_set_name": ""}})


def test_include_sources_filters_by_label():
    assert SLIM.keeps("RedPajamaC4") and not SLIM.keeps("RedPajamaCommonCrawl")


def test_no_include_list_keeps_every_source():
    spec = DatasetSpec(name="all", source="x", constant_source="s")

    assert spec.keeps("anything")


def test_an_empty_include_list_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="empty include_sources"):
        DatasetSpec(
            name="empty", source="x", constant_source="s", include_sources=()
        )


def test_the_holdout_is_the_newest_rows():
    spec = DatasetSpec(
        name="s", source="x", constant_source="c", holdout_last_n=5000
    )

    assert spec.holdout_boundary(1_000_000) == 995_000


def test_a_pool_smaller_than_the_holdout_leaves_no_training_rows():
    spec = DatasetSpec(name="s", source="x", constant_source="c", holdout_last_n=100)

    assert spec.holdout_boundary(50) == 0


def test_a_preset_is_a_plain_mapping_of_weights():
    assert presets.get_preset("slim-research")["RedPajamaC4"] == 25


def test_the_research_preset_downweights_c4_against_the_paper_preset():
    paper = presets.get_preset("slim-paper")
    research = presets.get_preset("slim-research")

    assert research["RedPajamaC4"] < paper["RedPajamaC4"]


def test_presets_cannot_be_mutated():
    with pytest.raises(TypeError):
        presets.SOURCE_PRESETS["slim-paper"]["RedPajamaC4"] = 1


def test_an_unknown_preset_names_the_known_ones():
    with pytest.raises(KeyError, match="slim-paper"):
        presets.get_preset("nonexistent")


def test_only_sources_builds_an_equal_weighted_preset():
    name, weights = presets.only_sources_preset(["RedPajamaBook", "RedPajamaC4"])

    assert dict(weights) == {"RedPajamaBook": 1, "RedPajamaC4": 1}
    assert name.startswith(presets.ONLY_PREFIX)


def test_only_sources_names_are_order_independent():
    first, _ = presets.only_sources_preset(["b", "a"])
    second, _ = presets.only_sources_preset(["a", "b"])

    assert first == second


def test_only_sources_rejects_a_source_the_spec_does_not_have():
    with pytest.raises(ValueError, match="unknown source"):
        presets.only_sources_preset(["Typo"], allowed=SLIM.include_sources)


def test_parsing_only_sources_splits_on_commas():
    assert presets.parse_only_sources(" a , b ") == ("a", "b")


def test_parsing_an_empty_only_sources_value_is_an_error():
    with pytest.raises(ValueError, match="parsed empty"):
        presets.parse_only_sources(" , ")


def test_an_explicit_preset_beats_the_spec_default():
    assert presets.resolve_preset_name("slim-paper", "slim-research") == "slim-paper"


def test_an_unset_preset_falls_back_to_the_spec_default():
    assert presets.resolve_preset_name("", "slim-research") == "slim-research"


def test_none_opts_out_of_the_spec_default():
    assert presets.resolve_preset_name("none", "slim-research") == ""


def test_a_synthesised_only_preset_cannot_be_looked_up_by_name():
    with pytest.raises(KeyError, match="synthesised only-sources preset"):
        presets.get_preset("_only_RedPajamaBook")
