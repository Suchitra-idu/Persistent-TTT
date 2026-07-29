from __future__ import annotations

import pytest

from ttt import cli
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig
from ttt.extensions import datasets
from ttt.extensions.strategies import SOURCE


class TestDefaults:
    def test_no_arguments_resolves_the_default_config(self):
        assert cli.resolve().train == TrainConfig()

    def test_no_arguments_resolves_the_default_mechanism(self):
        assert cli.resolve().ttt == TTTConfig()

    def test_no_arguments_resolves_the_default_dataset(self):
        assert cli.resolve().spec is datasets.get(datasets.DEFAULT)

    def test_no_arguments_resolves_the_default_strategy(self):
        assert cli.resolve().strategy.name == TrainConfig().strategy

    def test_the_default_base_model_is_the_proven_size(self):
        assert cli.resolve().base_model == "Qwen/Qwen3-0.6B"

    def test_layer_indices_stay_unknown_until_a_model_is_loaded(self):
        assert cli.resolve().ttt.layer_indices is None


class TestTrainOverrides:
    def test_a_named_field_reaches_the_config(self):
        assert cli.resolve(num_epochs=4).train.num_epochs == 4

    def test_an_unnamed_field_keeps_its_default(self):
        assert cli.resolve(num_epochs=4).train.seed == TrainConfig().seed

    def test_a_typo_is_rejected_by_name(self):
        with pytest.raises(ValueError, match=r"unknown option\(s\) \['num_epoch'\]"):
            cli.resolve(num_epoch=4)

    def test_a_wrongly_typed_value_is_rejected(self):
        with pytest.raises(TypeError, match="num_epochs must be int"):
            cli.resolve(num_epochs="four")

    def test_an_invalid_combination_is_rejected_by_the_config(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            cli.resolve(max_seq_len=10, min_doc_tokens=2048)

    def test_the_session_ablation_can_be_turned_off(self):
        assert cli.resolve(session_training=False).train.session_training is False

    def test_session_training_defaults_on(self):
        assert cli.resolve().train.session_training is True


class TestStrategy:
    def test_a_named_strategy_is_selected(self):
        assert cli.resolve(strategy="everlasting").strategy.name == "everlasting"

    def test_the_name_lands_on_the_train_config_too(self):
        assert cli.resolve(strategy="everlasting").train.strategy == "everlasting"

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(KeyError, match="unknown strategy"):
            cli.resolve(strategy="multi")

    def test_a_strategy_knob_configures_the_plugin(self):
        resolved = cli.resolve(strategy="hybrid", carry_min_tokens=5000)

        assert resolved.strategy.carry_min_tokens == 5000

    def test_every_hybrid_knob_flows_through(self):
        resolved = cli.resolve(
            strategy="hybrid",
            carry_min_tokens=5000,
            slice_min_tokens=1000,
            slices_min=3,
            slices_max=8,
        )

        assert (
            resolved.strategy.carry_min_tokens,
            resolved.strategy.slice_min_tokens,
            resolved.strategy.slices_min,
            resolved.strategy.slices_max,
        ) == (5000, 1000, 3, 8)

    def test_a_knob_the_strategy_does_not_have_is_rejected(self):
        with pytest.raises(ValueError, match="carry_min_tokens"):
            cli.resolve(strategy="everlasting", carry_min_tokens=5000)

    def test_an_invalid_knob_combination_is_rejected_by_the_plugin(self):
        with pytest.raises(ValueError, match="carry_min_tokens"):
            cli.resolve(strategy="hybrid", carry_min_tokens=10, slice_min_tokens=1000)

    def test_the_registry_is_not_mutated_by_configuring_a_strategy(self):
        from ttt.extensions import strategies

        cli.resolve(strategy="hybrid", carry_min_tokens=5000)

        assert strategies.get("hybrid").carry_min_tokens == 2100


class TestSourceScopedEvalCadence:
    def test_a_source_scoped_strategy_evaluates_more_often(self):
        resolved = cli.resolve(strategy="everlasting")

        assert resolved.train.eval_every == cli.EVERLASTING_EVAL_EVERY

    def test_that_only_applies_to_source_scoped_strategies(self):
        assert cli.resolve(strategy="everlasting").strategy.carry_scope == SOURCE

    def test_a_session_scoped_strategy_keeps_the_default_cadence(self):
        assert cli.resolve(strategy="hybrid").train.eval_every == TrainConfig().eval_every

    def test_an_explicit_cadence_wins(self):
        assert cli.resolve(strategy="everlasting", eval_every=10).train.eval_every == 10

    def test_an_explicit_zero_disables_eval_even_there(self):
        assert cli.resolve(strategy="everlasting", eval_every=0).train.eval_every == 0


class TestSourceWeights:
    def test_the_spec_default_preset_applies(self):
        assert "RedPajamaC4" in cli.resolve().weights

    def test_an_explicit_preset_wins(self):
        weights = cli.resolve(source_preset="slim-paper").weights

        assert weights["RedPajamaC4"] == 62

    def test_none_disables_balancing(self):
        assert cli.resolve(source_preset="none").weights is None

    def test_an_unknown_preset_is_rejected(self):
        with pytest.raises(KeyError, match="unknown source preset"):
            cli.resolve(source_preset="slim-nonsense")

    def test_only_sources_builds_equal_weights(self):
        weights = cli.resolve(only_sources="RedPajamaBook,RedPajamaArXiv").weights

        assert dict(weights) == {"RedPajamaBook": 1, "RedPajamaArXiv": 1}

    def test_only_sources_validates_against_the_spec(self):
        with pytest.raises(ValueError, match="unknown source"):
            cli.resolve(only_sources="RedPajamaNope")

    def test_only_sources_conflicts_with_an_explicit_preset(self):
        with pytest.raises(ValueError, match="conflicts with"):
            cli.resolve(only_sources="RedPajamaBook", source_preset="slim-paper")

    def test_an_empty_only_sources_is_rejected(self):
        with pytest.raises(ValueError, match="parsed empty"):
            cli.resolve(only_sources=",,")


class TestMechanismKnobs:
    def test_a_mechanism_knob_reaches_the_ttt_config(self):
        assert cli.resolve(chunk_size=64).ttt.chunk_size == 64

    def test_the_rq4_sweep_axes_are_both_exposed(self):
        resolved = cli.resolve(chunk_size=25, carried_decay=0.95)

        assert (resolved.ttt.chunk_size, resolved.ttt.carried_decay) == (25, 0.95)

    def test_a_decay_of_zero_is_a_value_not_an_omission(self):
        assert cli.resolve(carried_decay=0.0).ttt.carried_decay == 0.0

    def test_an_out_of_range_decay_is_rejected(self):
        with pytest.raises(ValueError, match="carried_decay"):
            cli.resolve(carried_decay=1.5)

    def test_layer_indices_derive_once_the_depth_is_known(self):
        assert cli.resolve(num_layers=8).ttt.layer_indices == (1, 3, 5, 7)

    def test_the_stride_is_configurable(self):
        assert cli.resolve(num_layers=8, layer_stride=4).ttt.layer_indices == (1, 5)

    def test_the_start_is_configurable(self):
        assert cli.resolve(num_layers=6, layer_start=0).ttt.layer_indices == (0, 2, 4)


class TestFromFlags:
    def test_zero_means_unset(self):
        assert cli.from_flags(num_epochs=0).train.num_epochs == TrainConfig().num_epochs

    def test_a_nonzero_int_is_taken(self):
        assert cli.from_flags(num_epochs=3).train.num_epochs == 3

    def test_an_empty_string_means_unset(self):
        assert cli.from_flags(run_name="").train.run_name == TrainConfig().run_name

    def test_grad_accum_is_spelled_the_way_the_old_cli_spelled_it(self):
        assert cli.from_flags(grad_accum=8).train.grad_accum_steps == 8

    def test_the_session_flag_is_tri_state(self):
        cases = (
            cli.from_flags(session=cli.UNSET_FLAG).train.session_training,
            cli.from_flags(session=0).train.session_training,
            cli.from_flags(session=1).train.session_training,
        )

        assert cases == (True, False, True)

    def test_a_zero_document_limit_means_no_limit(self):
        assert cli.from_flags(limit_docs=0).limit_docs is None

    def test_a_document_limit_is_taken(self):
        assert cli.from_flags(limit_docs=500).limit_docs == 500

    def test_a_float_zero_means_unset(self):
        assert cli.from_flags(eta=0.0).ttt.eta == TTTConfig().eta

    def test_a_typo_is_still_rejected(self):
        with pytest.raises(ValueError, match="unknown option"):
            cli.from_flags(grad_accume=8)


class TestEnvironmentDefaults:
    def test_the_dataset_can_be_pinned_by_environment(self):
        assert cli.env_defaults({"TTT_DATASET": "fixture"}) == {"dataset": "fixture"}

    def test_an_empty_environment_defaults_nothing(self):
        assert cli.env_defaults({}) == {}

    def test_the_base_model_can_be_overridden_outright(self):
        defaults = cli.env_defaults({"TTT_BASE_MODEL": "Qwen/Qwen3-4B"})

        assert cli.resolve(**defaults).base_model == "Qwen/Qwen3-4B"

    def test_the_model_size_composes_into_the_repo_id(self):
        defaults = cli.env_defaults({"TTT_MODEL_SIZE": "1.7B"})

        assert cli.resolve(**defaults).base_model == "Qwen/Qwen3-1.7B"

    def test_an_explicit_argument_beats_the_environment(self):
        defaults = cli.env_defaults({"TTT_DATASET": "fixture"})

        assert cli.resolve(**{**defaults, "dataset": "slimpajama-6b"}).spec.name == (
            "slimpajama-6b"
        )

    def test_unrelated_environment_variables_are_ignored(self):
        assert cli.env_defaults({"PATH": "/usr/bin", "HOME": "/root"}) == {}


class TestCheckpointPaths:
    def test_a_step_path_is_under_the_run(self):
        assert cli.step_path("ttt-v1.1", 600) == "ttt-v1.1/step_600"

    def test_a_bare_step_resumes_within_the_same_run(self):
        assert cli.resume_path("step_600", "ttt-v1.1") == "ttt-v1.1/step_600"

    def test_a_qualified_step_resumes_across_runs(self):
        assert cli.resume_path("other/step_600", "ttt-v1.1") == "other/step_600"

    def test_no_resume_is_an_empty_path(self):
        assert cli.resume_path("", "ttt-v1.1") == ""

    def test_a_saved_step_is_where_a_resume_looks_for_it(self):
        saved = cli.step_path("ttt-v1.1", 600)

        assert cli.resume_path("step_600", "ttt-v1.1") == saved


class TestDescribe:
    def test_it_records_every_train_field(self):
        described = dict(cli.resolve().describe())

        assert described["num_epochs"] == repr(TrainConfig().num_epochs)

    def test_it_records_the_dataset_and_model(self):
        described = dict(cli.resolve().describe())

        assert (described["dataset"], described["base_model"]) == (
            datasets.DEFAULT,
            "Qwen/Qwen3-0.6B",
        )

    def test_it_records_the_configured_strategy_not_just_its_name(self):
        described = dict(cli.resolve(strategy="hybrid", carry_min_tokens=5000).describe())

        assert "5000" in described["strategy"]

    def test_two_identical_resolutions_describe_identically(self):
        assert cli.resolve(num_epochs=2).describe() == cli.resolve(num_epochs=2).describe()

    def test_a_changed_option_changes_the_description(self):
        assert cli.resolve(num_epochs=2).describe() != cli.resolve(num_epochs=3).describe()
