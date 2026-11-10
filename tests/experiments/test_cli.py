from __future__ import annotations

import pytest

from ttt import cli
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig


class TestDefaults:
    def test_no_arguments_resolves_the_default_config(self):
        assert cli.resolve().train == TrainConfig()


    def test_layer_indices_stay_unknown_until_a_model_is_loaded(self):
        assert cli.resolve().ttt.layer_indices is None


class TestTrainOverrides:
    def test_a_named_field_reaches_the_config(self):
        assert cli.resolve(num_epochs=4).train.num_epochs == 4


    def test_a_typo_is_rejected_by_name(self):
        with pytest.raises(ValueError, match=r"unknown option\(s\) \['num_epoch'\]"):
            cli.resolve(num_epoch=4)

    def test_a_wrongly_typed_value_is_rejected(self):
        with pytest.raises(TypeError, match="num_epochs must be int"):
            cli.resolve(num_epochs="four")

    def test_an_invalid_combination_is_rejected_by_the_config(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            cli.resolve(max_seq_len=10, min_doc_tokens=2048)


class TestStrategy:
    def test_a_named_strategy_is_selected(self):
        assert cli.resolve(strategy="mixed").strategy.name == "mixed"


    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(KeyError, match="unknown strategy"):
            cli.resolve(strategy="multi")


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
        with pytest.raises(ValueError, match="docs_per_session"):
            cli.resolve(strategy="hybrid", docs_per_session=3)

    def test_an_invalid_knob_combination_is_rejected_by_the_plugin(self):
        with pytest.raises(ValueError, match="carry_min_tokens"):
            cli.resolve(strategy="hybrid", carry_min_tokens=10, slice_min_tokens=1000)


class TestSourceWeights:
    def test_the_spec_default_preset_applies(self):
        assert "RedPajamaC4" in cli.resolve().weights


class TestMechanismKnobs:

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


class TestFromFlags:


    def test_an_empty_string_means_unset(self):
        assert cli.from_flags(run_name="").train.run_name == TrainConfig().run_name


    def test_an_unset_document_limit_is_the_papers_4000(self):
        assert cli.from_flags(limit_docs=0).limit_docs == cli.DEFAULT_LIMIT_DOCS


    def test_a_zero_in_the_flags_is_taken_as_given(self):
        assert cli.from_flags(**cli.parse_flags("seed=0")).train.seed == 0

    def test_a_misspelt_boolean_is_rejected(self):
        with pytest.raises(ValueError, match="not a boolean"):
            cli.parse_flags("carry_gate=ture")

    def test_a_retired_option_is_accepted_at_its_recorded_value(self):
        assert cli.from_flags(**cli.parse_flags("state_aware_gate=false")).ttt == TTTConfig()

    def test_a_retired_option_at_any_other_value_is_rejected(self):
        with pytest.raises(ValueError, match="state_aware_gate was removed"):
            cli.from_flags(**cli.parse_flags("state_aware_gate=true"))

    def test_a_typo_is_still_rejected(self):
        with pytest.raises(ValueError, match="unknown option"):
            cli.from_flags(grad_accume=8)


class TestCheckpointPaths:

    def test_a_bare_step_resumes_within_the_same_run(self):
        assert cli.resume_path("step_600", "ttt-v1.1") == "ttt-v1.1/step_600"

    def test_a_qualified_step_resumes_across_runs(self):
        assert cli.resume_path("other/step_600", "ttt-v1.1") == "other/step_600"


    def test_a_saved_step_is_where_a_resume_looks_for_it(self):
        saved = cli.step_path("ttt-v1.1", 600)

        assert cli.resume_path("step_600", "ttt-v1.1") == saved


class TestParseFlags:

    def test_an_integer_field_arrives_as_an_integer(self):
        assert cli.parse_flags("num_epochs=3") == {"num_epochs": 3}

    def test_a_float_field_arrives_as_a_float(self):
        assert cli.parse_flags("lr_wdown=0.0005") == {"lr_wdown": 0.0005}


    @pytest.mark.parametrize("text", ["true", "1", "yes", "on", "TRUE"])
    def test_a_boolean_field_accepts_the_usual_spellings(self, text):
        assert cli.parse_flags(f"wandb_enabled={text}") == {"wandb_enabled": True}

    def test_a_boolean_field_is_false_when_it_says_so(self):
        """`bool("false")` is True, so a string would silently invert this."""
        assert cli.parse_flags("wandb_enabled=false") == {"wandb_enabled": False}

    def test_several_flags_separated_by_commas(self):
        assert cli.parse_flags("num_epochs=2,strategy=hybrid") == {
            "num_epochs": 2,
            "strategy": "hybrid",
        }


    def test_a_dataset_name_is_not_mistaken_for_a_number(self):
        assert cli.parse_flags("dataset=slimpajama-6b") == {"dataset": "slimpajama-6b"}

    def test_a_pair_without_a_value_is_rejected(self):
        with pytest.raises(ValueError, match="expected name=value"):
            cli.parse_flags("num_epochs")


    def test_a_typo_is_rejected_before_anything_reaches_a_gpu(self):
        with pytest.raises(ValueError, match="unknown option"):
            cli.from_flags(**cli.parse_flags("num_epoch=7"))
