from __future__ import annotations

import pytest

from ttt.core.config.ruler import BASE, RulerConfig, parse_ruler_flags
from ttt.core.types import COLD_CARRY, FRESH


class TestDefaults:
    def test_no_flags_resolves_the_default_config(self):
        assert parse_ruler_flags("") == RulerConfig()

    def test_the_default_regimes_are_both_arms(self):
        assert RulerConfig().regimes == (FRESH, COLD_CARRY)

    def test_the_default_prompt_style_is_base(self):
        assert RulerConfig().prompt_style == BASE


class TestParsing:
    def test_a_list_field_splits_on_colons(self):
        cfg = parse_ruler_flags("tasks=niah_single:vt")

        assert cfg.tasks == ("niah_single", "vt")

    def test_context_lengths_parse_as_ints(self):
        cfg = parse_ruler_flags("context_lengths=4096:8192")

        assert cfg.context_lengths == (4096, 8192)

    def test_a_scalar_field_parses_as_an_int(self):
        assert parse_ruler_flags("num_samples=5").num_samples == 5

    def test_several_fields_combine(self):
        cfg = parse_ruler_flags("tasks=vt,num_samples=3,seed=7")

        assert (cfg.tasks, cfg.num_samples, cfg.seed) == (("vt",), 3, 7)

    def test_an_unknown_flag_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="unknown ruler flag 'bogus'"):
            parse_ruler_flags("bogus=1")

    def test_a_str_field_parses_as_is(self):
        assert parse_ruler_flags("prompt_style=instruct").prompt_style == "instruct"


class TestValidation:
    def test_it_rejects_an_empty_task_list(self):
        with pytest.raises(ValueError, match="at least one task"):
            RulerConfig(tasks=())

    def test_it_rejects_a_non_positive_context_length(self):
        with pytest.raises(ValueError, match="context_lengths"):
            RulerConfig(context_lengths=(0,))

    def test_it_rejects_an_unknown_regime(self):
        with pytest.raises(ValueError, match="regimes"):
            RulerConfig(regimes=("bogus",))

    def test_it_rejects_an_unknown_prompt_style(self):
        with pytest.raises(ValueError, match="prompt_style"):
            RulerConfig(prompt_style="bogus")

    def test_the_prepared_example_key_is_stable_for_the_same_config(self):
        cfg = RulerConfig(num_samples=5, seed=1)

        assert cfg.key("vt", 4096) == cfg.key("vt", 4096)

    def test_the_prepared_example_key_changes_with_num_samples(self):
        a = RulerConfig(num_samples=5)
        b = RulerConfig(num_samples=6)

        assert a.key("vt", 4096) != b.key("vt", 4096)
