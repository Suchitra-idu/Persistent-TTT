"""Config resolution: strict, pure, and the same for plugin configs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ttt.core.config import resolve
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig


@dataclass(frozen=True)
class PluginConfig:
    """Stands in for a strategy plugin's colocated config (spec §7)."""

    carry_min_tokens: int = 2100
    slices_min: int = 2
    label: str = "hybrid"
    enabled: bool = True
    rate: float = 0.5


def test_merging_nothing_returns_the_defaults_unchanged():
    defaults = TrainConfig()

    assert resolve.merge(defaults, {}) is defaults


def test_merging_applies_the_override():
    merged = resolve.merge(TrainConfig(), {"num_epochs": 3})

    assert merged.num_epochs == 3


def test_merging_leaves_untouched_fields_at_their_defaults():
    merged = resolve.merge(TrainConfig(), {"num_epochs": 3})

    assert merged.seed == TrainConfig().seed


def test_merging_does_not_mutate_the_defaults():
    defaults = TrainConfig()

    resolve.merge(defaults, {"num_epochs": 9})

    assert defaults.num_epochs == 1


def test_an_unknown_field_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match=r"unknown config field\(s\) \['lr_lora_'\]"):
        resolve.merge(TrainConfig(), {"lr_lora_": 1e-4})


def test_a_wrong_type_is_rejected():
    with pytest.raises(TypeError, match="num_epochs must be int"):
        resolve.merge(TrainConfig(), {"num_epochs": "three"})


def test_a_bool_is_not_accepted_where_an_int_is_declared():
    with pytest.raises(TypeError, match="num_epochs must be int"):
        resolve.merge(TrainConfig(), {"num_epochs": True})


def test_an_int_is_accepted_where_a_float_is_declared():
    merged = resolve.merge(TrainConfig(), {"warmup_ratio": 0})

    assert merged.warmup_ratio == 0


def test_a_plugins_own_config_resolves_the_same_way():
    merged = resolve.merge(PluginConfig(), {"carry_min_tokens": 4096})

    assert merged.carry_min_tokens == 4096


def test_merging_still_runs_the_configs_own_validation():
    with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
        resolve.merge(TrainConfig(), {"grad_accum_steps": 0})


def test_merging_rejects_a_non_dataclass():
    with pytest.raises(TypeError, match="needs a dataclass instance"):
        resolve.merge({"a": 1}, {"a": 2})


def test_unset_sentinels_are_dropped():
    kept = resolve.only_set({"num_epochs": 0, "seed": 7}, unset=(0,))

    assert kept == {"seed": 7}


def test_a_false_flag_survives_a_zero_sentinel():
    """`--session 0` is a real choice; only the -1 sentinel means "unset"."""
    kept = resolve.only_set({"session_training": False, "seed": 0}, unset=(-1,))

    assert kept == {"session_training": False, "seed": 0}


def test_none_is_the_default_sentinel():
    assert resolve.only_set({"a": None, "b": 1}) == {"b": 1}


def test_several_sentinels_can_be_dropped_at_once():
    kept = resolve.only_set({"a": 0, "b": "", "c": 5}, unset=(0, ""))

    assert kept == {"c": 5}


def test_a_described_config_is_sorted_by_field_name():
    described = resolve.describe(PluginConfig())

    assert [name for name, _ in described] == sorted(name for name, _ in described)


def test_describing_covers_every_field():
    described = resolve.describe(PluginConfig())

    assert {name for name, _ in described} == {
        "carry_min_tokens",
        "slices_min",
        "label",
        "enabled",
        "rate",
    }


def test_two_identical_configs_describe_identically():
    assert resolve.describe(TTTConfig()) == resolve.describe(TTTConfig())


def test_a_changed_field_changes_the_description():
    changed = resolve.merge(TTTConfig(), {"chunk_size": 64})

    assert resolve.describe(changed) != resolve.describe(TTTConfig())
