"""OLD-vs-NEW CLI resolution (Phase 5's done-when).

The two TrainConfigs are not the same dataclass — D4 moved the strategy knobs
onto the plugins and D11 deleted the fields four cut modes needed — so parity is
asserted field by field over what survives, plus the rules that decide them.

`TTT_DATASET` is pinned before the old module loads: it reads the env at import
and would otherwise default to arxiv, which D9 dropped.
"""

from __future__ import annotations

import pytest
import train_modal
import ttt_config

from ttt import cli

# Named rather than read off the old tree's import-time singleton (PLAN §6.2).
OLD_SPEC = ttt_config.get_dataset_spec("slimpajama-6b")

pytestmark = pytest.mark.parity

# The old surface: 0 for an int knob, "" for a string, -1 for the session flag.
OLD_UNSET = dict(
    num_epochs=0,
    grad_accum=0,
    session=train_modal._UNSET,
    mode="",
    min_doc_tokens=0,
    hybrid_carry_min=0,
    hybrid_slice_min=0,
    hybrid_slices_min=0,
    hybrid_slices_max=0,
    eval_n_papers=0,
    eval_n_papers_per_source=0,
    eval_min_tokens=0,
    eval_every=0,
    source_preset="",
)

# Old name -> new name, for the fields both sides still have. `eval_n_papers`
# became `eval_n_docs` and the rest of the paper/doc rename is D10.
SHARED_FIELDS = {
    "num_epochs": "num_epochs",
    "grad_accum_steps": "grad_accum_steps",
    "min_doc_tokens": "min_doc_tokens",
    "max_seq_len": "max_seq_len",
    "eval_every": "eval_every",
    "eval_n_papers": "eval_n_docs",
    "eval_n_papers_per_source": "eval_n_docs_per_source",
    "eval_n_slices": "eval_n_slices",
    "eval_min_tokens": "eval_min_tokens",
    "eval_holdout_seed": "eval_holdout_seed",
    "seed": "seed",
    "log_every": "log_every",
    "save_every": "save_every",
    "run_name": "run_name",
    "lr_lora": "lr_lora",
    "lr_wdown": "lr_wdown",
    "lr_new_modules": "lr_new_modules",
    "weight_decay_full": "weight_decay_full",
    "weight_decay_lora": "weight_decay_lora",
    "warmup_ratio": "warmup_ratio",
    "warmup_min_steps": "warmup_min_steps",
    "max_grad_norm": "max_grad_norm",
    "lora_r": "lora_r",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "micro_batch_size": "micro_batch_size",
    "wandb_enabled": "wandb_enabled",
    "wandb_project": "wandb_project",
}

# `--mode` chose between four session regimes; two survive as strategies (D11).
MODES = {"hybrid": "hybrid", "everlasting": "everlasting"}

CASES = [
    pytest.param({}, {}, id="defaults"),
    pytest.param({"mode": "hybrid"}, {"strategy": "hybrid"}, id="hybrid"),
    pytest.param(
        {"mode": "everlasting"}, {"strategy": "everlasting"}, id="everlasting"
    ),
    pytest.param(
        {"num_epochs": 3, "grad_accum": 8},
        {"num_epochs": 3, "grad_accum": 8},
        id="epochs-and-accum",
    ),
    pytest.param(
        {"mode": "everlasting", "eval_every": 10},
        {"strategy": "everlasting", "eval_every": 10},
        id="explicit-eval-cadence",
    ),
    pytest.param(
        {"min_doc_tokens": 4096, "eval_min_tokens": 3000},
        {"min_doc_tokens": 4096, "eval_min_tokens": 3000},
        id="token-thresholds",
    ),
    pytest.param(
        {"eval_n_papers": 7, "eval_n_papers_per_source": 2},
        {"eval_n_docs": 7, "eval_n_docs_per_source": 2},
        id="eval-sample-sizes",
    ),
    pytest.param(
        {"source_preset": "slim-paper"}, {"source_preset": "slim-paper"}, id="preset"
    ),
    pytest.param(
        {"source_preset": "none"}, {"source_preset": "none"}, id="preset-off"
    ),
]


def old(**overrides):
    return train_modal._apply_cli_overrides(**{**OLD_UNSET, **overrides})


def new(**overrides):
    return cli.from_flags(**overrides)


@pytest.mark.parametrize(("old_args", "new_args"), CASES)
@pytest.mark.parametrize(("old_field", "new_field"), sorted(SHARED_FIELDS.items()))
def test_every_shared_field_resolves_the_same(old_args, new_args, old_field, new_field):
    assert getattr(old(**old_args), old_field) == getattr(
        new(**new_args).train, new_field
    )


@pytest.mark.parametrize(("old_mode", "new_strategy"), sorted(MODES.items()))
def test_each_surviving_mode_selects_the_strategy_that_replaced_it(
    old_mode, new_strategy
):
    assert new(strategy=new_strategy).strategy.name == new_strategy


@pytest.mark.parametrize(("old_mode", "new_strategy"), sorted(MODES.items()))
def test_each_surviving_mode_resolves_the_same_eval_cadence(old_mode, new_strategy):
    assert old(mode=old_mode).eval_every == new(strategy=new_strategy).train.eval_every


def test_a_source_scoped_run_still_defaults_to_the_tighter_eval_cadence():
    assert old(mode="everlasting").eval_every == 25


def test_an_explicit_cadence_still_beats_the_strategy_default():
    old_cfg = old(mode="everlasting", eval_every=10)

    assert (old_cfg.eval_every, new(strategy="everlasting", eval_every=10).train.eval_every) == (
        10,
        10,
    )


def test_a_source_scoped_run_still_trains_with_session_state():
    assert (
        old(mode="everlasting").session_training
        is new(strategy="everlasting").train.session_training
        is True
    )


def test_the_session_ablation_still_overrides_the_strategy():
    assert (
        old(mode="everlasting", session=0).session_training
        is new(strategy="everlasting", session=0).train.session_training
        is False
    )


@pytest.mark.parametrize("preset", ["slim-paper", "slim-research"])
def test_a_preset_resolves_to_the_same_weights(preset):
    assert dict(new(source_preset=preset).weights) == dict(
        ttt_config.get_source_preset(preset)
    )


def test_the_spec_default_preset_is_the_same_on_both_sides():
    assert dict(new().weights) == dict(
        ttt_config.get_source_preset(OLD_SPEC.default_source_preset)
    )


def test_only_sources_still_builds_equal_weights_over_the_picks():
    picks = "RedPajamaBook,RedPajamaArXiv"
    # The old path registered the ephemeral preset into a module-level dict; the
    # new one returns it, which is why D5 could delete the registry.
    registered = old(only_sources=picks).source_preset

    assert dict(new(only_sources=picks).weights) == dict(
        ttt_config.SOURCE_PRESETS[registered]
    )


def test_the_session_training_default_changed_deliberately():
    """D4: session_training is the A1 ablation, not a mode, so it defaults on."""
    assert (old().session_training, new().train.session_training) == (False, True)


def test_the_dataset_spec_agrees_on_what_is_held_out():
    assert new().spec.holdout_last_n == OLD_SPEC.holdout_last_n


def test_the_dataset_spec_agrees_on_which_sources_are_included():
    assert new().spec.include_sources == tuple(OLD_SPEC.include_sources)


def test_the_hybrid_knobs_moved_to_the_plugin_but_kept_their_values():
    old_cfg = old(mode="hybrid")
    strategy = new(strategy="hybrid").strategy

    assert (
        strategy.carry_min_tokens,
        strategy.slice_min_tokens,
        strategy.slices_min,
        strategy.slices_max,
    ) == (
        old_cfg.hybrid_carry_min_tokens,
        old_cfg.hybrid_slice_min_tokens,
        old_cfg.hybrid_slices_min,
        old_cfg.hybrid_slices_max,
    )


def test_the_hybrid_knobs_still_flow_through_from_the_command_line():
    old_cfg = old(
        mode="hybrid",
        hybrid_carry_min=5000,
        hybrid_slice_min=1200,
        hybrid_slices_min=3,
        hybrid_slices_max=8,
    )
    strategy = new(
        strategy="hybrid",
        carry_min_tokens=5000,
        slice_min_tokens=1200,
        slices_min=3,
        slices_max=8,
    ).strategy

    assert (
        strategy.carry_min_tokens,
        strategy.slice_min_tokens,
        strategy.slices_min,
        strategy.slices_max,
    ) == (
        old_cfg.hybrid_carry_min_tokens,
        old_cfg.hybrid_slice_min_tokens,
        old_cfg.hybrid_slices_min,
        old_cfg.hybrid_slices_max,
    )


def test_the_base_model_identity_is_unchanged():
    assert new().base_model == ttt_config.BASE_MODEL


def test_a_cut_mode_is_rejected_rather_than_silently_accepted():
    with pytest.raises(KeyError, match="unknown strategy"):
        new(strategy="single")


def test_the_old_path_still_accepts_the_modes_this_rebuild_dropped():
    assert old(mode="single").single_paper_sessions is True
