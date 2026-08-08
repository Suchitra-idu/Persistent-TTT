"""TRAIN_LANGS / EVAL_LANGS — the two specs the generic contract suite
(test_datasets_contract.py) doesn't check: each admits exactly its own
language list, whatever that list currently is (docs/experiments-map.md)."""

from __future__ import annotations

from ttt.core.config.lang_transfer import (
    EVAL_LANGUAGES,
    TRAIN_HOLDOUT_LAST_N,
    TRAIN_LANGUAGES,
    combined_dir,
)
from ttt.extensions.datasets.lang_transfer import EVAL_LANGS, TRAIN_LANGS


def test_train_only_admits_the_training_languages():
    assert set(TRAIN_LANGS.include_sources) == {code for code, _ in TRAIN_LANGUAGES}


def test_eval_admits_every_eval_language_including_the_overlap():
    assert set(EVAL_LANGS.include_sources) == {code for code, _ in EVAL_LANGUAGES}


def test_a_non_training_language_is_not_admitted_for_training():
    """Eastern Yiddish is in CANDIDATE_LANGUAGES, not TRAIN_LANGUAGES — the
    `include_sources` filter, not the two lists happening to differ, is what
    keeps a non-training language out; still true now that TRAIN_LANGUAGES
    and EVAL_LANGUAGES are the same ten languages."""
    assert not TRAIN_LANGS.keeps("yi")


def test_train_langs_defaults_to_the_balanced_preset():
    assert TRAIN_LANGS.default_source_preset == "lang-transfer-balanced"


def test_the_two_specs_point_at_different_directories():
    assert TRAIN_LANGS.source != EVAL_LANGS.source


def test_eval_langs_holdout_boundary_admits_its_whole_pool():
    assert EVAL_LANGS.holdout_boundary(9_000) == 0


def test_both_specs_point_at_the_combined_directory_not_the_raw_fetch_one():
    assert TRAIN_LANGS.source.endswith(combined_dir("train"))
    assert EVAL_LANGS.source.endswith(combined_dir("eval"))


def test_train_langs_holdout_matches_the_shared_constant():
    """Sized off DEFAULT_TARGET_ROWS (core.config.lang_transfer) rather than
    a bare number, so it can't silently go stale if that default changes."""
    assert TRAIN_LANGS.holdout_last_n == TRAIN_HOLDOUT_LAST_N
