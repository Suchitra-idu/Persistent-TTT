"""The language-transfer eval's language lists — pure data, no I/O."""

from __future__ import annotations

import pytest

from ttt.core.config import lang_transfer


def test_train_and_eval_are_currently_the_same_ten_languages():
    """Direct training on a weak language beat transfer to an untrained one
    (docs/experiments-map.md); EVAL_LANGUAGES stays its own name so a future
    run can point it elsewhere without restructuring anything."""
    assert lang_transfer.EVAL_LANGUAGES == lang_transfer.TRAIN_LANGUAGES
    assert len(lang_transfer.OVERLAP_LANGUAGES) == 10


def test_every_overlap_language_is_in_both_lists():
    train = {code for code, _ in lang_transfer.TRAIN_LANGUAGES}
    eval_ = {code for code, _ in lang_transfer.EVAL_LANGUAGES}

    assert set(lang_transfer.OVERLAP_LANGUAGES) == train & eval_


def test_language_codes_are_never_repeated_within_a_list():
    for languages in (lang_transfer.TRAIN_LANGUAGES, lang_transfer.EVAL_LANGUAGES):
        codes = [code for code, _ in languages]
        assert len(codes) == len(set(codes))


@pytest.mark.parametrize("role", ["train", "eval"])
def test_corpus_dir_is_role_scoped(role):
    assert lang_transfer.corpus_dir(role) == f"lang_corpus/{role}"


def test_corpus_dir_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="role must be"):
        lang_transfer.corpus_dir("both")


def test_languages_for_train_matches_the_constant():
    assert lang_transfer.languages_for("train") == lang_transfer.TRAIN_LANGUAGES


def test_languages_for_eval_matches_the_constant():
    assert lang_transfer.languages_for("eval") == lang_transfer.EVAL_LANGUAGES


def test_languages_for_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="role must be"):
        lang_transfer.languages_for("both")


def test_train_holdout_last_n_is_derived_not_hardcoded():
    expected = round(
        lang_transfer.TRAIN_HOLDOUT_FRACTION
        * lang_transfer.DEFAULT_TARGET_ROWS
        * len(lang_transfer.TRAIN_LANGUAGES)
    )

    assert lang_transfer.TRAIN_HOLDOUT_LAST_N == expected


def test_train_holdout_last_n_stays_a_small_fraction_of_the_train_pool():
    raw_pool = lang_transfer.DEFAULT_TARGET_ROWS * len(lang_transfer.TRAIN_LANGUAGES)

    assert lang_transfer.TRAIN_HOLDOUT_LAST_N < 0.2 * raw_pool


def test_candidate_languages_are_never_repeated():
    codes = [code for code, _ in lang_transfer.CANDIDATE_LANGUAGES]

    assert len(codes) == len(set(codes))


def test_candidate_languages_do_not_overlap_the_already_verified_ones():
    candidates = {code for code, _ in lang_transfer.CANDIDATE_LANGUAGES}
    verified = {code for code, _ in lang_transfer.TRAIN_LANGUAGES} | {
        code for code, _ in lang_transfer.EVAL_LANGUAGES
    }

    assert not candidates & verified
