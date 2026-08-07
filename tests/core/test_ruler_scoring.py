from __future__ import annotations

import pytest

from ttt.core.ruler_scoring import qa_score, recall_score


class TestRecallScore:
    def test_every_target_present_scores_one(self):
        assert recall_score("the cat sat on the mat", ["cat", "mat"]) == 1.0

    def test_no_target_present_scores_zero(self):
        assert recall_score("the dog ran", ["cat", "mat"]) == 0.0

    def test_a_partial_match_is_the_hit_fraction(self):
        assert recall_score("the cat sat", ["cat", "mat"]) == 0.5

    def test_matching_is_case_insensitive(self):
        assert recall_score("The CAT sat", ["cat"]) == 1.0

    def test_it_rejects_an_empty_target_list(self):
        with pytest.raises(ValueError, match="at least one target"):
            recall_score("anything", [])


class TestQaScore:
    def test_an_exact_match_scores_one(self):
        assert qa_score("Paris", ["Paris"]) == 1.0

    def test_it_takes_the_best_of_several_targets(self):
        assert qa_score("Paris", ["London", "Paris"]) == 1.0

    def test_disjoint_tokens_score_zero(self):
        assert qa_score("Paris", ["Tokyo"]) == 0.0

    def test_partial_token_overlap_scores_between_zero_and_one(self):
        score = qa_score("the capital is Paris", ["Paris"])

        assert 0.0 < score < 1.0

    def test_punctuation_and_case_do_not_affect_the_score(self):
        assert qa_score("PARIS!", ["paris"]) == 1.0

    def test_it_rejects_an_empty_target_list(self):
        with pytest.raises(ValueError, match="at least one target"):
            qa_score("anything", [])
