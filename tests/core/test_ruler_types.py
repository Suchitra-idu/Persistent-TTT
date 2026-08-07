from __future__ import annotations

import pytest

from ttt.core.ruler_types import QaPair, RulerExample, RulerResult


def _example(**overrides) -> RulerExample:
    defaults = dict(
        task="niah_single", length_bucket=4096, prompt="p", answer_prefix=" Answer:",
        targets=("t",),
    )
    return RulerExample(**{**defaults, **overrides})


def test_an_example_needs_a_task_name():
    with pytest.raises(ValueError, match="task name"):
        _example(task="")


def test_an_example_needs_a_nonempty_prompt():
    with pytest.raises(ValueError, match="prompt"):
        _example(prompt="")


def test_an_example_needs_a_nonempty_answer_prefix():
    with pytest.raises(ValueError, match="answer_prefix"):
        _example(answer_prefix="")


def test_an_example_needs_at_least_one_target():
    with pytest.raises(ValueError, match="target"):
        _example(targets=())


def test_a_qa_pair_needs_at_least_one_answer():
    with pytest.raises(ValueError, match="answer"):
        QaPair(query="q", answers=(), context="c")


def test_a_result_score_must_be_in_unit_range():
    with pytest.raises(ValueError, match="score"):
        RulerResult(
            task="vt", length_bucket=4096, regime="fresh", score=1.5, n_new_tokens=1
        )
