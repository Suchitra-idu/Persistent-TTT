"""Contract suite for the RULER_TASKS registry. Every plugin auto-enrols."""

from __future__ import annotations

import itertools

import pytest

from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.numpy_rng import NumpyRng
from ttt.core.ruler_types import QaPair, RulerContext
from ttt.extensions.ruler_tasks import RULER_TASKS

SEQ_LEN = 4000

CASES = [pytest.param(task, id=name) for name, task in sorted(RULER_TASKS.items())]


def _context() -> RulerContext:
    words = tuple("".join(p) for p in itertools.product("abcdefghij", repeat=4)) * 3
    qa_pairs = tuple(
        QaPair(
            query=f"What is thing {i}?",
            answers=(f"answer{i}",),
            context=f"Document about thing {i}.",
        )
        for i in range(20)
    )
    return RulerContext(haystack_words=words, qa_pairs=qa_pairs)


def _build(task):
    return task.build(NumpyRng(0), FakeTokenizer(), SEQ_LEN, _context())


@pytest.mark.parametrize("task", CASES)
def test_build_respects_the_token_budget(task):
    example = _build(task)

    assert len(FakeTokenizer().encode(example.prompt)) <= SEQ_LEN


@pytest.mark.parametrize("task", CASES)
def test_build_produces_at_least_one_target(task):
    assert _build(task).targets


@pytest.mark.parametrize("task", CASES)
def test_score_is_perfect_against_its_own_targets(task):
    example = _build(task)

    assert task.score(" ".join(example.targets), example) == 1.0


@pytest.mark.parametrize("task", CASES)
def test_score_is_imperfect_against_an_unrelated_prediction(task):
    example = _build(task)

    assert task.score("completely unrelated gibberish zzz", example) < 1.0


def test_niah_registers_its_four_variants():
    expected = {"niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery"}

    assert expected <= set(RULER_TASKS)


def test_every_task_names_itself_in_the_registry():
    for name, task in RULER_TASKS.items():
        assert task.name == name
