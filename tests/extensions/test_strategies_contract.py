"""Contract suite for the strategies registry. Every plugin auto-enrols."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from ttt.core.schedule import covers
from ttt.extensions.strategies import STRATEGIES, Strategy, register
from tests.extensions import _builders

CASES = [pytest.param(s, id=name) for name, s in sorted(STRATEGIES.items())]

# The fixed lengths straddle hybrid's slice thresholds, which random draws
# almost never hit.
LENGTHS = [1, 1_999, 2_000, 2_099, 2_100, 6_000, 6_001] + _builders.doc_lengths(
    40, seed=11
)
SOURCES = _builders.sources_for(LENGTHS, seed=12)


def build(strategy, seed):
    return strategy.build(LENGTHS, SOURCES, _builders.FakeRng(seed))


def spans_by_doc(sessions):
    spans = defaultdict(list)
    for session in sessions:
        for item in session.items:
            spans[item.doc_idx].append((item.start, item.end))
    return spans


def test_the_registry_has_every_surviving_strategy():
    assert sorted(STRATEGIES) == ["hybrid", "mixed"]


@pytest.mark.parametrize("strategy", CASES)
def test_every_registered_strategy_satisfies_the_protocol(strategy):
    assert isinstance(strategy, Strategy)


@pytest.mark.parametrize("strategy", CASES)
def test_every_session_is_non_empty(strategy):
    sessions = build(strategy, seed=1)

    assert all(len(session) > 0 for session in sessions)


@pytest.mark.parametrize("strategy", CASES)
def test_every_doc_appears_exactly_once(strategy):
    sessions = build(strategy, seed=1)

    docs = Counter(
        item.doc_idx for session in sessions for item in session.items
    )

    assert sorted(docs) == list(range(len(LENGTHS)))


@pytest.mark.parametrize("strategy", CASES)
def test_every_docs_items_partition_that_doc(strategy):
    spans = spans_by_doc(build(strategy, seed=1))

    assert all(
        covers(tuple(spans[doc_idx]), LENGTHS[doc_idx]) for doc_idx in spans
    )


@pytest.mark.parametrize("strategy", CASES)
def test_items_within_a_session_stay_in_document_order(strategy):
    spans = spans_by_doc(build(strategy, seed=1))

    assert all(value == sorted(value) for value in spans.values())


@pytest.mark.parametrize("strategy", CASES)
def test_count_matches_the_number_of_items_built(strategy):
    sessions = build(strategy, seed=1)

    assert strategy.count(LENGTHS) == sum(len(session) for session in sessions)


@pytest.mark.parametrize("strategy", CASES)
def test_the_same_seed_builds_the_same_schedule(strategy):
    assert build(strategy, seed=3) == build(strategy, seed=3)


@pytest.mark.parametrize("strategy", CASES)
def test_a_different_seed_builds_a_different_schedule(strategy):
    assert build(strategy, seed=3) != build(strategy, seed=4)


@pytest.mark.parametrize("strategy", CASES)
def test_an_empty_pool_builds_no_sessions(strategy):
    assert strategy.build([], [], _builders.FakeRng(1)) == ()


