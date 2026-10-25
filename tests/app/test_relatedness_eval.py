from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import relatedness_eval
from ttt.app.relatedness_eval import (
    LEVELS,
    OTHER_SOURCE,
    RANDOM,
    RELATED,
    RESET,
    SELF,
    Article,
    Case,
    RelatedRow,
)
from ttt.core.metrics import ALL_SOURCES

TARGET = tuple(range(100, 108))


def case(index=0):
    primes = {level: tuple(range(10 * (i + 1), 10 * (i + 1) + 4)) for i, level in enumerate(LEVELS)}
    return Case(
        index=index, target_title="T", target=TARGET, primes=primes,
        prime_titles={level: level for level in LEVELS},
    )


def measure(n_cases=2):
    wiring = _builders.Wiring()
    rows = relatedness_eval.measure(
        cases=[case(i) for i in range(n_cases)],
        compute=wiring.compute, fast_weights=wiring.fast_weights,
    )
    return rows, wiring


def test_each_case_is_scored_reset_and_after_every_prime():
    rows, _ = measure(n_cases=2)

    assert [r.arm for r in rows if r.doc_idx == 0] == [RESET, *LEVELS]
    assert len(rows) == 2 * (1 + len(LEVELS))


def test_every_arm_scores_exactly_the_target_tokens():
    _, wiring = measure(n_cases=1)

    scored = wiring.compute.eval_forwards[0::2][:1] + wiring.compute.eval_forwards[2::2]
    assert all(ids == TARGET for ids in scored)


def test_the_carry_is_reset_before_every_arm():
    _, wiring = measure(n_cases=1)

    assert wiring.fast_weights.events.count("reset_carry") >= 1 + len(LEVELS)


def test_each_prime_is_read_and_its_carry_promoted_before_the_target():
    _, wiring = measure(n_cases=1)

    primes = wiring.compute.eval_forwards[1::2]
    assert primes == [case().primes[level] for level in LEVELS]
    assert wiring.fast_weights.events.count("advance_carry") == len(LEVELS)


@pytest.mark.parametrize(
    ("title", "usable"),
    [
        ("Temple of the Tooth", True),
        ("Kandy", False),
        ("Kandy (city)", False),
        ("List of temples", False),
        ("A Very Long Title Of Many Words", False),
    ],
)
def test_only_multi_word_plain_titles_are_matched(title, usable):
    assert relatedness_eval.usable_title(title) is usable


def test_mentions_count_whole_word_title_occurrences():
    titles = {"Temple of the Tooth": 3, "New York": 5}

    found = relatedness_eval.mentions(
        "The Temple of the Tooth is old. Pilgrims visit the Temple of the Tooth daily.", titles
    )

    assert found == {3: 2}


def _articles():
    return [
        Article("Kandy City", "Kandy City hosts the Temple of the Tooth. The Temple of the Tooth is sacred."),
        Article("Temple of the Tooth", "A Buddhist temple in the hill country."),
        Article("Ocean Currents", "Water moves in large circulating systems."),
        Article("Lonely Town", "Mentions nothing at all."),
    ]


def test_the_related_prime_is_the_most_mentioned_article():
    picks = relatedness_eval.pick(
        _articles(), targets=[0], primes=[0, 1, 2, 3], n=1, rng=NumpyRng(0)
    )

    assert [(p.target, p.related, p.mentions) for p in picks] == [(0, 1, 2)]


def test_the_unrelated_prime_is_neither_target_related_nor_mentioned():
    picks = relatedness_eval.pick(
        _articles(), targets=[0], primes=[0, 1, 2, 3], n=1, rng=NumpyRng(0)
    )

    assert picks[0].unrelated in (2, 3)


def test_a_target_with_too_few_mentions_is_skipped():
    picks = relatedness_eval.pick(
        _articles(), targets=[0], primes=[0, 1, 2, 3], n=1, rng=NumpyRng(0), min_mentions=3
    )

    assert picks == []


def _row(arm, doc, ppl):
    return RelatedRow(arm=arm, source="Wikipedia", doc_idx=doc, target_title="T",
                      prime_title="", n_tokens=10, ppl=ppl)


def test_the_gain_is_reset_minus_primed_and_contrasts_compare_levels():
    ppl = {RESET: 10.0, SELF: 8.0, RELATED: 9.0, RANDOM: 9.5, OTHER_SOURCE: 10.5}
    rows = [_row(arm, doc, value) for doc in range(3) for arm, value in ppl.items()]

    pooled = {
        e.measure: e.interval.point
        for e in relatedness_eval.estimates(rows, rng=NumpyRng(0), n_boot=10)
        if e.source == ALL_SOURCES
    }

    assert pooled["gain_self"] == pytest.approx(2.0)
    assert pooled["gain_other_source"] == pytest.approx(-0.5)
    assert pooled["related_vs_random"] == pytest.approx(0.5)
    assert pooled["self_vs_related"] == pytest.approx(1.0)
