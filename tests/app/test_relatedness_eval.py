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


def _two_targets_one_related():
    return [
        Article("Kandy City", "Kandy City hosts the Temple of the Tooth. The Temple of the Tooth is sacred."),
        Article("Hill Country", "The Temple of the Tooth draws pilgrims. The Temple of the Tooth is in Kandy."),
        Article("Temple of the Tooth", "A Buddhist temple in the hill country."),
        Article("Ocean Currents", "Water moves in large circulating systems."),
        Article("Lonely Town", "Mentions nothing at all."),
    ]


def test_a_related_article_may_serve_several_targets_by_default():
    picks = relatedness_eval.pick(
        _two_targets_one_related(), targets=[0, 1], primes=[0, 1, 2, 3, 4], n=2, rng=NumpyRng(0)
    )

    assert [p.related for p in picks] == [2, 2]


def test_unique_related_gives_each_related_article_one_target():
    picks = relatedness_eval.pick(
        _two_targets_one_related(), targets=[0, 1], primes=[0, 1, 2, 3, 4], n=2,
        rng=NumpyRng(0), unique_related=True,
    )

    assert [p.related for p in picks] == [2]


def _carry(*values):
    import torch

    return relatedness_eval.Carry(deltas={layer: torch.tensor([[v]]) for layer, v in zip(_builders.LAYERS, values)})


def test_the_generic_arm_scores_the_target_with_the_generic_carry_installed():
    wiring = _builders.Wiring()

    rows = relatedness_eval.measure(
        cases=[case()], compute=wiring.compute, fast_weights=wiring.fast_weights,
        generic=_carry(*[1.0] * len(_builders.LAYERS)),
    )

    assert rows[-1].arm == relatedness_eval.GENERIC
    assert wiring.fast_weights.events[-3:-1] == ["reset_carry", "install(carry)"]
    assert wiring.compute.eval_forwards[-1] == TARGET


def test_the_in_context_arm_scores_only_the_target_after_the_related_prime():
    wiring = _builders.Wiring()

    rows = relatedness_eval.measure(
        cases=[case()], compute=wiring.compute, fast_weights=wiring.fast_weights, in_context=True,
    )

    prime = case().primes[RELATED]
    assert rows[-1].arm == relatedness_eval.IN_CONTEXT
    assert wiring.compute.eval_forwards[-1] == prime + TARGET
    assert wiring.compute.eval_score_from[-1] == len(prime)
    assert rows[-1].n_tokens == len(TARGET)


def chained_case(n_chain):
    return relatedness_eval.Case(
        index=0, target_title="T", target=TARGET, primes=case().primes,
        prime_titles=case().prime_titles,
        chain=tuple((900 + i,) * 4 for i in range(n_chain)),
    )


def test_chain_arms_score_the_target_after_each_requested_length():
    wiring = _builders.Wiring()

    rows = relatedness_eval.measure(
        cases=[chained_case(5)], compute=wiring.compute, fast_weights=wiring.fast_weights,
        chain_lengths=(1, 2, 5),
    )

    assert [r.arm for r in rows if r.arm.startswith("chain_")] == ["chain_1", "chain_2", "chain_5"]


def test_scoring_the_target_mid_chain_does_not_feed_the_chain():
    wiring = _builders.Wiring()

    relatedness_eval.measure(
        cases=[chained_case(2)], compute=wiring.compute, fast_weights=wiring.fast_weights,
        chain_lengths=(1, 2),
    )

    after_first = wiring.fast_weights.events[-7:]
    assert after_first[:4] == ["advance_carry", "reset_carry", "install(carry)", "advance_carry"]


def test_a_chain_shorter_than_the_longest_length_is_an_error():
    wiring = _builders.Wiring()

    with pytest.raises(ValueError, match="chain articles"):
        relatedness_eval.measure(
            cases=[chained_case(2)], compute=wiring.compute, fast_weights=wiring.fast_weights,
            chain_lengths=(5,),
        )


def test_the_generic_carry_is_the_mean_rescaled_to_the_median_norm():
    import torch

    carries = [
        relatedness_eval.Carry(deltas={1: torch.tensor([[3.0, 0.0]])}),
        relatedness_eval.Carry(deltas={1: torch.tensor([[0.0, 4.0]])}),
        relatedness_eval.Carry(deltas={1: torch.tensor([[5.0, 0.0]])}),
    ]

    generic = relatedness_eval.generic_carry(carries).deltas[1]

    assert float(generic.norm()) == pytest.approx(4.0)
    assert generic[0, 0] / generic[0, 1] == pytest.approx(2.0)


def test_text_sharing_enough_opening_word_sequences_overlaps():
    opening = " ".join(f"word{i}" for i in range(40))
    seen = relatedness_eval.shingles(opening)

    assert relatedness_eval.overlaps(opening + " later edits", seen)
    assert not relatedness_eval.overlaps("Entirely different text about other matters here.", seen)


def test_headline_contrasts_appear_with_adjusted_intervals_only_when_their_arms_exist():
    ppl = {RESET: 10.0, SELF: 8.0, RELATED: 9.0, RANDOM: 9.5, OTHER_SOURCE: 10.5,
           relatedness_eval.GENERIC: 9.6, relatedness_eval.IN_CONTEXT: 7.0}
    rows = [_row(arm, doc, value + 0.1 * doc * (arm == RELATED)) for doc in range(6) for arm, value in ppl.items()]

    pooled = {e.measure: e for e in relatedness_eval.estimates(rows, rng=NumpyRng(0), n_boot=200)
              if e.source == ALL_SOURCES}
    without = {e.measure for e in relatedness_eval.estimates(
        [r for r in rows if r.arm != relatedness_eval.IN_CONTEXT], rng=NumpyRng(0), n_boot=20)}

    adjusted, plain = pooled["related_vs_generic_adjusted"].interval, pooled["related_vs_generic"].interval
    assert adjusted.high - adjusted.low >= plain.high - plain.low
    assert "carry_vs_in_context" in pooled and "carry_vs_in_context" not in without


def test_a_model_compared_with_itself_has_no_gain_change():
    ppl = {RESET: 10.0, OTHER_SOURCE: 10.5}
    rows = [_row(arm, doc, value + doc) for doc in range(4) for arm, value in ppl.items()]

    (change,) = relatedness_eval.gain_changes(rows, rows, arms=[OTHER_SOURCE], rng=NumpyRng(0), n_boot=50)

    assert (change.interval.point, change.interval.low, change.interval.high) == (0.0, 0.0, 0.0)


def test_the_gain_change_is_the_candidates_gain_minus_the_baselines():
    base = [_row(arm, doc, value) for doc in range(4) for arm, value in {RESET: 10.0, OTHER_SOURCE: 11.0}.items()]
    cand = [_row(arm, doc, value) for doc in range(4) for arm, value in {RESET: 10.0, OTHER_SOURCE: 10.0}.items()]

    (change,) = relatedness_eval.gain_changes(cand, base, arms=[OTHER_SOURCE], rng=NumpyRng(0), n_boot=20)

    assert change.interval.point == pytest.approx(1.0)
    assert change.win_rate == 1.0


def test_the_longest_chain_is_contrasted_with_the_shortest():
    ppl = {RESET: 10.0, "chain_1": 9.5, "chain_2": 9.8, "chain_10": 10.4}
    rows = [_row(arm, doc, value) for doc in range(3) for arm, value in ppl.items()]

    pooled = {e.measure: e.interval.point for e in relatedness_eval.estimates(rows, rng=NumpyRng(0), n_boot=10)
              if e.source == ALL_SOURCES}

    assert pooled["chain_10_vs_chain_1"] == pytest.approx(-0.9)
