from __future__ import annotations

import numpy as np
import pytest

from ttt.app import token_gain_analysis as t
from ttt.app.relatedness_eval import IN_CONTEXT, OTHER_SOURCE, RANDOM, RELATED, RESET, SELF

N_POS = 1023


def case(i, related_title=None, prime_tokens=(1, 2)):
    target = [5000 + i] + [7, 9] * 511 + [7]
    return {
        "index": i,
        "target": target,
        "primes": {SELF: [], RELATED: list(prime_tokens), RANDOM: [3], OTHER_SOURCE: []},
        "prime_titles": {SELF: f"s{i}", RELATED: related_title or f"r{i}", RANDOM: f"x{i}", OTHER_SOURCE: "o"},
    }


def rows_for(n, arm_losses):
    rows = []
    for d in range(n):
        rows.append({"arm": RESET, "doc_idx": d, "losses": [1.0] * N_POS})
        for arm in (SELF, RELATED, RANDOM, OTHER_SOURCE, IN_CONTEXT):
            rows.append({"arm": arm, "doc_idx": d, "losses": list(arm_losses(arm, d))})
    return rows


def front_loaded(arm, d):
    losses = np.ones(N_POS)
    losses[:31] -= 0.5 + 0.01 * d
    return losses


def test_cases_sharing_a_related_prime_form_one_cluster():
    assert t.clusters([case(0, "A"), case(1, "B"), case(2, "A")]) == [0, 1, 0]


def test_novel_tokens_are_those_in_neither_prime():
    mask = t.novel_mask([case(0, prime_tokens=(7,))])

    assert not mask[0, ::2].any() and mask[0, 1::2].all()


def test_gain_confined_to_the_first_32_tokens_gives_a_front_share_of_one():
    a = t.analyse(rows_for(6, front_loaded), [case(i) for i in range(6)], n_boot=200)

    assert a.s32[SELF].point == pytest.approx(1.0) and a.a1


def test_gain_spread_evenly_fails_front_loading():
    a = t.analyse(rows_for(6, lambda arm, d: np.full(N_POS, 0.9 - 0.01 * d)), [case(i) for i in range(6)], n_boot=200)

    assert a.s32[RELATED].point == pytest.approx(31 / N_POS) and not a.a1


def test_related_beating_random_on_novel_tokens_only_is_a_novel_advantage():
    def losses(arm, d):
        out = np.ones(N_POS)
        if arm == RELATED:
            out[1::2] -= 0.1 + 0.01 * d
        return out

    a = t.analyse(rows_for(6, losses), [case(i, prime_tokens=(7,)) for i in range(6)], n_boot=200)

    assert a.novel_advantage.point > 0 and a.shared_advantage.point == pytest.approx(0.0) and a.a2


def test_carry_over_context_is_the_ratio_of_per_token_gains():
    a = t.analyse(rows_for(4, lambda arm, d: np.full(N_POS, 0.5 if arm == IN_CONTEXT else 0.75)),
                  [case(i) for i in range(4)], n_boot=50)

    assert a.carry_over_context == pytest.approx((0.5,) * len(t.BINS))
