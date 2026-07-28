"""OLD-vs-NEW session schedules under identical seeds (Phase 2's done-when).

The strategies draw from `rng` in the same order as the functions they replace,
so parity here is exact spans, not just matching counts.
"""

from __future__ import annotations

import numpy as np
import pytest
import train_utils

from ttt.extensions.strategies import EVERLASTING, HYBRID

pytestmark = pytest.mark.parity

MINIMUM = HYBRID.slice_min_tokens * HYBRID.slices_min
MAXIMUM = HYBRID.slice_min_tokens * HYBRID.slices_max

# Random lengths almost never land in the ~100-token band between "can be
# sliced" and "is allowed to be", which is exactly where the two sides could
# have disagreed.
BOUNDARIES = [
    MINIMUM - 1,
    MINIMUM,
    HYBRID.carry_min_tokens - 1,
    HYBRID.carry_min_tokens,
    MAXIMUM,
    MAXIMUM + 1,
]

LENGTHS = BOUNDARIES + [
    int(n) for n in np.random.default_rng(0).integers(200, 12_000, size=60)
]


def rng(seed):
    return np.random.default_rng(seed)


def spans(sessions):
    return [
        [(item.doc_idx, item.start, item.end) for item in session.items]
        for session in sessions
    ]


def old_spans(sessions):
    return [[(i.doc_idx, i.start, i.end) for i in session] for session in sessions]


def old_everlasting(lengths, generator):
    order = generator.permutation(len(lengths)).tolist()
    return [[train_utils.SessionItem(int(i), 0, int(lengths[i]))] for i in order]


def test_hybrid_reproduces_the_old_hybrid_schedule():
    old = train_utils.make_hybrid_sessions(
        len(LENGTHS),
        LENGTHS,
        rng(7),
        carry_min_tokens=HYBRID.carry_min_tokens,
        slice_min_tokens=HYBRID.slice_min_tokens,
        slices_min=HYBRID.slices_min,
        slices_max=HYBRID.slices_max,
    )

    assert spans(HYBRID.build(LENGTHS, rng(7))) == old_spans(old)


def test_hybrid_reproduces_the_old_item_count():
    old = train_utils.total_hybrid_items(
        LENGTHS,
        carry_min_tokens=HYBRID.carry_min_tokens,
        slice_min_tokens=HYBRID.slice_min_tokens,
        slices_min=HYBRID.slices_min,
        slices_max=HYBRID.slices_max,
    )

    assert HYBRID.count(LENGTHS) == old


def test_everlasting_reproduces_the_old_whole_doc_schedule():
    assert spans(EVERLASTING.build(LENGTHS, rng(7))) == old_spans(
        old_everlasting(LENGTHS, rng(7))
    )
