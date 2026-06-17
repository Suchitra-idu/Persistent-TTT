"""
Session persistence tests, the lifecycle helpers and the schedule.
The staging idempotence test simulates gradient checkpointing's double
forward, the exact scenario the two-phase commit exists for.
"""

import numpy as np
import torch
import torch.nn as nn

from conftest import C, D, scan
from inplace_ttt import (
    advance_session_state, reset_session_state, session_state_norms,
    set_session_mode,
)
import pytest

from train_utils import (
    SessionItem, build_session_items, equal_token_slices,
    expected_items_per_doc, make_session_schedule,
    make_single_paper_sessions, slice_doc,
)


class Wrap(nn.Module):
    """Minimal container so the module-tree helpers (which walk
    model.modules()) see the TTT module."""

    def __init__(self, m):
        super().__init__()
        self.mlp = m


# ---------------------------------------------------------------- lifecycle --
def test_carry_changes_outputs_and_reset_restores(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    set_session_mode(model, True)
    x1, x2 = torch.randn(2 * C, D), torch.randn(2 * C, D)

    reset_session_state(model)
    fresh = scan(m, tap, x2)

    reset_session_state(model)
    scan(m, tap, x1)
    advance_session_state(model)
    carried = scan(m, tap, x2)
    assert not torch.allclose(fresh, carried)     # paper 1 left a mark

    reset_session_state(model)
    again = scan(m, tap, x2)
    assert torch.equal(again, fresh)              # reset means reset


def test_staging_is_idempotent_under_recompute(module_factory):
    """Gradient checkpointing reruns the forward during backward. The
    staged _next_carried must be identical across reruns and the
    promoted state must advance exactly once."""
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    set_session_mode(model, True)
    reset_session_state(model)

    x = torch.randn(2 * C, D)
    scan(m, tap, x)
    first = m._next_carried.clone()
    scan(m, tap, x)                               # simulated recompute
    assert torch.equal(m._next_carried, first)
    assert m.carried_delta is None                # not advanced yet

    advance_session_state(model)
    assert torch.equal(m.carried_delta, first)
    assert m._next_carried is None
    advance_session_state(model)                  # double promote = no-op
    assert torch.equal(m.carried_delta, first)


def test_carried_state_is_fp32_and_detached(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    set_session_mode(model, True)
    reset_session_state(model)
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)
    assert m.carried_delta.dtype == torch.float32
    assert not m.carried_delta.requires_grad


def test_session_state_norms(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    assert session_state_norms(model) == {0: 0.0}
    set_session_mode(model, True)
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)
    norms = session_state_norms(model)
    assert norms[0] > 0


# ----------------------------------------------------------------- schedule --
def test_schedule_partitions_every_doc_exactly_once():
    rng = np.random.default_rng(7)
    sessions = make_session_schedule(503, lo=2, hi=6, rng=rng)
    flat = [d for s in sessions for d in s]
    assert sorted(flat) == list(range(503))


def test_schedule_sizes_within_bounds():
    rng = np.random.default_rng(7)
    sessions = make_session_schedule(500, lo=2, hi=6, rng=rng)
    assert all(2 <= len(s) <= 6 for s in sessions[:-1])
    assert 1 <= len(sessions[-1]) <= 6            # remainder may be short


def test_schedule_deterministic_per_seed():
    a = make_session_schedule(100, 2, 6, np.random.default_rng(1))
    b = make_session_schedule(100, 2, 6, np.random.default_rng(1))
    c = make_session_schedule(100, 2, 6, np.random.default_rng(2))
    assert a == b
    assert a != c


# ----------------------------------------------------------------- slicing --
def test_slice_doc_partition_is_contiguous_and_meets_minimum():
    rng = np.random.default_rng(5)
    slices = slice_doc(10_000, 3, 1024, rng)
    assert len(slices) == 3
    assert slices[0][0] == 0
    assert slices[-1][1] == 10_000
    for (_, b), (c, _) in zip(slices, slices[1:]):
        assert b == c                              # no gaps, no overlap
    assert all((e - s) >= 1024 for s, e in slices)


def test_slice_doc_fallback_when_too_short():
    """k * min_slice_tokens > L => one whole-doc slice."""
    rng = np.random.default_rng(5)
    assert slice_doc(2000, 3, 1024, rng) == [(0, 2000)]


def test_slice_doc_k_one_is_whole():
    rng = np.random.default_rng(5)
    assert slice_doc(10_000, 1, 1024, rng) == [(0, 10_000)]


def test_slice_doc_boundary_min_size_does_not_underflow():
    """L = k * min_tokens exactly; every slice must be exactly min."""
    rng = np.random.default_rng(5)
    slices = slice_doc(2048, 2, 1024, rng)
    assert slices == [(0, 1024), (1024, 2048)]


def test_build_session_items_disabled_matches_schedule():
    """slice_prob=0 must reproduce the legacy schedule structurally
    (one whole-doc item per paper, paper order preserved)."""
    rng = np.random.default_rng(5)
    doc_lengths = [5000] * 10
    sessions = [[2, 0, 4], [1, 3], [5, 6, 7, 8, 9]]
    out = build_session_items(sessions, doc_lengths, slice_prob=0.0,
                              slice_min=2, slice_max=4,
                              min_slice_tokens=1024, rng=rng)
    expected = [
        [SessionItem(d, 0, doc_lengths[d]) for d in s] for s in sessions
    ]
    assert out == expected


def test_build_session_items_covers_every_token_exactly_once():
    """For each paper, the union of its slices is exactly [0, L]."""
    rng = np.random.default_rng(5)
    doc_lengths = [5000 + (i % 7) * 1000 for i in range(20)]
    sessions = [[0, 1, 2], [3, 4], list(range(5, 20))]
    out = build_session_items(sessions, doc_lengths, slice_prob=1.0,
                              slice_min=2, slice_max=4,
                              min_slice_tokens=1024, rng=rng)
    by_doc = {}
    for session in out:
        for item in session:
            by_doc.setdefault(item.doc_idx, []).append((item.start, item.end))
    for d, ranges in by_doc.items():
        ranges.sort()
        assert ranges[0][0] == 0
        assert ranges[-1][1] == doc_lengths[d]
        for (_, b), (c, _) in zip(ranges, ranges[1:]):
            assert b == c                          # contiguous within doc


def test_build_session_items_preserves_paper_order_within_session():
    """Slices of paper i stay grouped and ordered; paper order inside
    a session is preserved across slicing."""
    rng = np.random.default_rng(5)
    doc_lengths = [10_000] * 5
    sessions = [[2, 0, 4, 1, 3]]
    out = build_session_items(sessions, doc_lengths, slice_prob=1.0,
                              slice_min=2, slice_max=3,
                              min_slice_tokens=1024, rng=rng)
    seen = []
    for item in out[0]:
        if not seen or seen[-1] != item.doc_idx:
            seen.append(item.doc_idx)
    assert seen == [2, 0, 4, 1, 3]


def test_build_session_items_every_paper_appears_at_least_once():
    """No paper is silently dropped, even at slice_prob=1.0 with short docs."""
    rng = np.random.default_rng(5)
    doc_lengths = [1500] * 12                      # all too short to slice
    sessions = [list(range(12))]
    out = build_session_items(sessions, doc_lengths, slice_prob=1.0,
                              slice_min=2, slice_max=4,
                              min_slice_tokens=1024, rng=rng)
    doc_ids = {item.doc_idx for item in out[0]}
    assert doc_ids == set(range(12))
    # And each falls back to one whole-doc item.
    assert len(out[0]) == 12


def test_build_session_items_deterministic_per_seed():
    doc_lengths = [10_000] * 20
    sessions = [[0, 1, 2], [3, 4, 5]]
    a = build_session_items(sessions, doc_lengths, 0.5, 2, 4, 1024,
                            np.random.default_rng(11))
    b = build_session_items(sessions, doc_lengths, 0.5, 2, 4, 1024,
                            np.random.default_rng(11))
    c = build_session_items(sessions, doc_lengths, 0.5, 2, 4, 1024,
                            np.random.default_rng(12))
    assert a == b
    assert a != c


def test_session_item_n_tokens():
    assert SessionItem(7, 100, 350).n_tokens == 250


def test_expected_items_per_doc_bounds():
    assert expected_items_per_doc(0.0, 2, 4) == 1.0
    assert expected_items_per_doc(1.0, 2, 4) == 3.0  # midpoint of [2,4]
    # slice_prob=0.5 halfway between 1 (no slice) and 3 (avg slice)
    assert expected_items_per_doc(0.5, 2, 4) == 2.0
    # slice_max=1 means slicing disabled regardless of prob
    assert expected_items_per_doc(1.0, 1, 1) == 1.0


# ----------------------------------------------------- equal slicing --
def test_equal_token_slices_exact_division():
    """1000 / 4 = 250 exactly, so all four slices should be equal."""
    assert equal_token_slices(1000, 4) == [
        (0, 250), (250, 500), (500, 750), (750, 1000),
    ]


def test_equal_token_slices_contiguous_and_covers_full_range():
    """For arbitrary L, slices must partition [0, L] with no gap or
    overlap, last boundary pinned exactly at L."""
    slices = equal_token_slices(1003, 4)
    assert slices[0][0] == 0
    assert slices[-1][1] == 1003
    for (_, b), (c, _) in zip(slices, slices[1:]):
        assert b == c


def test_equal_token_slices_drops_empty_when_more_slices_than_tokens():
    """n_slices > doc_length would produce zero-length slices via
    rounding; those must be dropped, not crashed on."""
    slices = equal_token_slices(3, 10)
    assert all(e > s for s, e in slices)
    # Total token count is preserved.
    assert sum(e - s for s, e in slices) == 3


def test_equal_token_slices_n_one_returns_whole_doc():
    assert equal_token_slices(1000, 1) == [(0, 1000)]


def test_equal_token_slices_rejects_zero_or_negative_n():
    with pytest.raises(ValueError):
        equal_token_slices(1000, 0)
    with pytest.raises(ValueError):
        equal_token_slices(1000, -3)


def test_equal_token_slices_sizes_balanced_within_one():
    """Token counts should differ by at most 1 between any two slices."""
    slices = equal_token_slices(1003, 4)
    sizes = [e - s for s, e in slices]
    assert max(sizes) - min(sizes) <= 1


# --------------------------------------------- single-paper sessions --
def test_single_paper_sessions_one_paper_per_session():
    """Every item in a session must come from the same paper."""
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 5
    sessions = make_single_paper_sessions(5, doc_lengths, 2, 4, 1024, rng)
    assert len(sessions) == 5
    for s in sessions:
        paper_ids = {it.doc_idx for it in s}
        assert len(paper_ids) == 1


def test_single_paper_sessions_every_paper_appears_exactly_once():
    """Each paper produces exactly one session; no paper is dropped or
    duplicated across the epoch's shuffle."""
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 10
    sessions = make_single_paper_sessions(10, doc_lengths, 2, 4, 1024, rng)
    seen = set()
    for s in sessions:
        pid = s[0].doc_idx
        assert pid not in seen
        seen.add(pid)
    assert seen == set(range(10))


def test_single_paper_sessions_slices_are_contiguous():
    """Slices within a session must cover [0, L] with no gaps/overlap."""
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 5
    sessions = make_single_paper_sessions(5, doc_lengths, 2, 4, 1024, rng)
    for s in sessions:
        L = doc_lengths[s[0].doc_idx]
        assert s[0].start == 0
        assert s[-1].end == L
        for prev, curr in zip(s, s[1:]):
            assert prev.end == curr.start


def test_single_paper_sessions_short_doc_falls_back_to_whole():
    """Paper too short for k*min_slice_tokens -> whole-paper single item.
    Must NOT silently emit zero-length slices or skip the paper."""
    rng = np.random.default_rng(7)
    doc_lengths = [1500] * 5  # 2 * 1024 = 2048 > 1500
    sessions = make_single_paper_sessions(5, doc_lengths, 2, 4, 1024, rng)
    for s in sessions:
        assert len(s) == 1
        assert s[0].start == 0
        assert s[0].end == 1500


def test_single_paper_sessions_respects_slice_min_tokens():
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 10
    sessions = make_single_paper_sessions(10, doc_lengths, 2, 4, 1024, rng)
    for s in sessions:
        if len(s) > 1:
            for item in s:
                assert item.end - item.start >= 1024


def test_single_paper_sessions_deterministic_per_seed():
    doc_lengths = [10_000] * 20
    a = make_single_paper_sessions(
        20, doc_lengths, 2, 4, 1024, np.random.default_rng(11),
    )
    b = make_single_paper_sessions(
        20, doc_lengths, 2, 4, 1024, np.random.default_rng(11),
    )
    c = make_single_paper_sessions(
        20, doc_lengths, 2, 4, 1024, np.random.default_rng(12),
    )
    assert a == b
    assert a != c


def test_single_paper_sessions_slice_count_within_bounds():
    """When the doc is long enough, the chosen k stays in [lo, hi]."""
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 50  # long enough for any k in [2, 4]
    sessions = make_single_paper_sessions(50, doc_lengths, 2, 4, 1024, rng)
    counts = [len(s) for s in sessions]
    assert min(counts) >= 2
    assert max(counts) <= 4
