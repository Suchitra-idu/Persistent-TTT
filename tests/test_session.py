"""Session persistence tests: lifecycle helpers, schedule, and slice sessions."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import C, D, scan
from inplace_ttt import (
    advance_session_state, iter_ttt_modules, reset_session_state, state_norms,
)
from train_utils import (
    SessionItem, equal_token_slices, expected_items_per_doc,
    make_session_schedule, make_slice_sessions, slice_doc,
)


class Wrap(nn.Module):
    """Container so module-tree helpers walking model.modules() see the TTT module."""

    def __init__(self, m):
        super().__init__()
        self.mlp = m


def _enable_session_mode(model, on):
    for m in iter_ttt_modules(model):
        m.session_mode = on


# ---------- lifecycle ----------

def test_carry_changes_outputs_and_reset_restores(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    _enable_session_mode(model, True)
    x1, x2 = torch.randn(2 * C, D), torch.randn(2 * C, D)

    reset_session_state(model)
    fresh = scan(m, tap, x2)

    reset_session_state(model)
    scan(m, tap, x1)
    advance_session_state(model)
    carried = scan(m, tap, x2)
    assert not torch.allclose(fresh, carried)

    reset_session_state(model)
    again = scan(m, tap, x2)
    assert torch.equal(again, fresh)


def test_staging_is_idempotent_under_recompute(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    _enable_session_mode(model, True)
    reset_session_state(model)

    x = torch.randn(2 * C, D)
    scan(m, tap, x)
    first = m._next_carried.clone()
    scan(m, tap, x)
    assert torch.equal(m._next_carried, first)
    assert m.carried_delta is None

    advance_session_state(model)
    assert torch.equal(m.carried_delta, first)
    assert m._next_carried is None
    advance_session_state(model)
    assert torch.equal(m.carried_delta, first)


def test_carried_state_is_fp32_and_detached(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    _enable_session_mode(model, True)
    reset_session_state(model)
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)
    assert m.carried_delta.dtype == torch.float32
    assert not m.carried_delta.requires_grad


def test_state_norms_reads_carried_delta(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    assert state_norms(model, source="session") == {0: 0.0}
    _enable_session_mode(model, True)
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)
    assert state_norms(model, source="session")[0] > 0
    # Streaming source is still zero because state.delta was never populated.
    assert state_norms(model, source="stream") == {0: 0.0}


# ---------- session schedule ----------

def test_schedule_partitions_every_doc_exactly_once():
    rng = np.random.default_rng(7)
    sessions = make_session_schedule(503, lo=2, hi=6, rng=rng)
    flat = [d for s in sessions for d in s]
    assert sorted(flat) == list(range(503))


def test_schedule_sizes_within_bounds():
    rng = np.random.default_rng(7)
    sessions = make_session_schedule(500, lo=2, hi=6, rng=rng)
    assert all(2 <= len(s) <= 6 for s in sessions[:-1])
    assert 1 <= len(sessions[-1]) <= 6


# ---------- slice_doc primitive ----------

def test_slice_doc_partition_is_contiguous_and_meets_minimum():
    rng = np.random.default_rng(5)
    slices = slice_doc(10_000, 3, 1024, rng)
    assert len(slices) == 3
    assert slices[0][0] == 0
    assert slices[-1][1] == 10_000
    for (_, b), (c, _) in zip(slices, slices[1:]):
        assert b == c
    assert all((e - s) >= 1024 for s, e in slices)


def test_slice_doc_fallback_when_too_short():
    rng = np.random.default_rng(5)
    assert slice_doc(2000, 3, 1024, rng) == [(0, 2000)]


def test_slice_doc_k_one_is_whole():
    rng = np.random.default_rng(5)
    assert slice_doc(10_000, 1, 1024, rng) == [(0, 10_000)]


def test_slice_doc_boundary_min_size_does_not_underflow():
    rng = np.random.default_rng(5)
    slices = slice_doc(2048, 2, 1024, rng)
    assert slices == [(0, 1024), (1024, 2048)]


# ---------- make_slice_sessions: multi-paper mode ----------

def _multi(num_docs, doc_lengths, rng, slice_prob=1.0,
           slice_range=(2, 4), session_papers=(2, 4), shuffle=True):
    return make_slice_sessions(
        num_docs, doc_lengths, rng,
        session_papers=session_papers, slice_prob=slice_prob,
        slice_range=slice_range, min_slice_tokens=1024, shuffle=shuffle,
    )


def test_multi_session_covers_every_token_exactly_once():
    rng = np.random.default_rng(5)
    doc_lengths = [5000 + (i % 7) * 1000 for i in range(20)]
    sessions = _multi(20, doc_lengths, rng)
    by_doc = {}
    for session in sessions:
        for item in session:
            by_doc.setdefault(item.doc_idx, []).append((item.start, item.end))
    assert set(by_doc.keys()) == set(range(20))
    for d, ranges in by_doc.items():
        ranges.sort()
        assert ranges[0][0] == 0
        assert ranges[-1][1] == doc_lengths[d]
        for (_, b), (c, _) in zip(ranges, ranges[1:]):
            assert b == c


def test_multi_session_slice_prob_zero_gives_one_item_per_doc():
    rng = np.random.default_rng(5)
    doc_lengths = [5000] * 10
    sessions = _multi(10, doc_lengths, rng, slice_prob=0.0)
    for s in sessions:
        for it in s:
            assert (it.start, it.end) == (0, doc_lengths[it.doc_idx])


def test_multi_session_short_docs_fall_back_to_one_item():
    rng = np.random.default_rng(5)
    doc_lengths = [1500] * 12   # all too short to slice
    sessions = _multi(12, doc_lengths, rng)
    docs = [it.doc_idx for s in sessions for it in s]
    assert sorted(docs) == list(range(12))


def test_multi_session_shuffle_false_preserves_order():
    rng = np.random.default_rng(5)
    doc_lengths = [10_000] * 5
    sessions = _multi(5, doc_lengths, rng, slice_prob=0.0,
                      session_papers=(5, 5), shuffle=False)
    docs = [it.doc_idx for s in sessions for it in s]
    assert docs == [0, 1, 2, 3, 4]


# ---------- make_slice_sessions: single-paper mode ----------

def _single(num_docs, doc_lengths, rng, slice_range=(2, 4)):
    return make_slice_sessions(
        num_docs, doc_lengths, rng,
        session_papers=(1, 1), slice_prob=1.0,
        slice_range=slice_range, min_slice_tokens=1024,
    )


def test_single_session_one_paper_per_session():
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 5
    sessions = _single(5, doc_lengths, rng)
    assert len(sessions) == 5
    for s in sessions:
        assert len({it.doc_idx for it in s}) == 1


def test_single_session_every_paper_appears_exactly_once():
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 10
    sessions = _single(10, doc_lengths, rng)
    seen = {s[0].doc_idx for s in sessions}
    assert seen == set(range(10))


def test_single_session_slices_are_contiguous_and_cover_doc():
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 5
    sessions = _single(5, doc_lengths, rng)
    for s in sessions:
        L = doc_lengths[s[0].doc_idx]
        assert s[0].start == 0
        assert s[-1].end == L
        for prev, curr in zip(s, s[1:]):
            assert prev.end == curr.start


def test_single_session_short_doc_falls_back_to_whole():
    rng = np.random.default_rng(7)
    doc_lengths = [1500] * 5
    sessions = _single(5, doc_lengths, rng)
    for s in sessions:
        assert len(s) == 1
        assert s[0].start == 0 and s[0].end == 1500


def test_single_session_respects_slice_min_tokens():
    rng = np.random.default_rng(7)
    doc_lengths = [10_000] * 10
    sessions = _single(10, doc_lengths, rng)
    for s in sessions:
        if len(s) > 1:
            assert all(it.end - it.start >= 1024 for it in s)


# ---------- determinism ----------

@pytest.mark.parametrize("builder", [
    lambda seed: make_session_schedule(100, 2, 6, np.random.default_rng(seed)),
    lambda seed: _multi(20, [10_000] * 20, np.random.default_rng(seed)),
    lambda seed: _single(20, [10_000] * 20, np.random.default_rng(seed)),
])
def test_builders_are_deterministic_per_seed(builder):
    a, b, c = builder(11), builder(11), builder(12)
    assert a == b
    assert a != c


# ---------- equal_token_slices ----------

def test_equal_token_slices_contiguous_and_covers_full_range():
    slices = equal_token_slices(1003, 4)
    assert slices[0][0] == 0
    assert slices[-1][1] == 1003
    for (_, b), (c, _) in zip(slices, slices[1:]):
        assert b == c
    sizes = [e - s for s, e in slices]
    assert max(sizes) - min(sizes) <= 1


def test_equal_token_slices_drops_empty_when_more_slices_than_tokens():
    slices = equal_token_slices(3, 10)
    assert all(e > s for s, e in slices)
    assert sum(e - s for s, e in slices) == 3


def test_equal_token_slices_rejects_zero_or_negative_n():
    with pytest.raises(ValueError):
        equal_token_slices(1000, 0)
    with pytest.raises(ValueError):
        equal_token_slices(1000, -3)


# ---------- expected_items_per_doc ----------

def test_expected_items_per_doc_matches_linear_expectation():
    # Halfway between not-sliced (1.0) and full slice range midpoint (3.0).
    assert expected_items_per_doc(0.5, 2, 4) == 2.0
