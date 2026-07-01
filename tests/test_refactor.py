"""Tests for helpers introduced by the cleanup refactor.

Covers:
  - state_norms(source={"session","stream"}) dispatch on carried_delta vs state.delta
  - make_slice_sessions unified API (single-paper and multi-paper modes)
  - _resolve_resume path resolution + missing-file guard
  - _token_weighted_ppl math
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import C, D, scan
from inplace_ttt import (
    advance_session_state, iter_ttt_modules, reset_session_state, state_norms,
)
from train_utils import make_slice_sessions


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mlp = m


# ---------- state_norms dispatch ----------

def test_state_norms_session_source_only_reads_carried_delta(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    for tm in iter_ttt_modules(model):
        tm.session_mode = True
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)

    session_out = state_norms(model, source="session")
    stream_out = state_norms(model, source="stream")
    assert session_out[0] > 0
    assert stream_out[0] == 0.0


def test_state_norms_stream_source_only_reads_state_delta(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    # Force a streaming commit by populating state.delta directly.
    m.state.delta = torch.randn_like(m.down_proj.weight).float()
    session_out = state_norms(model, source="session")
    stream_out = state_norms(model, source="stream")
    assert stream_out[0] > 0
    assert session_out[0] == 0.0


def test_state_norms_default_source_is_session(module_factory):
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    for tm in iter_ttt_modules(model):
        tm.session_mode = True
    scan(m, tap, torch.randn(2 * C, D))
    advance_session_state(model)
    default = state_norms(model)
    explicit = state_norms(model, source="session")
    assert default == explicit


def test_state_norms_no_modules_returns_empty_dict():
    empty_model = nn.Module()
    assert state_norms(empty_model, source="session") == {}
    assert state_norms(empty_model, source="stream") == {}


# ---------- make_slice_sessions API surface ----------

def test_slice_sessions_shape_matches_num_docs_in_single_paper_mode():
    rng = np.random.default_rng(0)
    doc_lengths = [10_000] * 7
    sessions = make_slice_sessions(
        7, doc_lengths, rng,
        session_papers=(1, 1), slice_prob=1.0,
        slice_range=(2, 4), min_slice_tokens=1024,
    )
    assert len(sessions) == 7
    for s in sessions:
        assert len({it.doc_idx for it in s}) == 1


def test_slice_sessions_multi_paper_grouping_size_within_range():
    rng = np.random.default_rng(0)
    doc_lengths = [5000] * 30
    sessions = make_slice_sessions(
        30, doc_lengths, rng,
        session_papers=(3, 5), slice_prob=0.0,
        slice_range=(1, 1), min_slice_tokens=1024,
    )
    unique_docs_per_session = [len({it.doc_idx for it in s}) for s in sessions]
    # All but the tail must sit within [3, 5]; tail may be shorter.
    assert all(3 <= n <= 5 for n in unique_docs_per_session[:-1])


def test_slice_sessions_shuffle_false_yields_stable_order():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(999)
    doc_lengths = [10_000] * 6
    kwargs = dict(
        session_papers=(6, 6), slice_prob=0.0,
        slice_range=(1, 1), min_slice_tokens=1024, shuffle=False,
    )
    a = make_slice_sessions(6, doc_lengths, rng_a, **kwargs)
    b = make_slice_sessions(6, doc_lengths, rng_b, **kwargs)
    assert [it.doc_idx for it in a[0]] == [0, 1, 2, 3, 4, 5]
    assert [it.doc_idx for it in b[0]] == [0, 1, 2, 3, 4, 5]


def test_slice_sessions_decrements_k_toward_feasibility():
    # L=2500 with min_slice_tokens=1024 permits k=2 (2*1024=2048<=2500) but
    # not k=3 (3*1024=3072>2500). Ensure we get exactly 2 items, not 1.
    rng = np.random.default_rng(0)
    doc_lengths = [2500]
    sessions = make_slice_sessions(
        1, doc_lengths, rng,
        session_papers=(1, 1), slice_prob=1.0,
        slice_range=(3, 3), min_slice_tokens=1024,
    )
    assert len(sessions[0]) == 2


# ---------- _resolve_resume ----------

def test_resolve_resume_none_returns_none_pair(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    assert _resolve_resume("", "any-run") == (None, None)


def test_resolve_resume_same_run_form(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    step_dir = tmp_path / "myrun" / "step_100"
    (step_dir / "adapter").mkdir(parents=True)
    (step_dir / "ttt_params.pt").write_bytes(b"")

    adapter, ttt = _resolve_resume("step_100", "myrun")
    assert adapter == str(step_dir / "adapter")
    assert ttt == str(step_dir / "ttt_params.pt")


def test_resolve_resume_cross_run_form(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    step_dir = tmp_path / "otherrun" / "step_50"
    (step_dir / "adapter").mkdir(parents=True)
    (step_dir / "ttt_params.pt").write_bytes(b"")

    adapter, ttt = _resolve_resume("otherrun/step_50", "currentrun")
    assert adapter.startswith(str(tmp_path / "otherrun" / "step_50"))
    assert not adapter.startswith(str(tmp_path / "currentrun"))


def test_resolve_resume_missing_files_raises(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    (tmp_path / "myrun" / "step_100" / "adapter").mkdir(parents=True)
    # ttt_params.pt intentionally missing.
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        _resolve_resume("step_100", "myrun")


# ---------- eval math ----------

def test_token_weighted_ppl_matches_log_mean():
    import math

    from train_modal import _token_weighted_ppl

    rows = [(100, 10.0, 0.0), (300, 20.0, 0.0)]   # (n_tok, ppl, state_ratio)
    # exp((100*log 10 + 300*log 20) / 400)
    expected = math.exp((100 * math.log(10) + 300 * math.log(20)) / 400)
    got = _token_weighted_ppl(rows)
    assert abs(got - expected) < 1e-12


def test_token_weighted_ppl_empty_returns_nan():
    import math

    from train_modal import _token_weighted_ppl

    assert math.isnan(_token_weighted_ppl([]))
