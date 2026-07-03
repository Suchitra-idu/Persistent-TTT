"""Tests for pure helpers inside train_modal.py.

Covers:
  - _resolve_resume path resolution + missing-file guard
  - _token_weighted_ppl math
"""

import math

import pytest


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


def test_token_weighted_ppl_matches_log_mean():
    from train_modal import _token_weighted_ppl

    rows = [(100, 10.0, 0.0), (300, 20.0, 0.0)]
    expected = math.exp((100 * math.log(10) + 300 * math.log(20)) / 400)
    assert abs(_token_weighted_ppl(rows) - expected) < 1e-12


def test_token_weighted_ppl_empty_returns_nan():
    from train_modal import _token_weighted_ppl

    assert math.isnan(_token_weighted_ppl([]))
