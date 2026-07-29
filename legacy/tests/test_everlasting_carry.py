"""Tests for the everlasting-carry mechanism.

Covers the pure helpers -- install / snapshot on TTT modules, save / load
round-trip of the per-source carrier dict, and the training-side session
schedule (whole-doc, one-item-per-session) -- so a regression is caught
without needing a Modal deploy.
"""

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import D, DFF
from inplace_ttt import (
    advance_session_state, install_carried_delta, iter_ttt_modules,
    reset_session_state, snapshot_carried_delta,
)
from ttt_wiring import load_per_source_carries, save_per_source_carries


class Wrap(nn.Module):
    """Container so module-tree helpers see the TTT module via .modules()."""

    def __init__(self, m):
        super().__init__()
        self.mlp = m


def _enable(model):
    for m in iter_ttt_modules(model):
        m.session_mode = True


# ---------- install / snapshot ----------

def test_snapshot_returns_empty_when_no_carry(module_factory):
    m, _, _ = module_factory()
    snap = snapshot_carried_delta(Wrap(m))
    assert snap == {}


def test_install_then_snapshot_roundtrips(module_factory):
    m, _, _ = module_factory()
    model = Wrap(m)
    src = {int(m.cfg.layer_indices[0]): torch.randn(1, D, DFF)}

    install_carried_delta(model, src)
    snap = snapshot_carried_delta(model, to_cpu=False)

    key = int(m.cfg.layer_indices[0])
    assert key in snap
    assert torch.allclose(snap[key], src[key].float())


def test_install_missing_key_is_noop(module_factory):
    """A snapshot dict that lacks this module's layer index leaves
    carried_delta untouched -- cold-start behavior for unseen sources."""
    m, _, _ = module_factory()
    model = Wrap(m)
    layer_key = int(m.cfg.layer_indices[0])

    seed = torch.randn(1, D, DFF)
    install_carried_delta(model, {layer_key: seed})
    assert m.carried_delta is not None

    # New install lacks our layer -- prior value should remain.
    install_carried_delta(model, {999: torch.randn(1, D, DFF)})
    assert torch.allclose(m.carried_delta, seed.float())


def test_snapshot_is_detached_and_fp32(module_factory):
    m, _, _ = module_factory()
    model = Wrap(m)
    # Provide a bf16-esque tensor to force the fp32 cast on snapshot.
    install_carried_delta(
        model,
        {int(m.cfg.layer_indices[0]): torch.randn(1, D, DFF).to(torch.float64)},
    )
    snap = snapshot_carried_delta(model, to_cpu=True)
    for t in snap.values():
        assert t.dtype == torch.float32
        assert not t.requires_grad
        assert t.device.type == "cpu"


def test_snapshot_survives_reset(module_factory):
    """Snapshot returns clones -- clearing the module doesn't touch the copy."""
    m, _, _ = module_factory()
    model = Wrap(m)
    layer_key = int(m.cfg.layer_indices[0])
    src = torch.randn(1, D, DFF)
    install_carried_delta(model, {layer_key: src})
    snap = snapshot_carried_delta(model)
    reset_session_state(model)
    assert m.carried_delta is None
    assert torch.allclose(snap[layer_key], src.float())


# ---------- save / load ----------

def test_save_load_per_source_carries_roundtrip(tmp_path):
    path = str(tmp_path / "per_source_carries.pt")
    carries = {
        "RedPajamaC4": {5: torch.randn(1, 4, 8), 11: torch.randn(1, 4, 8)},
        "RedPajamaGithub": {5: torch.randn(1, 4, 8)},
    }
    n_updates = {"RedPajamaC4": 12, "RedPajamaGithub": 3}

    save_per_source_carries(carries, path,
                            meta={"n_updates": n_updates, "step": 200})
    loaded, meta = load_per_source_carries(path)

    assert set(loaded) == {"RedPajamaC4", "RedPajamaGithub"}
    for src, per_layer in carries.items():
        for k, v in per_layer.items():
            assert torch.allclose(loaded[src][int(k)], v.float())
    assert meta["n_updates"] == n_updates
    assert meta["step"] == 200


def test_load_per_source_carries_missing_file():
    loaded, meta = load_per_source_carries("/nonexistent/path.pt")
    assert loaded == {}
    assert meta == {}


def test_save_load_empty_carries(tmp_path):
    """Empty dict on disk -> empty dict on load (used by non-everlasting
    checkpoints if the file is created by accident)."""
    path = str(tmp_path / "empty.pt")
    save_per_source_carries({}, path)
    loaded, meta = load_per_source_carries(path)
    assert loaded == {}


# ---------- train-loop schedule ----------

def test_make_epoch_sessions_everlasting_is_whole_doc():
    """In everlasting mode, every doc becomes a length-1 session with
    the item covering the whole doc; no slicing regardless of length."""
    from ttt_config import TRAIN_CFG
    from train_modal import _make_epoch_sessions

    cfg = dataclasses.replace(TRAIN_CFG, everlasting_carry=True)
    lengths = [1000, 50_000, 200]
    rng = np.random.default_rng(0)
    sessions = _make_epoch_sessions(cfg, len(lengths), lengths, rng)

    assert len(sessions) == len(lengths)
    seen_doc_ids = []
    for session in sessions:
        assert len(session) == 1
        item = session[0]
        assert item.start == 0
        assert item.end == lengths[item.doc_idx]
        seen_doc_ids.append(item.doc_idx)
    assert sorted(seen_doc_ids) == list(range(len(lengths)))


def test_items_per_epoch_everlasting_equals_num_docs():
    from ttt_config import TRAIN_CFG
    from train_modal import _items_per_epoch

    cfg = dataclasses.replace(TRAIN_CFG, everlasting_carry=True)
    assert _items_per_epoch(cfg, [1000, 2000, 3000]) == 3


def test_apply_overrides_mode_everlasting_sets_flags():
    from train_modal import _apply_cli_overrides
    from tests.test_train_modal_utils import _defaults  # reuse defaults

    kw = _defaults(); kw["mode"] = "everlasting"
    cfg = _apply_cli_overrides(**kw)
    assert cfg.everlasting_carry is True
    assert cfg.single_paper_sessions is False
    assert cfg.hybrid_sessions is False
    # session_training is force-enabled since the mechanism relies on it.
    assert cfg.session_training is True


def test_apply_overrides_mode_everlasting_defaults_eval_every_25():
    """Everlasting mode has ~10x fewer optimizer steps per epoch, so the
    default eval_every=100 fires too rarely -- CLI dispatch bumps it to 25."""
    from train_modal import _apply_cli_overrides
    from tests.test_train_modal_utils import _defaults

    kw = _defaults(); kw["mode"] = "everlasting"
    cfg = _apply_cli_overrides(**kw)
    assert cfg.eval_every == 25


def test_apply_overrides_explicit_eval_every_wins_over_mode_default():
    """--eval-every 10 with --mode everlasting keeps the user's 10."""
    from train_modal import _apply_cli_overrides
    from tests.test_train_modal_utils import _defaults

    kw = _defaults(); kw["mode"] = "everlasting"; kw["eval_every"] = 10
    cfg = _apply_cli_overrides(**kw)
    assert cfg.eval_every == 10


def test_apply_overrides_mode_everlasting_respects_explicit_session_off():
    """--session 0 alongside --mode everlasting is a user override we
    don't stomp on (config validation happens at train() start)."""
    from train_modal import _apply_cli_overrides
    from tests.test_train_modal_utils import _defaults

    kw = _defaults(); kw["mode"] = "everlasting"; kw["session"] = 0
    cfg = _apply_cli_overrides(**kw)
    assert cfg.everlasting_carry is True
    assert cfg.session_training is False


def test_mode_line_reports_everlasting():
    from ttt_config import TRAIN_CFG
    from train_modal import _mode_line

    cfg = dataclasses.replace(TRAIN_CFG, everlasting_carry=True)
    line = _mode_line(cfg)
    assert "everlasting_carry=True" in line


# ---------- end-to-end forward with installed carry ----------

def test_installed_carry_affects_forward(module_factory):
    """Sanity: installing a nonzero carrier changes the forward output vs
    a fresh session with no carrier."""
    m, _, tap = module_factory(randomize=True)
    model = Wrap(m)
    _enable(model)

    x = torch.randn(2 * m.cfg.chunk_size, D)
    tap.current = x.unsqueeze(0)

    reset_session_state(model)
    with torch.no_grad():
        fresh = m(x.unsqueeze(0))[0]

    reset_session_state(model)
    install_carried_delta(
        model,
        {int(m.cfg.layer_indices[0]): torch.randn(1, D, DFF) * 0.1},
    )
    with torch.no_grad():
        seeded = m(x.unsqueeze(0))[0]

    assert not torch.allclose(fresh, seeded)
