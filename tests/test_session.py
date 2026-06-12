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
from train_utils import make_session_schedule


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
