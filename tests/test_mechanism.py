"""
Mechanism tests for InPlaceTTTMLP, organized by the guarantee each one
protects. Every test here corresponds to a property that, if silently
broken, would invalidate experiments rather than crash them.
"""

import dataclasses

import pytest
import torch

from conftest import C, D, scan
from inplace_ttt import TTTState
from ttt_config import TTTConfig


# ---------------------------------------------------------------- identity --
def test_exact_identity_at_zero_init(module_factory):
    """W_target zero-init must make the module bit-equivalent to the
    plain MLP. This is the local twin of the remote sanity_check; if it
    fails, training would start from a corrupted model."""
    m, mlp, tap = module_factory(randomize=False)
    x = torch.randn(3 * C + 1, D)
    out = scan(m, tap, x)
    z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    expected = z @ mlp.down_proj.weight.T
    assert torch.equal(out, expected)


def test_conv_grad_blocked_until_wtarget_moves(module_factory):
    """Zero W_target blocks gradient to the conv kernel (LoRA B=0
    dynamics). W_target itself must receive gradient immediately."""
    m, _, tap = module_factory(randomize=False)
    m.train()
    x = torch.randn(2 * C + 2, D)
    tap.current = x.unsqueeze(0)
    m(x.unsqueeze(0)).sum().backward()
    assert m.w_target.grad is not None and m.w_target.grad.abs().sum() > 0
    conv_g = m.target_conv.weight.grad
    assert conv_g is None or conv_g.abs().sum() == 0


# --------------------------------------------------------------- causality --
def test_strict_causality_of_outputs(module_factory):
    """Perturbing token p must leave every output before p unchanged.
    Violation means future leakage, which fabricates eval gains."""
    m, _, tap = module_factory(randomize=True)
    x = torch.randn(3 * C + 2, D)
    base = scan(m, tap, x)
    p = C + 2                       # inside the second chunk
    x2 = x.clone()
    x2[p] += 1.0
    pert = scan(m, tap, x2)
    assert torch.allclose(base[:p], pert[:p], atol=1e-12)
    assert not torch.allclose(base[p:], pert[p:])


def test_first_chunk_never_sees_updates(module_factory):
    """Chunk 0 must be computed with the pristine W0 regardless of how
    wild the targets are."""
    m, mlp, tap = module_factory(randomize=True)
    x = torch.randn(3 * C, D)
    out = scan(m, tap, x)
    z = mlp.act_fn(mlp.gate_proj(x[:C])) * mlp.up_proj(x[:C])
    assert torch.allclose(out[:C], z @ mlp.down_proj.weight.T, atol=1e-12)


# ------------------------------------------------- stream/scan equivalence --
def test_stream_matches_scan(module_factory):
    """The autoregressive streaming path (prefill + token-by-token with
    rolling conv context and chunk commits) must produce exactly what
    the parallel scan produces for the same sequence. This covers the
    EmbeddingTap context buffer, pending-chunk logic, and commit
    boundaries in one property."""
    m, _, tap = module_factory(randomize=True)
    N = 3 * C + 3
    x = torch.randn(N, D)
    expected = scan(m, tap, x)

    m.stateful = True
    tap.stateful = True
    tap.reset_stream()
    m.state = TTTState()
    pieces, i = [], 0
    for size in [C + 1, 1, 1, C - 1, 2]:        # awkward boundaries on purpose
        pieces.append(x[i:i + size])
        i += size
    pieces.append(x[i:])

    outs = []
    for piece in pieces:
        if len(piece) == 0:
            continue
        tap.hook(None, None, piece.unsqueeze(0))   # simulate embed forward
        with torch.no_grad():
            outs.append(m(piece.unsqueeze(0))[0])
    got = torch.cat(outs, dim=0)
    # fp32 staging of committed deltas bounds the error
    assert torch.allclose(got, expected, atol=1e-6)


# ----------------------------------------------------------- evolve switch --
def test_evolve_off_scan_is_plain_mlp(module_factory):
    m, mlp, tap = module_factory(randomize=True)
    m.ttt_evolve = False
    x = torch.randn(3 * C, D)
    out = scan(m, tap, x)
    z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    assert torch.allclose(out, z @ mlp.down_proj.weight.T, atol=1e-12)


def test_evolve_off_stream_applies_but_never_updates(module_factory):
    """evolve=False with imported state must APPLY the state (memory
    kept) while never changing it (learning stopped). This is the
    distinction between stop-learning and forget-everything."""
    m, mlp, tap = module_factory(randomize=True)
    m.stateful, tap.stateful, m.ttt_evolve = True, True, False
    delta = torch.randn(1, D, m.down_proj.weight.shape[1]).float()
    m.state.delta = delta.clone()

    x = torch.randn(2 * C + 1, D)
    tap.hook(None, None, x.unsqueeze(0))
    with torch.no_grad():
        out = m(x.unsqueeze(0))[0]

    z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    expected = z @ mlp.down_proj.weight.T + m.cfg.eta * (
        z @ delta[0].to(z.dtype).T
    )
    assert torch.allclose(out, expected, atol=1e-9)
    assert torch.equal(m.state.delta, delta)        # unchanged
    assert m.state.pending_tokens == 0              # nothing buffered


# ----------------------------------------------------------------- guards --
def test_batch_size_change_mid_session_raises(module_factory):
    m, _, tap = module_factory(randomize=True)
    m.session_mode = True
    scan(m, tap, torch.randn(2 * C, D))
    m.carried_delta = m._next_carried
    with pytest.raises(RuntimeError, match="Batch size changed"):
        tap.current = torch.randn(2, 2 * C, D)
        with torch.no_grad():
            m(torch.randn(2, 2 * C, D))


# --------------------------------------------------------------- clipping --
def test_clip_disabled_is_noop(module_factory):
    m, _, _ = module_factory(randomize=True)
    d = torch.randn(1, 3, D, 16)
    assert torch.equal(m._clip(d), d)


def test_clip_enabled_caps_frobenius_norm(cfg, module_factory):
    clip_cfg = dataclasses.replace(cfg, clip_enabled=True, clip_tau=1e-3,
                                   clip_at_inference_only=True)
    m, _, _ = module_factory(randomize=True, config=clip_cfg)
    big = torch.randn(1, D, 16) * 100
    clipped = m._clip(big)
    assert (clip_cfg.eta * clipped).norm() <= clip_cfg.clip_tau * 1.0001
    m.train()
    assert torch.equal(m._clip(big), big)   # inference-only flag respected
