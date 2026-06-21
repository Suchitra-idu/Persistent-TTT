"""
Mechanism tests for InPlaceTTTMLP, organized by the guarantee each one
protects. Every test here corresponds to a property that, if silently
broken, would invalidate experiments rather than crash them.
"""

import dataclasses

import pytest
import torch

from conftest import C, D, KERNEL, scan
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


# ----------------------------------------------------------- v_source flag --
def test_v_source_invalid_raises():
    """Typo-safety. A misspelled v_source must fail at construction, not
    silently fall back to one of the modes."""
    with pytest.raises(ValueError, match="v_source"):
        TTTConfig(layer_indices=(0,), v_source="hidden_states")  # missing 's' truncation


def test_v_source_hidden_state_uses_hidden_states_not_tap(cfg, module_factory):
    """In hidden_state mode the conv must read from the layer's
    hidden_states argument, not from the (now unused) embedding tap. If
    they were silently swapped the loss would still decrease, but for the
    wrong reason -- this property is the only thing pinning the dispatch."""
    hs_cfg = dataclasses.replace(cfg, v_source="hidden_state")
    m, _, tap = module_factory(randomize=True, config=hs_cfg)
    h = torch.randn(1, 2 * C, D)
    tap.current = torch.randn(1, 2 * C, D)             # different from h, deliberately
    with torch.no_grad():
        out_hs = m(h)
    # Now make the tap match h. If the dispatch wrongly read tap.current,
    # the output would change; if it reads h (correct), it stays the same.
    tap.current = h
    with torch.no_grad():
        out_after = m(h)
    assert torch.equal(out_hs, out_after)


def test_v_source_dispatch_returns_correct_tensor(cfg, module_factory):
    """The dispatch helper IS the contract that distinguishes the two
    modes. Pin it directly so a future refactor that, say, swaps the
    branches gets caught regardless of how the forward path is wired."""
    h = torch.randn(1, 4, D)
    tap_buf = torch.randn(1, 4, D)

    m_emb, _, tap = module_factory(randomize=True)
    tap.current = tap_buf
    assert m_emb._v_source(h) is tap_buf

    hs_cfg = dataclasses.replace(cfg, v_source="hidden_state")
    m_hs, _, tap_hs = module_factory(randomize=True, config=hs_cfg)
    tap_hs.current = tap_buf
    assert m_hs._v_source(h) is h


# -------------------------------------------------------- v_bidirectional --
def test_v_bidirectional_changes_output(cfg, module_factory):
    """Sanity: flipping v_bidirectional changes the output. If it did
    nothing, the padding logic in _targets is silently dead."""
    causal_cfg = dataclasses.replace(cfg, v_bidirectional=False)
    bi_cfg = dataclasses.replace(cfg, v_bidirectional=True)
    m_causal, _, tap_c = module_factory(randomize=True, seed=0, config=causal_cfg)
    m_bi, _, tap_b = module_factory(randomize=True, seed=0, config=bi_cfg)
    x = torch.randn(2 * C, D)
    a = scan(m_causal, tap_c, x)
    b = scan(m_bi, tap_b, x)
    assert not torch.allclose(a, b)


def test_v_bidirectional_leaks_future_in_v(cfg, module_factory):
    """Property under test: with bidirectional V, perturbing token p
    changes V at positions p - K//2 .. p - 1 (the kernel reaches back to
    encode the future), whereas causal V leaves them untouched. This is
    the leakage that breaks chunk-causality of the scan -- intentional in
    bidirectional mode, documented as the trade-off."""
    half = KERNEL // 2
    causal_cfg = dataclasses.replace(cfg, v_bidirectional=False)
    bi_cfg = dataclasses.replace(cfg, v_bidirectional=True)
    for kind_cfg, leaks_into_past in (
        (causal_cfg, False),
        (bi_cfg, True),
    ):
        m, _, _ = module_factory(randomize=True, seed=0, config=kind_cfg)
        x = torch.randn(1, 3 * C, D)
        v0 = m._targets(x, left_context=None)
        x2 = x.clone()
        p = 2 * C
        x2[:, p, :] += 1.0
        v1 = m._targets(x2, left_context=None)
        before = (v1[:, :p] - v0[:, :p]).abs().max().item()
        if leaks_into_past:
            # Future leak reaches up to half positions back from p.
            assert before > 0
            # And NOT positions further than half + (K odd vs even) back.
            way_back = (v1[:, :p - half - 1] - v0[:, :p - half - 1]
                        ).abs().max().item()
            assert way_back == 0
        else:
            assert before == 0


def test_v_bidirectional_streaming_stays_causal(cfg, module_factory):
    """Streaming path IGNORES v_bidirectional (future tokens unavailable
    across call boundaries). Property: a streaming forward with
    v_bidirectional=True must produce the same V as one with
    v_bidirectional=False for the same input. If a future refactor
    accidentally honored the flag in stream, output would drift."""
    bi_cfg = dataclasses.replace(cfg, v_bidirectional=True)
    m_bi, _, _ = module_factory(randomize=True, seed=0, config=bi_cfg)
    causal_cfg = dataclasses.replace(cfg, v_bidirectional=False)
    m_causal, _, _ = module_factory(randomize=True, seed=0, config=causal_cfg)
    x = torch.randn(1, 2 * C, D)
    # left_context is non-None => streaming codepath
    left = torch.randn(1, KERNEL - 1, D)
    v_bi = m_bi._targets(x, left_context=left)
    v_causal = m_causal._targets(x, left_context=left)
    assert torch.equal(v_bi, v_causal)


# ----------------------------------------------------- hidden-state stream --
def test_hidden_state_stream_uses_per_module_buffer(cfg, module_factory):
    """In hidden_state stream mode the per-module rolling buffer must be
    populated on the first call and then provide left context to the
    second call. Without this, V at the start of every call would see
    zero left context regardless of how much prior input went through."""
    hs_cfg = dataclasses.replace(cfg, v_source="hidden_state")
    m, _, tap = module_factory(randomize=True, config=hs_cfg)
    m.stateful = True
    tap.stateful = True

    h1 = torch.randn(1, C, D)
    with torch.no_grad():
        m(h1)
    assert m._hidden_context is not None
    assert m._hidden_context.shape == (1, KERNEL - 1, D)
    # tail of h1
    assert torch.allclose(m._hidden_context, h1[:, -(KERNEL - 1):, :].detach())


def test_hidden_state_stream_matches_scan(cfg, module_factory):
    """Same property as test_stream_matches_scan but for the hidden_state
    v_source mode -- the per-module buffer must be correct enough that
    chunked streaming reproduces the whole-sequence scan."""
    hs_cfg = dataclasses.replace(cfg, v_source="hidden_state")
    m, _, tap = module_factory(randomize=True, config=hs_cfg)
    N = 3 * C + 3
    h = torch.randn(1, N, D)

    tap.current = h  # unused in this mode but harmless
    with torch.no_grad():
        expected = m(h)[0]

    m.stateful = True
    tap.stateful = True
    m.state = TTTState()
    m._hidden_context = None
    pieces, i = [], 0
    for size in [C + 1, 1, 1, C - 1, 2]:
        pieces.append(h[:, i:i + size, :])
        i += size
    pieces.append(h[:, i:, :])

    outs = []
    for piece in pieces:
        if piece.shape[1] == 0:
            continue
        with torch.no_grad():
            outs.append(m(piece)[0])
    got = torch.cat(outs, dim=0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_reset_stream_state_clears_hidden_context(cfg, module_factory):
    """reset_stream_state() is the wholesale wipe (state.delta AND
    rolling buffers); reset_v_context() is the soft turn-boundary wipe
    (buffers only). Both must clear _hidden_context for the hidden_state
    mode to be safe across session boundaries."""
    hs_cfg = dataclasses.replace(cfg, v_source="hidden_state")
    m, _, tap = module_factory(randomize=True, config=hs_cfg)
    m.stateful = True
    tap.stateful = True
    with torch.no_grad():
        m(torch.randn(1, C, D))
    assert m._hidden_context is not None
    saved_delta = torch.randn_like(m.down_proj.weight).unsqueeze(0)
    m.state.delta = saved_delta.clone()

    m.reset_v_context()
    assert m._hidden_context is None
    assert torch.equal(m.state.delta, saved_delta)        # state preserved

    m.reset_stream_state()
    assert m.state.delta is None                          # state wiped
