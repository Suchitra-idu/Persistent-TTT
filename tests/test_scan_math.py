"""
Numerical verification of the chunk-wise scan path against a naive
sequential apply-then-update reference. Runs locally on CPU in seconds,
no Modal, no GPU, no Qwen download.

Run after ANY change to InPlaceTTTMLP:

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python tests/test_scan_math.py

Checks
  1. per-document mode  == sequential loop with per-doc resets
  2. session carry mode == sequential loop with persistent state
"""

import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import inplace_ttt as I
from inplace_ttt import EmbeddingTap, InPlaceTTTMLP
from ttt_config import TTTConfig

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D, DFF, C = 8, 16, 4
CFG = TTTConfig(layer_indices=(0,), chunk_size=C, eta=0.05,
                conv_kernel_size=3, normalize_delta_by_chunk=True,
                v_source="embedding", v_bidirectional=False)


def build_module():
    mlp = types.SimpleNamespace(
        gate_proj=torch.nn.Linear(D, DFF, bias=False),
        up_proj=torch.nn.Linear(D, DFF, bias=False),
        down_proj=torch.nn.Linear(DFF, D, bias=False),
        act_fn=torch.nn.SiLU(),
    )
    tap = EmbeddingTap(CFG.conv_kernel_size)
    m = InPlaceTTTMLP(mlp, D, CFG, tap)
    # w_target is zero-init by design; randomize so updates are nonzero
    torch.nn.init.normal_(m.w_target, std=0.5)
    torch.nn.init.normal_(m.target_conv.weight, std=0.5)
    m.eval()
    return m, mlp, tap


def reference(m, mlp, papers, carry_across):
    """Naive sequential apply-then-update over chunks."""
    W0 = mlp.down_proj.weight.clone()
    S = torch.zeros_like(W0)
    outs = []
    for x0 in papers:
        if not carry_across:
            S = torch.zeros_like(W0)
        z = mlp.act_fn(mlp.gate_proj(x0)) * mlp.up_proj(x0)
        xp = torch.nn.functional.pad(
            x0.T.unsqueeze(0), (CFG.conv_kernel_size - 1, 0)
        )
        v = m.target_conv(xp)[0].T @ m.w_target
        out = torch.zeros(z.shape[0], D)
        for s in range(0, z.shape[0], C):
            zc, vc = z[s:s + C], v[s:s + C]
            out[s:s + C] = zc @ (W0 + CFG.eta * S).T   # apply
            # Divide by actual non-padded chunk size (not constant C) so the
            # last partial chunk isn't silently under-normalized. Matches
            # the chunked implementation's per-chunk-size division.
            S = S + (vc.T @ zc) / max(vc.shape[0], 1)  # then update
        outs.append(out)
    return outs


def run_module(m, tap, papers, session):
    m.session_mode = session
    m.carried_delta = None
    m._next_carried = None
    outs = []
    for x0 in papers:
        tap.current = x0.unsqueeze(0)
        with torch.no_grad():
            outs.append(m(x0.unsqueeze(0))[0])
        if session and m._next_carried is not None:
            m.carried_delta = m._next_carried      # advance_session_state
            m._next_carried = None
    return outs


def max_err(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def _papers():
    torch.manual_seed(1)
    return [torch.randn(10, D), torch.randn(7, D), torch.randn(13, D)]


def test_scan_matches_sequential_reference_per_document():
    m, mlp, tap = build_module()
    papers = _papers()
    e = max_err(reference(m, mlp, papers, False),
                run_module(m, tap, papers, False))
    assert e < 1e-9, f"per-document scan diverges, max err {e:.2e}"


def test_scan_matches_sequential_reference_session_carry():
    m, mlp, tap = build_module()
    papers = _papers()
    e = max_err(reference(m, mlp, papers, True),
                run_module(m, tap, papers, True))
    # fp32 staging of carried state bounds this around 1e-7 in fp64 land
    assert e < 1e-6, f"session carry diverges, max err {e:.2e}"


if __name__ == "__main__":
    test_scan_matches_sequential_reference_per_document()
    test_scan_matches_sequential_reference_session_carry()
    print("PASS, scan path matches the sequential reference in both modes")
