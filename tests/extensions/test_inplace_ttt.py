"""The TTT module over the core kernels: identity at init, the two paths agree,
and the three resets touch three different state families.
"""

from __future__ import annotations

import pytest
import torch

from ttt.extensions.mechanism import InPlaceTTTMLP, iter_ttt_modules
from tests.extensions import _builders

CHUNK = 4
LONG = 12
SCALE = 0.3
CONV = 0.5


def base_mlp_out(module, h):
    z = module.act_fn(module.gate_proj(h)) * module.up_proj(h)
    return module.down_proj(z)


def scanning(**overrides):
    return _builders.ttt_module(chunk_size=CHUNK, **overrides)


def test_a_freshly_built_module_is_identity_to_the_base_mlp():
    module = scanning()
    h = _builders.hidden_states(LONG)

    assert torch.allclose(module(h), base_mlp_out(module, h), atol=1e-6)


def test_a_module_with_evolution_off_takes_the_plain_mlp_path():
    module = scanning(w_target_scale=SCALE)
    module.evolve = False
    h = _builders.hidden_states(LONG)

    assert torch.equal(module(h), base_mlp_out(module, h))


def test_a_single_chunk_item_outside_a_session_skips_the_scan():
    module = scanning(w_target_scale=SCALE)
    h = _builders.hidden_states(CHUNK)

    module(h)

    assert module.gate_mean is None


def test_a_single_chunk_item_inside_a_session_still_stages_a_carry():
    module = scanning(w_target_scale=SCALE)
    module.session_mode = True
    module(_builders.hidden_states(CHUNK))

    module.advance_carry()

    assert module.carried is not None and module.carried.abs().sum() > 0


def test_the_stream_path_matches_the_scan_path():
    scan_module = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    stream_module = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    stream_module.stateful = True
    h = _builders.hidden_states(LONG)

    assert torch.allclose(scan_module(h), stream_module(h), atol=1e-5)


def test_the_stream_path_matches_the_scan_path_across_call_boundaries():
    scan_module = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    stream_module = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    stream_module.stateful = True
    h = _builders.hidden_states(LONG)

    streamed = torch.cat(
        [stream_module(h[:, i : i + 2]) for i in range(0, LONG, 2)], dim=1
    )

    assert torch.allclose(scan_module(h), streamed, atol=1e-5)


def test_the_conv_left_context_spans_a_call_boundary():
    whole = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    split = scanning(w_target_scale=SCALE, conv_scale=CONV, clip_enabled=False)
    whole.stateful = split.stateful = True
    h = _builders.hidden_states(LONG)

    piecewise = torch.cat([split(h[:, :2]), split(h[:, 2:])], dim=1)

    assert torch.allclose(whole(h), piecewise, atol=1e-5)


def test_a_freshly_built_delta_module_is_identity_to_the_base_mlp():
    module = scanning(update_rule="delta")
    h = _builders.hidden_states(LONG)

    assert torch.allclose(module(h), base_mlp_out(module, h), atol=1e-6)


def test_a_single_chunk_delta_item_still_evolves_within_itself():
    """Where the hebbian path skips a lone chunk entirely (see
    `test_a_single_chunk_item_outside_a_session_skips_the_scan`), delta's
    per-token causality means it can't: token 1 already sees token 0's write."""
    module = scanning(update_rule="delta", w_target_scale=SCALE)
    h = _builders.hidden_states(CHUNK)

    module(h)

    assert module.gate_mean is not None


def test_the_delta_rule_actually_contributes_something():
    frozen = scanning(update_rule="delta", w_target_scale=SCALE)
    evolving = scanning(update_rule="delta", w_target_scale=SCALE)
    frozen.evolve = False
    h = _builders.hidden_states(LONG)

    assert not torch.allclose(evolving(h), frozen(h), atol=1e-6)


def test_a_freshly_built_delta_chunk_module_is_identity_to_the_base_mlp():
    module = scanning(update_rule="delta_chunk")
    h = _builders.hidden_states(LONG)

    assert torch.allclose(module(h), base_mlp_out(module, h), atol=1e-6)


def test_a_single_chunk_delta_chunk_item_outside_a_session_skips_the_scan():
    """Unlike "delta", "delta_chunk" freezes the whole first chunk at zero,
    same as "hebbian" — the shortcut in `_scan_forward` still applies."""
    module = scanning(update_rule="delta_chunk", w_target_scale=SCALE)
    h = _builders.hidden_states(CHUNK)

    module(h)

    assert module.gate_mean is None


def test_the_delta_chunk_rule_actually_contributes_something():
    frozen = scanning(update_rule="delta_chunk", w_target_scale=SCALE)
    evolving = scanning(update_rule="delta_chunk", w_target_scale=SCALE)
    frozen.evolve = False
    h = _builders.hidden_states(LONG)

    assert not torch.allclose(evolving(h), frozen(h), atol=1e-6)


def test_truncate_every_one_starves_w_target_of_gradient():
    """The bug a real run surfaced: v only ever shapes a *later* read, never
    its own chunk's, so fully truncating the recurrence (the only setting
    safe from the backward-chain instability on its own) leaves w_target —
    and anything else that only shapes v — with no path to the loss at all."""
    module = scanning(update_rule="delta_chunk", w_target_scale=SCALE, truncate_every=1)
    h = _builders.hidden_states(LONG)

    module(h).sum().backward()

    assert module.w_target.grad is None or module.w_target.grad.abs().sum() == 0


def test_a_wider_truncation_window_lets_w_target_learn():
    module = scanning(update_rule="delta_chunk", w_target_scale=SCALE, truncate_every=5)
    h = _builders.hidden_states(LONG)

    module(h).sum().backward()

    assert module.w_target.grad is not None
    assert module.w_target.grad.abs().sum() > 0


def test_a_zero_decay_carry_keeps_only_the_last_items_delta():
    first, second = _builders.hidden_states(LONG), _builders.hidden_states(LONG, seed=8)
    both = scanning(w_target_scale=SCALE, carried_decay=0.0)
    last_only = scanning(w_target_scale=SCALE, carried_decay=0.0)
    both.session_mode = last_only.session_mode = True

    for h in (first, second):
        both(h)
        both.advance_carry()
    last_only(second)
    last_only.advance_carry()

    assert torch.allclose(both.carried, last_only.carried, atol=1e-6)


def test_a_carry_decay_of_one_sums_every_items_delta():
    first, second = _builders.hidden_states(LONG), _builders.hidden_states(LONG, seed=8)
    both, only_first, only_second = (
        scanning(w_target_scale=SCALE, carried_decay=1.0) for _ in range(3)
    )
    for module in (both, only_first, only_second):
        module.session_mode = True

    for h in (first, second):
        both(h)
        both.advance_carry()
    only_first(first)
    only_first.advance_carry()
    only_second(second)
    only_second.advance_carry()

    assert torch.allclose(
        both.carried, only_first.carried + only_second.carried, atol=1e-6
    )


def test_a_carry_survives_only_once_advance_is_called():
    module = scanning(w_target_scale=SCALE)
    module.session_mode = True

    module(_builders.hidden_states(LONG))

    assert module.carried is None


def test_a_batch_size_change_mid_session_is_an_error():
    module = scanning(w_target_scale=SCALE)
    module.session_mode = True
    module(_builders.hidden_states(LONG))
    module.advance_carry()

    with pytest.raises(RuntimeError, match="batch size changed"):
        module(_builders.hidden_states(LONG, batch=2))


def test_resetting_the_carry_leaves_the_stream_delta_alone():
    module = scanning(w_target_scale=SCALE)
    module.stateful = True
    module(_builders.hidden_states(LONG))

    module.reset_carry()

    assert module.stream.delta is not None


def test_resetting_the_stream_leaves_the_carry_alone():
    module = scanning(w_target_scale=SCALE)
    module.session_mode = True
    module(_builders.hidden_states(LONG))
    module.advance_carry()

    module.reset_stream()

    assert module.carried is not None


def test_resetting_the_conv_context_leaves_the_stream_delta_alone():
    module = scanning(w_target_scale=SCALE)
    module.stateful = True
    module(_builders.hidden_states(LONG))

    module.reset_v_context()

    assert module.stream.delta is not None and module._hidden_context is None


def test_a_partially_filled_chunk_is_pending_rather_than_committed():
    module = scanning(w_target_scale=SCALE)
    module.stateful = True

    module(_builders.hidden_states(CHUNK - 1))

    assert module.stream.delta is None and module.stream.pending_tokens == CHUNK - 1


def test_clipping_bounds_the_streamed_state():
    tau = 1e-4
    module = scanning(w_target_scale=SCALE, clip_enabled=True, clip_tau=tau)
    module.stateful = True
    module.eval()

    module(_builders.hidden_states(LONG))

    assert float((module.cfg.eta * module.stream.delta).norm()) <= tau * (1 + 1e-6)


def test_the_gate_records_its_mean_and_spread():
    module = scanning(w_target_scale=SCALE)

    module(_builders.hidden_states(LONG))

    assert module.gate_mean is not None and module.gate_std is not None


def test_a_disabled_gate_records_nothing():
    module = scanning(w_target_scale=SCALE, output_gate=False)

    module(_builders.hidden_states(LONG))

    assert module.gate_mean is None


def test_a_disabled_gate_leaves_the_ttt_term_unscaled():
    gated = scanning(w_target_scale=SCALE)
    ungated = scanning(w_target_scale=SCALE, output_gate=False)
    h = _builders.hidden_states(LONG)

    assert not torch.allclose(gated(h), ungated(h), atol=1e-6)


def test_iter_ttt_modules_finds_every_patched_layer():
    holder = torch.nn.ModuleList([scanning(), scanning(), _builders.FakeMLP(6, 8, seed=1)])

    assert len(list(iter_ttt_modules(holder))) == 2


def test_the_module_is_a_drop_in_for_the_mlp_it_replaced():
    original = _builders.FakeMLP(6, 8, seed=1)

    module = InPlaceTTTMLP(original, 6, scanning().cfg)

    assert module.down_proj is original.down_proj
