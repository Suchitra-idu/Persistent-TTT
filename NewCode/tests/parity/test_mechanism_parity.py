"""OLD-vs-NEW numeric parity for the TTT mechanism (Phase 6's done-when).

Both sides run in fp64 on CPU from bit-identical weights, so any disagreement is
a ported bug rather than float noise. The old side is pinned to
`v_source="hidden_state"` — the only mode the new code has (D11).
"""

from __future__ import annotations

import pytest
import torch

from tests.parity import _builders as build

pytestmark = pytest.mark.parity

TOLERANCE = 1e-12

# Stream against scan is not an fp64 comparison: `_commit_chunk` accumulates in
# fp32 on both sides by design (bf16 drifts over thousands of commits), so the
# residual sits at float32 epsilon. Old-vs-new stays exact because both downcast
# identically — which is the whole point of comparing them rather than an oracle.
FP32_TOLERANCE = 1e-6

CHUNK = build.COMMON["chunk_size"]

# Below, at, and above a chunk boundary, plus a length with a short final chunk.
LENGTHS = [1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK, 3 * CHUNK + 2]


def agree(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).detach().abs().max())


class TestScanPath:
    @pytest.mark.parametrize("n_tokens", LENGTHS)
    def test_the_scan_output_agrees(self, n_tokens):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(n_tokens))

        assert agree(old, new) < TOLERANCE

    @pytest.mark.parametrize("n_tokens", LENGTHS)
    def test_a_frozen_fast_weight_agrees(self, n_tokens):
        pair = build.pair()
        pair.set_mode(evolve=False, stream=False, session=False)

        old, new = pair.both(build.hidden(n_tokens))

        assert agree(old, new) < TOLERANCE

    def test_the_scan_actually_contributes_something(self):
        """Without this the parity above would hold on two zeroed branches."""
        pair = build.pair()
        states = build.hidden(3 * CHUNK)

        pair.set_mode(evolve=True, stream=False, session=False)
        evolving, _ = pair.both(states)
        pair.set_mode(evolve=False, stream=False, session=False)
        frozen, _ = pair.both(states)

        assert agree(evolving, frozen) > 1e-6

    @pytest.mark.parametrize("batch", [1, 3])
    def test_it_agrees_across_batch_sizes(self, batch):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK, batch=batch))

        assert agree(old, new) < TOLERANCE


class TestStreamPath:
    @pytest.mark.parametrize("n_tokens", LENGTHS)
    def test_the_stream_output_agrees(self, n_tokens):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=True, session=False)

        old, new = pair.both(build.hidden(n_tokens))

        assert agree(old, new) < TOLERANCE

    def test_the_committed_stream_state_agrees(self):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=True, session=False)

        pair.both(build.hidden(3 * CHUNK))

        assert agree(*pair.stream_states()) < TOLERANCE

    def test_it_agrees_token_by_token(self):
        """Chat streams one token per call, which is where a left-context bug
        would show and a whole-prompt call would not."""
        pair = build.pair()
        pair.set_mode(evolve=True, stream=True, session=False)
        states = build.hidden(3 * CHUNK)

        worst = max(
            agree(*pair.both(states[:, index : index + 1]))
            for index in range(states.shape[1])
        )

        assert worst < TOLERANCE

    def test_a_partial_chunk_leaves_both_uncommitted(self):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=True, session=False)

        pair.both(build.hidden(CHUNK - 1))

        assert pair.stream_states() == (None, None)

    def test_a_frozen_stream_agrees(self):
        pair = build.pair()
        pair.set_mode(evolve=False, stream=True, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE


class TestStreamMatchesScan:
    """Streaming and scanning are the same computation with the clip off.

    The clip is what separates them: scan clips the state it applies without
    feeding it back, the stream clips the state it stores, so with the clip on
    the stream compounds and the scan does not.
    """

    def test_the_old_stream_matches_the_old_scan_without_the_clip(self):
        scan, stream = self._paths(clip_enabled=False)

        assert agree(scan[0], stream[0]) < FP32_TOLERANCE

    def test_the_new_stream_matches_the_new_scan_the_same_way(self):
        scan, stream = self._paths(clip_enabled=False)

        assert agree(scan[1], stream[1]) < FP32_TOLERANCE

    def test_the_two_sides_disagree_with_the_scan_by_the_same_amount(self):
        """Whatever the fp32 commit costs, it costs both sides identically."""
        scan, stream = self._paths(clip_enabled=False)

        assert agree(scan[0], stream[0]) == pytest.approx(
            agree(scan[1], stream[1]), abs=TOLERANCE
        )

    def test_the_residual_is_float32_noise_and_not_a_ported_bug(self):
        scan, stream = self._paths(clip_enabled=False)

        assert agree(scan[1], stream[1]) > TOLERANCE

    def test_the_two_paths_are_not_trivially_identical(self):
        scan, stream = self._paths(clip_enabled=True, clip_tau=1e-6)

        assert agree(scan[0], stream[0]) > 1e-9

    def _paths(self, **overrides):
        states = build.hidden(3 * CHUNK)

        scanning = build.pair(**overrides)
        scanning.set_mode(evolve=True, stream=False, session=False)
        scan = scanning.both(states)

        streaming = build.pair(**overrides)
        streaming.set_mode(evolve=True, stream=True, session=False)
        stream = streaming.both(states)
        return scan, stream


class TestSessionCarry:
    @pytest.mark.parametrize("decay", [1.0, 0.9, 0.0])
    def test_the_staged_carry_agrees(self, decay):
        pair = build.pair(carried_decay=decay)
        pair.set_mode(evolve=True, stream=False, session=True)

        pair.both(build.hidden(3 * CHUNK))
        pair.advance_carry()

        assert agree(*pair.carries()) < TOLERANCE

    @pytest.mark.parametrize("decay", [1.0, 0.9, 0.0])
    def test_the_carry_agrees_after_several_items(self, decay):
        pair = build.pair(carried_decay=decay)
        pair.set_mode(evolve=True, stream=False, session=True)

        for item in range(4):
            pair.both(build.hidden(3 * CHUNK, seed=20 + item))
            pair.advance_carry()

        assert agree(*pair.carries()) < TOLERANCE

    @pytest.mark.parametrize("decay", [1.0, 0.9, 0.0])
    def test_the_output_agrees_once_a_carry_is_installed(self, decay):
        pair = build.pair(carried_decay=decay)
        pair.set_mode(evolve=True, stream=False, session=True)
        pair.both(build.hidden(3 * CHUNK))
        pair.advance_carry()

        old, new = pair.both(build.hidden(3 * CHUNK, seed=31))

        assert agree(old, new) < TOLERANCE

    def test_a_decay_below_one_actually_bounds_the_carry(self):
        """Otherwise the decay cases above would all be the same computation."""
        summed = build.pair(carried_decay=1.0)
        decayed = build.pair(carried_decay=0.5)
        for pair in (summed, decayed):
            pair.set_mode(evolve=True, stream=False, session=True)
            for item in range(4):
                pair.both(build.hidden(3 * CHUNK, seed=20 + item))
                pair.advance_carry()

        assert float(summed.carries()[0].norm()) > float(decayed.carries()[0].norm())

    def test_a_short_item_still_stages_a_carry_on_both_sides(self):
        """The scan's early-out is skipped in session mode; if either side took
        it, that item's delta would silently never reach the carry."""
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=True)

        pair.both(build.hidden(CHUNK - 1))
        pair.advance_carry()

        assert agree(*pair.carries()) < TOLERANCE

    def test_resetting_clears_both(self):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=True)
        pair.both(build.hidden(3 * CHUNK))
        pair.advance_carry()

        pair.reset_carry()

        assert pair.carries() == (None, None)


class TestClip:
    @pytest.mark.parametrize("clip_enabled", [True, False])
    def test_the_scan_agrees_either_way(self, clip_enabled):
        pair = build.pair(clip_enabled=clip_enabled)
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    @pytest.mark.parametrize("clip_enabled", [True, False])
    def test_the_stream_agrees_either_way(self, clip_enabled):
        pair = build.pair(clip_enabled=clip_enabled)
        pair.set_mode(evolve=True, stream=True, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    def test_a_biting_clip_agrees(self):
        pair = build.pair(clip_tau=1e-4)
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    def test_the_clip_is_actually_biting_at_that_tau(self):
        loose = build.pair(clip_tau=1e6)
        tight = build.pair(clip_tau=1e-4)
        for pair in (loose, tight):
            pair.set_mode(evolve=True, stream=False, session=False)

        assert agree(
            loose.both(build.hidden(3 * CHUNK))[0],
            tight.both(build.hidden(3 * CHUNK))[0],
        ) > 1e-9

    def test_inference_only_clipping_agrees_in_training_mode(self):
        pair = build.pair(clip_at_inference_only=True)
        pair.old.train()
        pair.new.train()
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE


class TestGate:
    @pytest.mark.parametrize("output_gate", [True, False])
    def test_the_scan_agrees_either_way(self, output_gate):
        pair = build.pair(output_gate=output_gate)
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    @pytest.mark.parametrize("output_gate", [True, False])
    def test_the_stream_agrees_either_way(self, output_gate):
        pair = build.pair(output_gate=output_gate)
        pair.set_mode(evolve=True, stream=True, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    def test_the_gate_changes_the_output(self):
        gated = build.pair(output_gate=True)
        ungated = build.pair(output_gate=False)
        for pair in (gated, ungated):
            pair.set_mode(evolve=True, stream=False, session=False)

        assert agree(
            gated.both(build.hidden(3 * CHUNK))[0],
            ungated.both(build.hidden(3 * CHUNK))[0],
        ) > 1e-9

    def test_the_gate_std_is_where_the_two_sides_deliberately_differ(self):
        """Chat streams one token per call, and the *sample* std of one element
        is nan — so the old side's per-turn gate diagnostic was nan every turn.
        The new side measures a population std."""
        import math

        pair = build.pair()
        pair.set_mode(evolve=True, stream=True, session=False)
        pair.both(build.hidden(3 * CHUNK))
        pair.both(build.hidden(1, seed=41))

        assert math.isnan(pair.old._gate_std) and pair.new.gate_std == 0.0

    def test_the_gate_std_differs_by_exactly_the_bessel_correction(self):
        import math

        n_tokens = 3 * CHUNK
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=False)

        pair.both(build.hidden(n_tokens))

        assert pair.old._gate_std * math.sqrt(
            (n_tokens - 1) / n_tokens
        ) == pytest.approx(pair.new.gate_std, rel=1e-12)

    def test_the_recorded_gate_mean_agrees(self):
        pair = build.pair()
        pair.set_mode(evolve=True, stream=False, session=False)

        pair.both(build.hidden(3 * CHUNK))

        assert abs(pair.old._gate_mean - pair.new.gate_mean) < TOLERANCE


class TestBidirectionalAblation:
    """Knowingly invalid under next-token prediction, carried pending a research
    call (PLAN §9.2) — so it has to be ported faithfully, not quietly dropped."""

    def test_the_scan_agrees(self):
        pair = build.pair(v_bidirectional=True)
        pair.set_mode(evolve=True, stream=False, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    def test_the_stream_ignores_it_on_both_sides(self):
        pair = build.pair(v_bidirectional=True)
        pair.set_mode(evolve=True, stream=True, session=False)

        old, new = pair.both(build.hidden(3 * CHUNK))

        assert agree(old, new) < TOLERANCE

    def test_it_changes_the_scan_output(self):
        causal = build.pair(v_bidirectional=False)
        both_ways = build.pair(v_bidirectional=True)
        for pair in (causal, both_ways):
            pair.set_mode(evolve=True, stream=False, session=False)

        assert agree(
            causal.both(build.hidden(3 * CHUNK))[0],
            both_ways.both(build.hidden(3 * CHUNK))[0],
        ) > 1e-9


def test_the_two_sides_start_from_identical_weights():
    pair = build.pair()

    assert all(
        torch.equal(old, new)
        for old, new in zip(
            pair.old.state_dict().values(), pair.new.state_dict().values(), strict=True
        )
    )


def test_the_fast_weight_is_not_left_at_its_zero_init():
    """A zeroed w_target makes every TTT term exactly zero, and every parity
    assertion above would pass while comparing nothing."""
    assert float(build.pair().new.w_target.abs().sum()) > 0.0
