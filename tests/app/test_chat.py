from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.app import chat
from ttt.app.chat import FULL, NONE, ChatSession, Sampling, Switches
from ttt.ports.fast_weights import STREAM

GREEDY = Sampling(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=6)
DECAY = 0.5
LETTERS = tuple(ord(c) + 5 for c in "hello!")


def session(*, switches=None, sampling=GREEDY, carries=None, decay=DECAY, **kwargs):
    wiring, generation, tokenizer = _builders.chat_wiring(LETTERS, chunk_size=2)
    return (
        ChatSession(
            generation=generation,
            fast_weights=wiring.fast_weights,
            tokenizer=tokenizer,
            rng=wiring.rng,
            decay=decay,
            switches=switches or Switches(),
            sampling=sampling,
            carries=carries or {},
            **kwargs,
        ),
        wiring,
        generation,
    )


class TestTurn:
    def test_it_decodes_the_tokens_it_generated(self):
        chatter, _, _ = session()

        assert chatter.turn("hi").text == "hello!"

    def test_it_reports_the_tokens_it_generated(self):
        chatter, _, _ = session()

        assert chatter.turn("hi").token_ids == LETTERS

    def test_it_prefills_the_templated_prompt(self):
        chatter, _, generation = session()

        chatter.turn("hi")

        assert generation.prompts[0]

    def test_a_system_prompt_reaches_the_template(self):
        chatter, _, generation = session(system_prompt="be terse")
        tokenizer = _builders.chat_wiring()[2]

        chatter.turn("hi")

        assert "be terse" in tokenizer.decode(list(generation.prompts[0]))

    def test_the_within_turn_cache_does_not_survive_the_turn(self):
        chatter, _, generation = session()

        chatter.turn("one")
        chatter.turn("two")

        assert generation.cache_resets == 3

    def test_thinking_is_split_out_of_the_answer(self):
        thinking = tuple(ord(c) + 5 for c in "<think>why</think>ok")
        wiring, _, tokenizer = _builders.chat_wiring()
        from ttt.adapters.fake_generation import FakeGeneration

        chatter = ChatSession(
            generation=FakeGeneration(thinking, vocab_size=261),
            fast_weights=wiring.fast_weights,
            tokenizer=tokenizer,
            rng=wiring.rng,
            decay=DECAY,
            sampling=Sampling(
                temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=len(thinking)
            ),
        )

        turn = chatter.turn("hi")

        assert (turn.thinking, turn.text) == ("why", "ok")


class TestWithinTurnSwitch:
    def test_on_it_evolves_the_fast_weight_as_tokens_stream(self):
        chatter, wiring, _ = session(switches=Switches(within_turn=True))

        chatter.turn("hi")

        assert wiring.fast_weights.state_ratio(family=STREAM) > 0.0

    def test_off_it_leaves_the_fast_weight_frozen(self):
        chatter, wiring, _ = session(switches=Switches(within_turn=False))

        chatter.turn("hi")

        assert wiring.fast_weights.state_ratio(family=STREAM) == 0.0

    def test_off_it_is_reported_as_no_growth(self):
        chatter, _, _ = session(switches=Switches(within_turn=False))

        assert chatter.turn("hi").diagnostics.state_growth == 0.0


class TestCrossTurnSwitch:
    def test_off_it_leaves_no_residual_state_at_the_boundary(self):
        chatter, wiring, _ = session(switches=Switches(cross_turn=False))

        chatter.turn("hi")

        assert wiring.fast_weights.state_ratio(family=STREAM) == 0.0

    def test_on_it_leaves_the_decayed_state(self):
        chatter, wiring, _ = session(switches=Switches(cross_turn=True))

        during = chatter.turn("hi").diagnostics.state_ratio

        assert wiring.fast_weights.state_ratio(family=STREAM) == pytest.approx(
            DECAY * during
        )

    def test_on_it_compounds_across_turns(self):
        chatter, _, _ = session(switches=Switches(cross_turn=True))

        first = chatter.turn("one").diagnostics.state_ratio
        second = chatter.turn("two").diagnostics.state_ratio

        assert second > first

    def test_off_every_turn_starts_from_the_same_state(self):
        chatter, _, _ = session(switches=Switches(cross_turn=False))

        first = chatter.turn("one").diagnostics.state_ratio
        second = chatter.turn("two").diagnostics.state_ratio

        assert second == pytest.approx(first)

    def test_on_it_clears_the_convolution_left_context(self):
        chatter, wiring, _ = session(switches=Switches(cross_turn=True))

        chatter.turn("hi")

        assert "reset_v_context" in wiring.fast_weights.events

    def test_a_decay_of_one_is_the_unbounded_sum_the_defect_described(self):
        chatter, wiring, _ = session(switches=Switches(cross_turn=True), decay=1.0)

        during = chatter.turn("hi").diagnostics.state_ratio

        assert wiring.fast_weights.state_ratio(family=STREAM) == pytest.approx(during)


class TestSeedSwitch:
    def test_a_named_source_carrier_reaches_the_stream(self):
        chatter, wiring, _ = session(
            switches=Switches(seed="alpha", within_turn=False),
            carries={"alpha": _builders.carry()},
        )

        assert wiring.fast_weights.state_ratio(family=STREAM) > 0.0

    def test_an_unseeded_session_starts_from_zero(self):
        chatter, wiring, _ = session(switches=Switches(within_turn=False))

        assert wiring.fast_weights.state_ratio(family=STREAM) == 0.0

    def test_a_source_with_no_carrier_starts_from_zero(self):
        chatter, wiring, _ = session(
            switches=Switches(seed="gamma", within_turn=False),
            carries={"alpha": _builders.carry()},
        )

        assert wiring.fast_weights.state_ratio(family=STREAM) == 0.0

    def test_a_mismatched_source_can_be_installed_on_purpose(self):
        chatter, wiring, _ = session(
            switches=Switches(seed="beta", within_turn=False),
            carries={"beta": _builders.carry(value=9.0)},
        )

        assert wiring.fast_weights.state_ratio(family=STREAM) == pytest.approx(9.0)

    def test_the_seed_does_not_land_in_the_carry_family(self):
        chatter, wiring, _ = session(
            switches=Switches(seed="alpha"), carries={"alpha": _builders.carry()}
        )

        assert wiring.fast_weights.snapshot().is_empty

    def test_a_reset_reinstalls_the_seed(self):
        chatter, wiring, _ = session(
            switches=Switches(seed="alpha", within_turn=False),
            carries={"alpha": _builders.carry()},
        )
        chatter.turn("hi")

        chatter.reset()

        assert wiring.fast_weights.state_ratio(family=STREAM) > 0.0


class TestContextControl:
    def test_none_shows_the_model_no_earlier_turn(self):
        chatter, _, generation = session(switches=Switches(context=NONE))
        tokenizer = _builders.chat_wiring()[2]

        chatter.turn("apricot")
        chatter.turn("banana")

        assert "apricot" not in tokenizer.decode(list(generation.prompts[1]))

    def test_full_shows_the_model_the_conversation(self):
        chatter, _, generation = session(switches=Switches(context=FULL))
        tokenizer = _builders.chat_wiring()[2]

        chatter.turn("apricot")
        chatter.turn("banana")

        assert "apricot" in tokenizer.decode(list(generation.prompts[1]))

    def test_history_is_recorded_whatever_the_control_says(self):
        chatter, _, _ = session(switches=Switches(context=NONE))

        chatter.turn("hi")

        assert [message["role"] for message in chatter.history] == [
            "user",
            "assistant",
        ]

    def test_an_unknown_context_is_rejected(self):
        with pytest.raises(ValueError, match="unknown context"):
            Switches(context="some")


class TestDiagnostics:
    def test_it_reports_the_partial_chunk(self):
        chatter, _, _ = session()

        assert chatter.turn("hi").diagnostics.chunk_size == 2

    def test_it_reports_the_gate_statistics(self):
        chatter, _, _ = session()

        diagnostics = chatter.turn("hi").diagnostics

        assert (diagnostics.gate_mean, diagnostics.gate_std) == (0.5, 0.0)

    def test_growth_is_the_share_of_the_state_this_turn_added(self):
        chatter, _, _ = session(switches=Switches(cross_turn=True))

        assert chatter.turn("hi").diagnostics.state_growth == pytest.approx(1.0)

    def test_growth_falls_once_a_state_has_accumulated(self):
        chatter, _, _ = session(switches=Switches(cross_turn=True))
        chatter.turn("one")

        assert chatter.turn("two").diagnostics.state_growth < 1.0


class TestSameSeedAB:
    def test_identical_switches_and_seed_give_identical_streams(self):
        sampling = Sampling(temperature=1.0, top_p=1.0, max_new_tokens=6, seed=99)
        arms = [session(sampling=sampling)[0] for _ in range(2)]

        first, second = chat.ab_turn("hi", arms)

        assert first.token_ids == second.token_ids

    def test_the_same_arm_redraws_identically_after_a_reset(self):
        sampling = Sampling(temperature=1.0, top_p=1.0, max_new_tokens=6, seed=99)
        chatter, _, _ = session(sampling=sampling)

        first = chatter.turn("hi").token_ids
        chatter.reset()

        assert chatter.turn("hi").token_ids == first

    def test_each_turn_draws_from_its_own_generator(self):
        sampling = Sampling(temperature=50.0, top_p=1.0, max_new_tokens=6, seed=99)
        wiring, _, tokenizer = _builders.chat_wiring()
        from ttt.adapters.fake_generation import FakeGeneration

        chatter = ChatSession(
            generation=FakeGeneration(range(32), vocab_size=32),
            fast_weights=wiring.fast_weights,
            tokenizer=tokenizer,
            rng=wiring.rng,
            decay=DECAY,
            sampling=sampling,
        )

        assert chatter.turn("one").token_ids != chatter.turn("two").token_ids

    def test_an_unseeded_session_asks_for_no_generator(self):
        chatter, _, _ = session(sampling=GREEDY)

        assert chatter._generator() is None


def test_a_reset_clears_the_conversation():
    chatter, _, _ = session()
    chatter.turn("hi")

    chatter.reset()

    assert (chatter.history, chatter.turns) == ([], 0)


@pytest.mark.parametrize(
    "switches",
    [
        Switches(within_turn=False),
        Switches(cross_turn=False),
        Switches(seed="alpha"),
        Switches(context=FULL),
    ],
)
def test_every_switch_setting_still_produces_a_turn(switches):
    chatter, _, _ = session(switches=switches, carries={"alpha": _builders.carry()})

    assert chatter.turn("hi").text == "hello!"
