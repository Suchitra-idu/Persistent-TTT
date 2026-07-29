from __future__ import annotations

import pytest

from tests.app import _builders as app_builders
from ttt.app.chat import FULL, NONE, ChatSession, Sampling, Switches
from ttt.experiments import chat_repl
from ttt.experiments.chat_repl import AB, DIAG, MULTILINE, QUIT, RESET, SAY, SWITCH

GREEDY = Sampling(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=6)
LETTERS = tuple(ord(c) + 5 for c in "hello!")


def session(**kwargs) -> ChatSession:
    wiring, generation, tokenizer = app_builders.chat_wiring(LETTERS, chunk_size=2)
    return ChatSession(
        generation=generation,
        fast_weights=wiring.fast_weights,
        tokenizer=tokenizer,
        rng=wiring.rng,
        decay=0.5,
        sampling=GREEDY,
        **kwargs,
    )


class TestParse:
    def test_a_bare_line_is_a_turn(self):
        assert chat_repl.parse("hello there") == chat_repl.Command(SAY, "hello there")

    def test_a_slash_is_a_command(self):
        assert chat_repl.parse("/quit").kind == QUIT

    def test_commands_have_short_forms(self):
        assert (chat_repl.parse("/q").kind, chat_repl.parse("/r").kind) == (QUIT, RESET)

    def test_a_command_keeps_its_argument(self):
        assert chat_repl.parse("/ab what is a carry") == chat_repl.Command(
            AB, "what is a carry"
        )

    def test_a_switch_keeps_both_words(self):
        assert chat_repl.parse("/switch cross_turn off").text == "cross_turn off"

    def test_an_unknown_command_is_rejected(self):
        with pytest.raises(ValueError, match="unknown command"):
            chat_repl.parse("/nonsense")

    def test_a_pasted_prompt_prefix_is_stripped(self):
        assert chat_repl.parse("you> hello").text == "hello"

    def test_a_pasted_reply_prefix_is_stripped(self):
        assert chat_repl.parse("bot> hello").text == "hello"

    def test_an_empty_line_is_an_empty_turn(self):
        assert chat_repl.parse("   ") == chat_repl.Command(SAY, "")

    def test_multiline_is_recognised(self):
        assert chat_repl.parse("/m").kind == MULTILINE

    def test_diagnostics_are_recognised(self):
        assert chat_repl.parse("/diag").kind == DIAG


class TestApplySwitch:
    def test_a_toggle_turns_off(self):
        assert chat_repl.apply_switch(Switches(), "cross_turn off").cross_turn is False

    def test_a_toggle_turns_on(self):
        switched = chat_repl.apply_switch(Switches(within_turn=False), "within_turn on")

        assert switched.within_turn is True

    @pytest.mark.parametrize("value", ["on", "true", "1"])
    def test_every_spelling_of_on_is_accepted(self, value):
        assert chat_repl.apply_switch(Switches(), f"cross_turn {value}").cross_turn

    def test_a_bad_toggle_value_is_rejected(self):
        with pytest.raises(ValueError, match="takes on or off"):
            chat_repl.apply_switch(Switches(), "cross_turn maybe")

    def test_the_context_control_is_settable(self):
        assert chat_repl.apply_switch(Switches(), "context full").context == FULL

    def test_an_unknown_context_is_rejected(self):
        with pytest.raises(ValueError, match="context must be one of"):
            chat_repl.apply_switch(Switches(), "context sometimes")

    def test_the_seed_takes_a_source_name(self):
        assert chat_repl.apply_switch(Switches(), "seed RedPajamaBook").seed == (
            "RedPajamaBook"
        )

    def test_an_empty_seed_clears_it(self):
        assert chat_repl.apply_switch(Switches(seed="x"), "seed").seed == ""

    def test_an_unknown_switch_is_rejected(self):
        with pytest.raises(ValueError, match="unknown switch"):
            chat_repl.apply_switch(Switches(), "carry on")

    def test_it_leaves_the_other_switches_alone(self):
        switched = chat_repl.apply_switch(Switches(seed="x"), "cross_turn off")

        assert switched.seed == "x"


class TestControlArm:
    def test_the_control_switches_the_carry_off_entirely(self):
        control = chat_repl.control_for(Switches())

        assert (control.within_turn, control.cross_turn) == (False, False)

    def test_the_control_keeps_the_context_setting(self):
        control = chat_repl.control_for(Switches(context=FULL))

        assert control.context == FULL

    def test_the_control_keeps_the_seed(self):
        assert chat_repl.control_for(Switches(seed="alpha")).seed == "alpha"


class TestAbLines:
    def test_it_reports_both_arms(self):
        chatter = session()

        lines = chat_repl.ab_lines("hi", chatter, chat_repl.control_for(Switches()))

        assert len(lines) == 6

    def test_each_arm_is_labelled_with_its_switches(self):
        chatter = session()

        lines = chat_repl.ab_lines("hi", chatter, chat_repl.control_for(Switches()))

        assert "cross_turn=True" in lines[0] and "cross_turn=False" in lines[3]

    def test_it_restores_the_switches_it_was_given(self):
        chatter = session(switches=Switches(seed="alpha"))

        chat_repl.ab_lines("hi", chatter, chat_repl.control_for(chatter.switches))

        assert chatter.switches == Switches(seed="alpha")

    def test_it_leaves_the_session_reset(self):
        chatter = session()

        chat_repl.ab_lines("hi", chatter, chat_repl.control_for(Switches()))

        assert chatter.turns == 0

    def test_the_control_arm_leaves_no_state(self):
        chatter = session()

        lines = chat_repl.ab_lines("hi", chatter, chat_repl.control_for(Switches()))

        assert "state/W0 0.000e+00" in lines[5]

    def test_it_restores_the_switches_even_when_a_turn_raises(self):
        chatter = session(switches=Switches(seed="alpha"))
        chatter.turn = _explode

        with pytest.raises(ZeroDivisionError):
            chat_repl.ab_lines("hi", chatter, chat_repl.control_for(chatter.switches))

        assert chatter.switches == Switches(seed="alpha")


def _explode(_message):
    raise ZeroDivisionError("turn blew up")


class TestFormatting:
    def test_diagnostics_name_the_four_numbers(self):
        turn = session().turn("hi")

        line = chat_repl.format_diagnostics(turn)

        assert "state/W0" in line and "pending" in line and "gate" in line

    def test_diagnostics_survive_a_model_with_no_gate(self):
        chatter = session()
        chatter.fast_weights.gate_stats = lambda: None

        assert "gate" not in chat_repl.format_diagnostics(chatter.turn("hi"))

    def test_the_switch_line_names_every_switch(self):
        line = chat_repl.format_switches(Switches())

        assert all(
            name in line for name in ("within_turn", "cross_turn", "seed", "context")
        )

    def test_an_unset_seed_reads_as_none(self):
        assert "(none)" in chat_repl.format_switches(Switches())


class TestArguments:
    def test_the_switches_come_off_the_command_line(self):
        args = chat_repl.build_parser().parse_args(
            ["--no-cross-turn", "--context", "full", "--seed-source", "alpha"]
        )

        assert chat_repl.switches_from(args) == Switches(
            within_turn=True, cross_turn=False, seed="alpha", context=FULL
        )

    def test_the_defaults_run_the_full_mechanism_with_no_context(self):
        args = chat_repl.build_parser().parse_args([])

        assert chat_repl.switches_from(args) == Switches(
            within_turn=True, cross_turn=True, seed="", context=NONE
        )

    def test_thinking_defaults_off(self):
        assert chat_repl.build_parser().parse_args([]).thinking is False

    def test_the_sampling_seed_reaches_the_sampler(self):
        args = chat_repl.build_parser().parse_args(["--sampling-seed", "99"])

        assert chat_repl.sampling_from(args).seed == 99

    def test_an_unset_sampling_seed_leaves_the_draw_free(self):
        assert chat_repl.sampling_from(chat_repl.build_parser().parse_args([])).seed is None

    def test_an_unknown_context_is_rejected_at_the_command_line(self):
        with pytest.raises(SystemExit):
            chat_repl.build_parser().parse_args(["--context", "sometimes"])
