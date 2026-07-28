"""Nucleus filtering, stop tokens, and chat-string splitting."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core._builders import generator
from ttt.core import sampling_text

LOGITS = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])


def _logits(*values: float) -> torch.Tensor:
    return torch.tensor([list(values)])


@given(
    top_p=st.floats(min_value=0.0, max_value=1.0),
    top_k=st.integers(min_value=0, max_value=5),
    temperature=st.floats(min_value=0.05, max_value=5.0),
)
@settings(max_examples=60, deadline=None)
def test_the_filtered_mass_is_a_probability_distribution(top_p, top_k, temperature):
    probs, _ = sampling_text.nucleus_filter(
        LOGITS, temperature=temperature, top_p=top_p, top_k=top_k
    )

    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(probs.min()) >= 0.0


@given(top_p=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=40, deadline=None)
def test_the_top_token_always_survives_the_nucleus_cut(top_p):
    """At top_p=0 the shift is the only thing standing between us and nothing."""
    probs, indices = sampling_text.nucleus_filter(
        LOGITS, temperature=1.0, top_p=top_p, top_k=0
    )

    assert float(probs[0, 0]) > 0.0
    assert int(indices[0, 0]) == 0


def test_top_p_of_zero_leaves_exactly_one_token():
    probs, _ = sampling_text.nucleus_filter(LOGITS, temperature=1.0, top_p=0.0)

    assert int((probs > 0).sum()) == 1


def test_top_k_truncates_to_k_tokens():
    probs, _ = sampling_text.nucleus_filter(
        LOGITS, temperature=1.0, top_p=1.0, top_k=2
    )

    assert int((probs > 0).sum()) == 2


def test_top_k_larger_than_the_vocabulary_keeps_everything():
    probs, _ = sampling_text.nucleus_filter(
        LOGITS, temperature=1.0, top_p=1.0, top_k=500
    )

    assert int((probs > 0).sum()) == LOGITS.shape[-1]


def test_a_low_temperature_sharpens_the_distribution():
    sharp, _ = sampling_text.nucleus_filter(LOGITS, temperature=0.1, top_p=1.0)
    flat, _ = sampling_text.nucleus_filter(LOGITS, temperature=2.0, top_p=1.0)

    assert float(sharp[0, 0]) > float(flat[0, 0])


def test_indices_are_returned_in_descending_probability_order():
    _, indices = sampling_text.nucleus_filter(
        _logits(0.0, 5.0, 1.0), temperature=1.0, top_p=1.0
    )

    assert indices[0].tolist() == [1, 2, 0]


def test_filtering_rejects_a_temperature_of_zero():
    """Greedy is `next_token`'s job; a zero here would be a division by zero."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        sampling_text.nucleus_filter(LOGITS, temperature=0.0, top_p=1.0)


def test_a_non_positive_temperature_decodes_greedily():
    picked = sampling_text.next_token(
        _logits(1.0, 9.0, 2.0), temperature=0.0, top_p=1.0
    )

    assert picked == 1


def test_sampling_is_reproducible_under_the_same_generator_seed():
    """The same-seed A/B (D14) is only meaningful if this holds."""
    first = sampling_text.next_token(
        LOGITS, temperature=1.0, top_p=0.95, top_k=20, generator=generator(11)
    )
    second = sampling_text.next_token(
        LOGITS, temperature=1.0, top_p=0.95, top_k=20, generator=generator(11)
    )

    assert first == second


def test_top_p_of_zero_forces_the_argmax_even_when_sampling():
    picked = sampling_text.next_token(
        _logits(1.0, 9.0, 2.0), temperature=1.0, top_p=0.0, generator=generator(3)
    )

    assert picked == 1


def test_stop_ids_collect_eos_pad_and_the_chat_markers():
    ids = sampling_text.stop_token_ids(
        eos_token_id=2,
        pad_token_id=0,
        special_token_ids={"<|im_end|>": 151645, "<|endoftext|>": 151643},
    )

    assert ids == frozenset({0, 2, 151645, 151643})


def test_a_missing_pad_id_is_skipped_rather_than_defaulted_to_zero():
    """Token 0 is a real vocabulary token; stopping on it truncates turns."""
    ids = sampling_text.stop_token_ids(eos_token_id=2, pad_token_id=None)

    assert ids == frozenset({2})


def test_a_special_token_that_maps_to_unk_is_not_a_stop_token():
    ids = sampling_text.stop_token_ids(
        eos_token_id=2,
        special_token_ids={"<|im_end|>": 99},
        unk_token_id=99,
    )

    assert ids == frozenset({2})


def test_a_special_token_the_tokenizer_does_not_know_is_skipped():
    ids = sampling_text.stop_token_ids(
        eos_token_id=2, special_token_ids={"<|im_end|>": None}
    )

    assert ids == frozenset({2})


def test_chat_markers_are_stripped_from_decoded_text():
    stripped = sampling_text.strip_chat_specials("hi<|im_end|> there<|endoftext|>")

    assert stripped == "hi there"


def test_thinking_is_split_out_when_both_tags_are_present():
    thinking, answer = sampling_text.split_thinking("<think> hmm </think>the answer")

    assert (thinking, answer) == ("hmm", "the answer")


def test_an_unclosed_thinking_tag_leaves_the_whole_reply_as_the_answer():
    """Swallowing everything after a dangling opener would lose the response."""
    thinking, answer = sampling_text.split_thinking("<think>hmm, no closer")

    assert (thinking, answer) == ("", "<think>hmm, no closer")


def test_text_without_thinking_tags_is_all_answer():
    assert sampling_text.split_thinking("plain reply") == ("", "plain reply")


def test_a_closing_tag_before_an_opening_one_is_not_a_split():
    thinking, answer = sampling_text.split_thinking("</think>stray")

    assert (thinking, answer) == ("", "</think>stray")
