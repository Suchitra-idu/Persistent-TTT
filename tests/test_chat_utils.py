"""
Chat-mode helper tests. Pure CPU torch, no model, no Modal.

The integration code (KV cache lifecycle, fast-weight persistence) lives
in infer_modal.py and can't be tested without a real model -- but the
pure helpers below cover every choice that wouldn't be obvious from
reading the code: top-1 retention at top_p=0, greedy-at-T=0, stop-token
assembly. Prompt format is now produced by tokenizer.apply_chat_template
inside infer_modal.py so there is no standalone formatter to test.
"""

import sys
import types

import torch

from chat_utils import chat_stop_token_ids, sample_top_p


# ----------------------------------------------------------- sampling --
def test_sample_top_p_greedy_at_temperature_zero():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    # Run several times; temperature=0 must be deterministic.
    picks = {sample_top_p(logits, temperature=0.0, top_p=0.9)
             for _ in range(10)}
    assert picks == {1}


def test_sample_top_p_negative_temperature_is_also_greedy():
    """temperature <= 0 takes the greedy branch unconditionally."""
    logits = torch.tensor([[0.1, 0.2, 5.0]])
    assert sample_top_p(logits, temperature=-1.0, top_p=0.5) == 2


def test_sample_top_p_returns_top1_when_top_p_is_zero():
    """top_p=0 must still pick SOME single token (the argmax), even
    though every sorted_prob > 0 -- the shift-by-one in the mask is
    what keeps the top-1 in the nucleus. With a clear argmax this is
    deterministic regardless of sort stability."""
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])  # argmax = 1
    picks = {sample_top_p(logits, temperature=1.0, top_p=0.0)
             for _ in range(20)}
    assert picks == {1}


def test_sample_top_p_returns_id_in_vocab_range():
    torch.manual_seed(1)
    logits = torch.randn(1, 32)
    for _ in range(50):
        tid = sample_top_p(logits, temperature=1.0, top_p=0.9)
        assert 0 <= tid < 32


def test_sample_top_p_concentrated_logits_always_pick_argmax():
    """One token has > top_p mass on its own: must always be sampled."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    picks = {sample_top_p(logits, temperature=1.0, top_p=0.5)
             for _ in range(50)}
    assert picks == {0}


def test_sample_top_p_uniform_explores_multiple_tokens():
    """With uniform logits and top_p=1, sampling should explore the vocab."""
    torch.manual_seed(0)
    logits = torch.zeros(1, 8)
    seen = {sample_top_p(logits, temperature=1.0, top_p=1.0)
            for _ in range(200)}
    assert len(seen) >= 4


def test_sample_top_p_bf16_logits_dont_break_softmax():
    """sample_top_p casts to float; bf16 inputs must work."""
    logits = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.bfloat16)
    assert sample_top_p(logits, temperature=0.0, top_p=0.9) == 1


# ------------------------------------------------------ stop tokens --
def _fake_tokenizer(eos_id=2, unk_id=0, known=None):
    """Minimal tokenizer stand-in for stop-token assembly tests."""
    known = known or {}

    def convert(tok):
        if tok in known:
            return known[tok]
        return unk_id

    return types.SimpleNamespace(
        eos_token_id=eos_id, unk_token_id=unk_id,
        convert_tokens_to_ids=convert,
    )


def test_chat_stop_token_ids_includes_eos():
    tok = _fake_tokenizer(eos_id=2)
    assert 2 in chat_stop_token_ids(tok)


def test_chat_stop_token_ids_picks_up_qwen_specials_when_present():
    tok = _fake_tokenizer(
        eos_id=2, unk_id=0,
        known={"<|im_end|>": 151645, "<|endoftext|>": 151643},
    )
    ids = chat_stop_token_ids(tok)
    assert ids == {2, 151645, 151643}


def test_chat_stop_token_ids_ignores_unknown_specials():
    """Tokenizers without Qwen specials map them to UNK -- must skip."""
    tok = _fake_tokenizer(eos_id=2, unk_id=0)
    assert chat_stop_token_ids(tok) == {2}


def test_chat_stop_token_ids_tolerates_no_eos():
    tok = _fake_tokenizer(eos_id=None, unk_id=0,
                          known={"<|im_end|>": 99})
    assert chat_stop_token_ids(tok) == {99}


def test_chat_stop_token_ids_tolerates_convert_raising():
    """If convert_tokens_to_ids raises (some custom tokenizers do for
    unknown strings), that special is just skipped -- function must not
    crash."""

    def convert(_):
        raise RuntimeError("nope")

    tok = types.SimpleNamespace(
        eos_token_id=7, unk_token_id=0, convert_tokens_to_ids=convert,
    )
    assert chat_stop_token_ids(tok) == {7}
