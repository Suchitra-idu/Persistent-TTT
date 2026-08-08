"""HfTokenizer's char-cap: a fake underlying tokenizer, no network/model."""

from __future__ import annotations

from ttt.adapters.hf_tokenizer import HfTokenizer


class _SpyTokenizer:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []
        self.seen_kwargs: dict = {}

    def __call__(self, texts, *, add_special_tokens, **kwargs):
        batch = [texts] if isinstance(texts, str) else texts
        self.seen_texts.extend(batch)
        self.seen_kwargs = kwargs
        return _Encoding([[ord(c) for c in text] for text in batch])


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def test_encode_batch_caps_a_pathologically_long_document_before_tokenizing():
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)

    tokenizer.encode_batch(["x" * 1_000_000], max_length=10)

    assert len(spy.seen_texts[0]) == 10 * 20


def test_encode_batch_does_not_touch_a_document_within_the_safety_margin():
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)
    text = "x" * 100

    tokenizer.encode_batch([text], max_length=10)

    assert spy.seen_texts[0] == text


def test_encode_caps_a_pathologically_long_document_too():
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)

    tokenizer.encode("x" * 1_000_000, max_length=10)

    assert len(spy.seen_texts[0]) == 10 * 20


def test_no_max_length_means_no_cap():
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)
    text = "x" * 1_000_000

    tokenizer.encode_batch([text])

    assert spy.seen_texts[0] == text
    assert spy.seen_kwargs == {}


def test_a_max_length_asks_the_tokenizer_to_truncate_natively():
    """Not just the post-hoc `ids[:max_length]` slice — passing HF's own
    `truncation=True` is what stops its over-length warning from firing."""
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)

    tokenizer.encode_batch(["hello"], max_length=10)

    assert spy.seen_kwargs == {"truncation": True, "max_length": 10}


def test_encode_batch_on_an_empty_list_returns_no_rows():
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)

    assert tokenizer.encode_batch([]) == []
