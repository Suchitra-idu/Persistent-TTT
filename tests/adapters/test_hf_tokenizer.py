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


def test_a_max_length_asks_the_tokenizer_to_truncate_natively():
    """Not just the post-hoc `ids[:max_length]` slice — passing HF's own
    `truncation=True` is what stops its over-length warning from firing."""
    spy = _SpyTokenizer()
    tokenizer = HfTokenizer(spy)

    tokenizer.encode_batch(["hello"], max_length=10)

    assert spy.seen_kwargs == {"truncation": True, "max_length": 10}


