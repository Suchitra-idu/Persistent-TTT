"""HfTokenizer — the Tokenizer port over `transformers.AutoTokenizer`."""

from __future__ import annotations

from typing import Sequence

# Cap characters before tokenizing: a 200k-token book segfaulted the fast
# tokenizer. 20 chars/token is far above any real text, so nothing kept is cut.
_SAFETY_CHARS_PER_TOKEN = 20


def _capped(text: str, max_length: int | None) -> str:
    if max_length is None:
        return text
    limit = max_length * _SAFETY_CHARS_PER_TOKEN
    return text[:limit] if len(text) > limit else text


def _truncation(max_length: int | None) -> dict:
    """Native truncation, which also silences the tokenizer's over-length warning."""
    if max_length is None:
        return {}
    return {"truncation": True, "max_length": max_length}


class HfTokenizer:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str) -> "HfTokenizer":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_id))

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        ids = self._tokenizer(
            _capped(text, max_length), add_special_tokens=False, **_truncation(max_length)
        ).input_ids
        return ids

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]:
        # The fast tokenizer raises IndexError on an empty batch.
        if not texts:
            return []
        capped = [_capped(text, max_length) for text in texts]
        batch = self._tokenizer(
            capped, add_special_tokens=False, **_truncation(max_length)
        ).input_ids
        return [list(ids) for ids in batch]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids))
