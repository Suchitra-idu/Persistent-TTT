"""HfTokenizer — the Tokenizer port over `transformers.AutoTokenizer`."""

from __future__ import annotations

from typing import Mapping, Sequence

# A generous upper bound on chars/token (measured tokenizer efficiency in
# this project tops out around 3.4 for English) — text past this many chars
# for a given max_length would be truncated away anyway, so capping it before
# tokenizing changes nothing for any real document, only for a pathological
# one. Without this, a 200k+-token raw document (a real SlimPajama book) gets
# fully tokenized before truncation ever applies, which segfaulted the fast
# tokenizer during training rather than raising.
_SAFETY_CHARS_PER_TOKEN = 20


def _capped(text: str, max_length: int | None) -> str:
    if max_length is None:
        return text
    limit = max_length * _SAFETY_CHARS_PER_TOKEN
    return text[:limit] if len(text) > limit else text


def _truncation(max_length: int | None) -> dict:
    """Native HF truncation, not just a post-hoc `ids[:max_length]` slice —
    the same slice still runs after as a cheap belt-and-braces check, but
    this is what stops the tokenizer's own over-length warning from firing
    on every long document."""
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

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        return self._tokenizer.unk_token_id

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        ids = self._tokenizer(
            _capped(text, max_length), add_special_tokens=False, **_truncation(max_length)
        ).input_ids
        return ids if max_length is None else ids[:max_length]

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]:
        # The fast tokenizer indexes into its own output unconditionally and
        # raises IndexError on an empty batch instead of returning [].
        if not texts:
            return []
        capped = [_capped(text, max_length) for text in texts]
        batch = self._tokenizer(
            capped, add_special_tokens=False, **_truncation(max_length)
        ).input_ids
        if max_length is None:
            return [list(ids) for ids in batch]
        return [list(ids[:max_length]) for ids in batch]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def token_id(self, token: str) -> int | None:
        converted = self._tokenizer.convert_tokens_to_ids(token)
        if converted is None or converted == self._tokenizer.unk_token_id:
            return None
        return int(converted)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        return self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
