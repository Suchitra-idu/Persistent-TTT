"""HfTokenizer — the Tokenizer port over `transformers.AutoTokenizer`."""

from __future__ import annotations

from typing import Mapping, Sequence


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
        ids = self._tokenizer(text, add_special_tokens=False).input_ids
        return ids if max_length is None else ids[:max_length]

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]:
        batch = self._tokenizer(list(texts), add_special_tokens=False).input_ids
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
