"""Tokenizer — text to ids and back."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, *, max_length: int | None = None) -> list[int]: ...

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...
