"""FakeTokenizer — byte-level and deterministic: one id per UTF-8 byte, so a test can predict a token count."""

from __future__ import annotations

from typing import Sequence


class FakeTokenizer:
    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        ids = list(text.encode("utf-8"))
        return ids if max_length is None else ids[:max_length]

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]:
        return [self.encode(text, max_length=max_length) for text in texts]

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")
