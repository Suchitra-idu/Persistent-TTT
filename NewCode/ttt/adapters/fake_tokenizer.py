"""FakeTokenizer — byte-level and deterministic. No download, no vocab file.

Ids are UTF-8 byte values offset past a small special-token table, so encode
and decode round-trip exactly and a test can predict a token count.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
)
BYTE_OFFSET = len(SPECIAL_TOKENS)
VOCAB_SIZE = BYTE_OFFSET + 256

_SPECIAL_RE = re.compile("|".join(re.escape(token) for token in SPECIAL_TOKENS))


class FakeTokenizer:
    def __init__(self, *, pad_token_id: int | None = None) -> None:
        self._ids = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        self._pad_token_id = pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._ids["<|endoftext|>"]

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        return None

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in _SPECIAL_RE.finditer(text):
            ids.extend(_bytes_to_ids(text[position : match.start()]))
            ids.append(self._ids[match.group()])
            position = match.end()
        ids.extend(_bytes_to_ids(text[position:]))
        return ids if max_length is None else ids[:max_length]

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]:
        return [self.encode(text, max_length=max_length) for text in texts]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        out: list[str] = []
        pending: list[int] = []
        for token_id in token_ids:
            if token_id >= BYTE_OFFSET:
                pending.append(token_id - BYTE_OFFSET)
                continue
            out.append(_ids_to_text(pending))
            pending = []
            if not skip_special_tokens:
                out.append(SPECIAL_TOKENS[token_id])
        out.append(_ids_to_text(pending))
        return "".join(out)

    def token_id(self, token: str) -> int | None:
        return self._ids.get(token)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not enable_thinking:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)


def _bytes_to_ids(text: str) -> list[int]:
    return [byte + BYTE_OFFSET for byte in text.encode("utf-8")]


def _ids_to_text(byte_values: Sequence[int]) -> str:
    return bytes(byte_values).decode("utf-8", errors="replace")
