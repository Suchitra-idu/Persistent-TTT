"""Tokenizer — text to ids and back, plus the chat template."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def unk_token_id(self) -> int | None: ...

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]: ...

    def encode_batch(
        self, texts: Sequence[str], *, max_length: int | None = None
    ) -> list[list[int]]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...

    def token_id(self, token: str) -> int | None:
        """None for a token this vocabulary does not have, never the unk id —
        `core.sampling_text.stop_token_ids` would otherwise stop on unk."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str: ...
