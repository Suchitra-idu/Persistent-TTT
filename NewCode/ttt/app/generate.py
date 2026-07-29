"""The one token loop. Chat and completions both come through here (D14)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Sequence

from ttt.core import sampling_text
from ttt.ports.generation import Generation

MAX_TOKENS = "max_tokens"
STOP_TOKEN = "stop_token"


@dataclass(frozen=True)
class Completion:
    token_ids: tuple[int, ...]
    stop_reason: str
    stop_token_id: int | None = None


def generate(
    *,
    generation: Generation,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    stop_ids: AbstractSet[int] = frozenset(),
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    generator=None,
) -> Completion:
    """A stop token ends the completion without being emitted.

    `generator` is what makes two arms of a same-seed A/B draw identically, so
    a difference between them attributes to the carry (D14).
    """
    logits = generation.prefill(prompt_ids)
    emitted: list[int] = []

    for _ in range(max_new_tokens):
        token = sampling_text.next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if token in stop_ids:
            return Completion(tuple(emitted), STOP_TOKEN, token)
        emitted.append(token)
        logits = generation.step(token)

    return Completion(tuple(emitted), MAX_TOKENS)
