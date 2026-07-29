"""Token sampling and chat-string handling — the pure half of generation.

`nucleus_filter` is RNG-free so the filtering properties are testable without
a draw; only `next_token` touches a generator, and it is injected.
"""

from __future__ import annotations

import torch

CHAT_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def nucleus_filter(
    logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter a [1, V] logits row; returns (probs, indices) sorted descending.

    The cumulative mask is shifted one right so the top-1 token always
    survives; without that a low top_p can filter everything.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0 for sampling, got {temperature}; "
            "greedy decoding is `next_token`'s temperature<=0 path"
        )
    scaled = logits.float() / temperature
    if top_k > 0:
        kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)

    cumulative = sorted_probs.cumsum(dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    filtered = sorted_probs.masked_fill(mask, 0.0)
    return filtered / filtered.sum(dim=-1, keepdim=True), sorted_idx


def next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    generator: torch.Generator | None = None,
) -> int:
    """Greedy when temperature <= 0, else sampled.

    `generator` is what makes the same-prompt A/B (D14) attributable to the
    carry switches rather than to sampling noise.
    """
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    probs, indices = nucleus_filter(
        logits, temperature=temperature, top_p=top_p, top_k=top_k
    )
    pick = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(indices.gather(-1, pick)[0, 0].item())


def stop_token_ids(
    *,
    eos_token_id: int | None,
    pad_token_id: int | None = None,
    special_token_ids: dict[str, int | None] | None = None,
    unk_token_id: int | None = None,
) -> frozenset[int]:
    """Takes ids, not a tokenizer: the tokenizer is a port and this is set math.

    A missing pad id is skipped rather than defaulted to 0 (a real vocabulary
    token), and a special token that maps to unk is skipped for the same reason.
    """
    ids: set[int] = set()
    if pad_token_id is not None:
        ids.add(int(pad_token_id))
    if eos_token_id is not None:
        ids.add(int(eos_token_id))
    for token_id in (special_token_ids or {}).values():
        if token_id is not None and token_id != unk_token_id:
            ids.add(int(token_id))
    return frozenset(ids)


def strip_chat_specials(text: str) -> str:
    for token in CHAT_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def split_thinking(text: str) -> tuple[str, str]:
    """Split into (thinking, answer); a dangling opener leaves it all answer."""
    open_idx = text.find(_THINK_OPEN)
    if open_idx == -1:
        return "", text
    close_idx = text.find(_THINK_CLOSE, open_idx + len(_THINK_OPEN))
    if close_idx == -1:
        return "", text
    thinking = text[open_idx + len(_THINK_OPEN):close_idx].strip()
    return thinking, text[close_idx + len(_THINK_CLOSE):]
