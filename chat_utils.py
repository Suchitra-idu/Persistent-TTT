"""Pure helpers for the interactive chat REPL (CPU unit-testable, no Modal/GPU deps)."""


# Single source of truth for stop-token ids and decoded-string stripping.
CHAT_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def sample_top_p(logits, temperature: float, top_p: float,
                 top_k: int = 0) -> int:
    """Temperature + top_k + nucleus sampling on a [1, V] logits row.

    Greedy when temperature <= 0; top_p=1 disables nucleus; top_k<=0 disables top-k.
    The top-1 token is always kept even at top_p=0. Qwen3 recommends top_k=20.
    """
    import torch

    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scaled = logits.float() / temperature
    if top_k and top_k > 0:
        kth = scaled.topk(top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Always keep the highest-prob token (shift mask one right).
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    filtered = sorted_probs.masked_fill(mask, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(filtered, num_samples=1)
    return int(sorted_idx.gather(-1, pick)[0, 0].item())


def chat_stop_token_ids(tokenizer) -> set:
    """Token ids that should end a chat turn.

    A missing pad_token_id is skipped; do NOT default to id 0 (a real vocab token).
    """
    ids = set()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        ids.add(int(pad_id))
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in CHAT_SPECIAL_TOKENS:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid is not None and tid != unk:
            ids.add(int(tid))
    return ids


def strip_chat_specials(text: str) -> str:
    """Remove ChatML scaffolding markers from a decoded string."""
    for tok in CHAT_SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


def split_thinking(text: str) -> tuple[str, str]:
    """Split an assistant response into (thinking, answer).

    Only splits when BOTH <think> and </think> are present with opener before closer.
    """
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = text.find(open_tag)
    if open_idx == -1:
        return "", text
    close_idx = text.find(close_tag, open_idx + len(open_tag))
    if close_idx == -1:
        return "", text
    thinking = text[open_idx + len(open_tag):close_idx].strip()
    answer = text[close_idx + len(close_tag):]
    return thinking, answer
