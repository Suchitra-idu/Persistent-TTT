"""
Pure helpers for the interactive chat REPL.

Lives at the project root (no Modal / GPU / model dependency) so the
sampling and prompt-formatting logic is unit-testable on CPU. The
TTTInference class imports from here; the harder-to-test glue (KV
cache, fast-weight bookkeeping) stays in infer_modal.py.
"""


def sample_top_p(logits, temperature: float, top_p: float) -> int:
    """Temperature + nucleus sampling on a [1, V] logits row. Returns
    the sampled token id. Greedy when temperature <= 0; top_p=1 disables
    the nucleus filter. The top-1 token is always kept even at top_p=0,
    so the function never returns a degenerate token."""
    import torch

    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Drop the suffix whose cumulative prob exceeds top_p, but always
    # keep the highest-prob token (shift the mask one position right).
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    filtered = sorted_probs.masked_fill(mask, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(filtered, num_samples=1)
    return int(sorted_idx.gather(-1, pick)[0, 0].item())


def format_user_turn(user_message: str, system_prompt: str = "") -> str:
    """Plain "[<system>\\n\\n]User: ... \\nAssistant: " turn text. The
    chat loop feeds ONLY this string to the model each turn -- the
    static system prompt is allowed (it is an instruction, not
    conversation history) but PRIOR USER/ASSISTANT TURNS are not.
    Cross-turn conversation memory in this research setup MUST be
    carried by the persistent TTT fast weights; re-supplying earlier
    turns as in-context tokens would defeat the test of the mechanism."""
    if system_prompt.strip():
        return f"{system_prompt.rstrip()}\n\nUser: {user_message}\nAssistant: "
    return f"User: {user_message}\nAssistant: "


def chat_stop_token_ids(tokenizer) -> set:
    """Token ids that should end a chat turn. Always eos_token_id; adds
    Qwen-style <|im_end|> / <|endoftext|> when those tokens exist in the
    vocab so generations halt at a natural boundary even on a base
    (non-instruct) model."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid is not None and tid != unk:
            ids.add(int(tid))
    return ids
