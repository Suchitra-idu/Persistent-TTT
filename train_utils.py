"""Pure training utilities extracted from train_modal.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionItem:
    """One unit of work in a session: a contiguous token range of a doc."""

    doc_idx: int
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


def make_session_schedule(num_docs: int, lo: int, hi: int, rng) -> list:
    """Shuffle all docs, partition the order into sessions of size n ~ Uniform[lo, hi]."""
    order = rng.permutation(num_docs)
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(lo, hi + 1))
        sessions.append(order[i:i + n].tolist())
        i += n
    return sessions


def slice_doc(doc_length: int, k: int, min_slice_tokens: int, rng) -> list:
    """Partition [0, doc_length] into k contiguous slices each >= min_slice_tokens."""
    if k <= 1 or k * min_slice_tokens > doc_length:
        return [(0, doc_length)]
    free = doc_length - k * min_slice_tokens
    cuts = sorted(int(rng.integers(0, free + 1)) for _ in range(k - 1))
    boundaries = (
        [0]
        + [c + (i + 1) * min_slice_tokens for i, c in enumerate(cuts)]
        + [doc_length]
    )
    return [(boundaries[i], boundaries[i + 1]) for i in range(k)]


def equal_token_slices(doc_length: int, n_slices: int) -> list:
    """Partition [0, doc_length] into n_slices consecutive ranges of roughly equal size."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    boundaries = [round(i * doc_length / n_slices) for i in range(n_slices + 1)]
    boundaries[-1] = doc_length
    return [(boundaries[i], boundaries[i + 1])
            for i in range(n_slices)
            if boundaries[i + 1] > boundaries[i]]


def make_slice_sessions(
    num_docs: int,
    doc_lengths,
    rng,
    *,
    session_papers: tuple,
    slice_prob: float,
    slice_range: tuple,
    min_slice_tokens: int,
    shuffle: bool = True,
) -> list:
    """Build sessions of SessionItems.

    Session size drawn Uniform(session_papers[0], session_papers[1]).
    Each paper is sliced into k ~ Uniform(slice_range) with probability slice_prob.
    k decrements toward feasibility so a doc that could take k=3 slices but not k=4
    yields 3 items, not silently 1.

    shuffle=False keeps doc order intact (used by inference where the caller
    controls order); training shuffles between epochs to decorrelate.

    Single-paper-per-session mode: session_papers=(1, 1), slice_prob=1.0,
    slice_range=(min, max).
    Multi-paper mode: session_papers=(lo, hi), slice_prob=p, slice_range=(min, max).
    """
    order = (rng.permutation(num_docs).tolist() if shuffle
             else list(range(num_docs)))
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(session_papers[0], session_papers[1] + 1))
        session = []
        for doc_idx in order[i:i + n]:
            L = int(doc_lengths[doc_idx])
            if slice_range[1] > 1 and rng.random() < slice_prob:
                k = int(rng.integers(slice_range[0], slice_range[1] + 1))
                while k > 1 and k * min_slice_tokens > L:
                    k -= 1
            else:
                k = 1
            for s, e in slice_doc(L, k, min_slice_tokens, rng):
                session.append(SessionItem(int(doc_idx), s, e))
        sessions.append(session)
        i += n
    return sessions


def expected_items_per_doc(slice_prob: float, slice_min: int,
                           slice_max: int) -> float:
    """Coarse expectation of SessionItems per paper, used to size the LR schedule."""
    if slice_max <= 1 or slice_prob <= 0:
        return 1.0
    return (1.0 - slice_prob) + slice_prob * 0.5 * (slice_min + slice_max)


def count_unigrams(input_ids_iter, vocab_size: int):
    """Single-pass unigram tally over an input-ids iterable."""
    import torch

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for ids in input_ids_iter:
        t = torch.as_tensor(ids, dtype=torch.int64)
        if not t.numel():
            continue
        # Range-check BEFORE bincount: bincount would silently grow the vector otherwise.
        if int(t.max()) >= vocab_size or int(t.min()) < 0:
            raise ValueError(
                f"token id out of range [0, {vocab_size}): "
                f"saw min={int(t.min())}, max={int(t.max())}"
            )
        counts += torch.bincount(t, minlength=vocab_size)
    return counts


def common_mask_from_counts(counts, keep_fraction: float):
    """Return a BoolTensor marking the smallest freq-sorted prefix covering (1 - keep_fraction)."""
    import torch

    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(
            f"keep_fraction must be in (0, 1], got {keep_fraction}"
        )
    vocab_size = int(counts.numel())
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    total = int(counts.sum().item())
    drop_budget = int(round((1.0 - keep_fraction) * total))
    if drop_budget <= 0:
        return mask

    sorted_counts, sorted_ids = torch.sort(counts, descending=True, stable=True)
    cum = sorted_counts.cumsum(0)
    budget_t = torch.tensor(drop_budget, dtype=cum.dtype)
    idx = int(torch.searchsorted(cum, budget_t, right=False).item())
    n_dropped = min(idx + 1, vocab_size)
    mask[sorted_ids[:n_dropped]] = True
    return mask


def apply_loss_mask(ids, common_mask, first_tokens: int = 0):
    """Return a labels tensor with CE-ignored positions per common_mask and first_tokens.

    common_mask must live on the same device as ids when supplied.
    Inputs are NOT mutated.
    """
    if common_mask is None and first_tokens <= 0:
        return ids
    labels = ids.clone()
    if common_mask is not None:
        labels[common_mask[ids]] = -100
    if first_tokens > 0:
        k = min(first_tokens, labels.shape[-1])
        labels[..., :k] = -100
    return labels


def protect_by_predicate(mask, tokenizer, predicate):
    """Force-unmask token ids whose decoded (stripped) form satisfies predicate (mutates mask)."""
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    flipped = []
    for tid in indices:
        decoded = tokenizer.decode([int(tid)]).strip()
        if predicate(decoded):
            flipped.append(int(tid))
            mask[tid] = False
    return sorted(flipped)


def protect_numeric_tokens(mask, tokenizer):
    """Force-unmask token ids whose decoded form is a pure-digit string."""
    return protect_by_predicate(
        mask, tokenizer, lambda s: bool(s) and s.isdigit(),
    )


def protect_symbol_tokens(mask, tokenizer, symbols):
    """Force-unmask token ids whose decoded form is exactly one of `symbols`."""
    symbol_set = frozenset(symbols)
    return protect_by_predicate(
        mask, tokenizer, lambda s: s in symbol_set,
    )


def protect_token_ids(mask, tokenizer, terms, min_piece_chars: int = 3):
    """Force-unmask token ids reached by tokenizing any of `terms` (mutates mask).

    Unmasks a variant's pieces only when every piece has >= min_piece_chars
    non-whitespace chars when decoded (rejects acronym splits like ' V' + 'AE').
    add_special_tokens=False is critical: BOS/EOS must not glue onto the variant.
    """
    touched = {}
    for term in terms:
        if not term:
            continue
        variants = set()
        for base in (term, term.lower(), term.upper(), term.capitalize()):
            variants.add(base)
            variants.add(" " + base)
        flipped = []
        for v in sorted(variants):
            ids = tokenizer.encode(v, add_special_tokens=False)
            if not ids:
                continue
            pieces = [tokenizer.decode([int(tid)]) for tid in ids]
            if any(len(p.strip()) < min_piece_chars for p in pieces):
                continue
            for tid in ids:
                tid = int(tid)
                if 0 <= tid < mask.numel() and bool(mask[tid]):
                    flipped.append(tid)
                    mask[tid] = False
        touched[term] = sorted(set(flipped))
    return touched


def apply_protect_passes(mask, tokenizer, protect_terms,
                         protect_numeric: bool, protect_symbols):
    """Apply all three protect overlays on a frequency-built mask (mutates in place).
    Returns (n_freed_terms, n_freed_numeric, n_freed_symbols)."""
    n_freed_terms = 0
    if protect_terms:
        touched = protect_token_ids(mask, tokenizer, protect_terms)
        n_freed_terms = sum(len(v) for v in touched.values())
    n_freed_numeric = (len(protect_numeric_tokens(mask, tokenizer))
                       if protect_numeric else 0)
    n_freed_symbols = (len(protect_symbol_tokens(mask, tokenizer, protect_symbols))
                       if protect_symbols else 0)
    return n_freed_terms, n_freed_numeric, n_freed_symbols


def load_reference_counts(path: str, expected_vocab_size: int):
    """Load a precomputed reference unigram count tensor; returns (counts, meta) or (None, None)."""
    import os
    import torch

    if not path or not os.path.exists(path):
        return None, None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    counts = blob["counts"]
    if counts.numel() != expected_vocab_size:
        raise RuntimeError(
            f"reference counts vocab_size {counts.numel()} != "
            f"expected {expected_vocab_size}; rebuild the reference "
            f"after a tokenizer change via build_reference_counts"
        )
    return counts, blob
