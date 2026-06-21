"""
Pure training utilities, extracted from train_modal.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionItem:
    """One unit of work in a session: a contiguous token range of a doc.

    The training loop processes a SessionItem exactly as it used to
    process a whole paper -- one forward/backward, fast weights carry
    forward via advance_session_state. When start=0 and end=len(doc) the
    item is the whole paper (the legacy notion). When build_session_items
    slices a paper into multiple consecutive items inside the same
    session, fast weight carry now spans BOTH intra-paper slice
    boundaries and inter-paper boundaries, which fixes overfitting to
    same-paper-only carry patterns."""

    doc_idx: int
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


def make_session_schedule(num_docs: int, lo: int, hi: int, rng) -> list:
    """Shuffle all docs, then partition the order into sessions of
    random size n ~ Uniform[lo, hi]. Every doc appears exactly once per
    epoch; only the grouping (and therefore the fast weight carry
    structure) is random. The final session may be shorter than lo."""
    order = rng.permutation(num_docs)
    sessions, i = [], 0
    while i < num_docs:
        n = int(rng.integers(lo, hi + 1))
        sessions.append(order[i:i + n].tolist())
        i += n
    return sessions


def slice_doc(doc_length: int, k: int, min_slice_tokens: int, rng) -> list:
    """Partition [0, doc_length] into k contiguous slices each of size
    >= min_slice_tokens at random boundaries. Returns [(start, end), ...]
    covering the full range with no gaps or overlap. Falls back to a
    single whole-doc slice when k <= 1 or the doc is too short to admit
    k slices at the required minimum size."""
    if k <= 1 or k * min_slice_tokens > doc_length:
        return [(0, doc_length)]
    # Reserve min_slice_tokens per segment, randomize where the leftover
    # tokens land. Sampling k-1 sorted cuts in [0, free] then adding the
    # cumulative reserve makes every segment >= min_slice_tokens by
    # construction.
    free = doc_length - k * min_slice_tokens
    cuts = sorted(int(rng.integers(0, free + 1)) for _ in range(k - 1))
    boundaries = (
        [0]
        + [c + (i + 1) * min_slice_tokens for i, c in enumerate(cuts)]
        + [doc_length]
    )
    return [(boundaries[i], boundaries[i + 1]) for i in range(k)]


def equal_token_slices(doc_length: int, n_slices: int) -> list:
    """Partition [0, doc_length] into n_slices consecutive token ranges
    of roughly equal size. Boundary positions are nearest-integer
    rounded, the last boundary is pinned at doc_length, and empty
    (zero-token) ranges are dropped -- so passing n_slices > doc_length
    returns fewer than n_slices items instead of crashing.

    Used by single-paper eval: same paper at every slice = same content
    distribution, so the per-position carry signal is isolated from
    paper-to-paper variation."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    boundaries = [round(i * doc_length / n_slices) for i in range(n_slices + 1)]
    boundaries[-1] = doc_length
    return [(boundaries[i], boundaries[i + 1])
            for i in range(n_slices)
            if boundaries[i + 1] > boundaries[i]]


def build_session_items(
    sessions: list,
    doc_lengths,
    slice_prob: float,
    slice_min: int,
    slice_max: int,
    min_slice_tokens: int,
    rng,
) -> list:
    """Expand a list of (doc-index) sessions into a list of SessionItem
    sessions, with random per-paper slicing inside each session.

    For each paper, with probability slice_prob, split into
    k ~ Uniform[slice_min, slice_max] consecutive token slices via
    slice_doc; otherwise emit one whole-paper item. Slices of the same
    paper stay grouped together and ordered, and the order of papers
    inside a session is preserved. With slice_prob=0 the output is
    structurally identical to the legacy schedule (one item per paper,
    full range), so disabling the feature is a zero-cost no-op."""
    out = []
    for session in sessions:
        items = []
        for doc_idx in session:
            L = int(doc_lengths[doc_idx])
            if slice_max > 1 and rng.random() < slice_prob:
                k = int(rng.integers(slice_min, slice_max + 1))
            else:
                k = 1
            for s, e in slice_doc(L, k, min_slice_tokens, rng):
                items.append(SessionItem(int(doc_idx), s, e))
        out.append(items)
    return out


def make_single_paper_sessions(num_docs: int, doc_lengths, lo: int, hi: int,
                                min_slice_tokens: int, rng) -> list:
    """Build sessions where each session = ONE paper randomly sliced
    into k ~ Uniform[lo, hi] consecutive pieces. Every paper appears
    exactly once per epoch in shuffled order. Falls back to fewer
    slices (down to a single whole-paper item) when the paper is too
    short to admit k slices of min_slice_tokens each.

    Use when you want EVERY item in a session to share content with
    the rest: the carry has guaranteed signal to learn from, no risk
    of an unrelated paper's content being silently "carried" between
    items as noise. Trades the cross-paper memory training signal for
    a cleaner intra-paper one -- pick the mode that matches the
    research question."""
    order = rng.permutation(num_docs)
    sessions = []
    for doc_idx in order.tolist():
        L = int(doc_lengths[doc_idx])
        k = int(rng.integers(lo, hi + 1))
        # Cap k by feasibility so slice_doc never silently collapses
        # to k=1 just because we asked for more than the paper allows.
        while k > 1 and k * min_slice_tokens > L:
            k -= 1
        ranges = slice_doc(L, k, min_slice_tokens, rng)
        sessions.append(
            [SessionItem(int(doc_idx), s, e) for s, e in ranges]
        )
    return sessions


def expected_items_per_doc(slice_prob: float, slice_min: int,
                           slice_max: int) -> float:
    """Coarse expectation of the number of SessionItems a paper produces
    under build_session_items, used to size the LR schedule. Assumes
    most papers admit the sampled k; short papers occasionally fall back
    to k=1, so the real count is slightly lower (the cosine schedule
    then doesn't fully anneal -- benign)."""
    if slice_max <= 1 or slice_prob <= 0:
        return 1.0
    return (1.0 - slice_prob) + slice_prob * 0.5 * (slice_min + slice_max)


def grad_norms(named_groups: dict) -> dict:
    """L2 norm of accumulated gradients per parameter group. The
    grad/new signal is the wiring-broken detector; see observability."""
    import torch

    out = {}
    for name, params in named_groups.items():
        sq = sum(
            p.grad.detach().float().pow(2).sum()
            for p in params if p.grad is not None
        )
        out[name] = float(torch.as_tensor(sq).sqrt())
    return out


# ---------------------------------------------------------------------------
# Content-token loss masking.
#
# Standard next-token CE on a long document is dominated by predictable
# function words, punctuation, and corpus-style tokens. The handful of
# tokens that actually carry document-specific information (entities,
# numbers, technical terms) contribute almost nothing to the gradient.
# This pair of helpers lets the training loop concentrate the loss on
# rare/content-bearing tokens by setting common-token positions to -100,
# which CrossEntropyLoss skips by default.
# ---------------------------------------------------------------------------
def build_common_token_mask(input_ids_iter, vocab_size: int,
                            keep_fraction: float):
    """Returns a [vocab_size] BoolTensor where True marks 'common' tokens
    whose loss should be ignored.

    Algorithm: count unigram occurrences across the supplied corpus, sort
    token ids by frequency descending, then mark the smallest prefix of
    token ids whose cumulative count covers (1 - keep_fraction) of total
    occurrences. The unmarked tail therefore carries approximately
    keep_fraction of corpus token positions -- the rare-token signal.

    input_ids_iter -- any iterable yielding 1D int sequences (e.g. a
        list of lists, or `(ex["input_ids"] for ex in ds)`).
    vocab_size -- width of the returned mask. Token ids observed must
        all be in [0, vocab_size).
    keep_fraction -- must be in (0, 1]. 1.0 disables masking (returns
        all-False); smaller values mask more aggressively. Validated at
        the boundary; behavior at exactly the limits is documented in
        tests.

    Determinism: with the same input_ids_iter, vocab_size, and
    keep_fraction the returned mask is bit-exact, including the
    tie-breaking order from `torch.sort(descending=True)` (stable sort
    on counts; ties broken by original token id order).

    Thin wrapper around the count + threshold pair so callers that
    want both passes in one shot stay clean. Argument validation
    lives inside the two helpers, which run independent of each
    other (the diagnostic counts once and thresholds multiple times)."""
    counts = count_unigrams(input_ids_iter, vocab_size)
    return common_mask_from_counts(counts, keep_fraction)


def count_unigrams(input_ids_iter, vocab_size: int):
    """Single-pass unigram tally over an input-ids iterable. Lifted out
    of build_common_token_mask so diagnostics can reuse counts across
    multiple thresholds or cumulative prefixes (one pass, not N).

    Returns a [vocab_size] int64 CPU tensor. Out-of-range token ids
    are rejected loudly at the call site that caused them, rather than
    being silently absorbed into a wider bincount vector that would
    later shape-mismatch."""
    import torch

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for ids in input_ids_iter:
        t = torch.as_tensor(ids, dtype=torch.int64)
        if not t.numel():
            continue
        # Range-check BEFORE bincount: bincount silently returns a
        # vector sized max(minlength, max(t)+1), which would then shape-
        # mismatch the running counts and crash with a confusing error.
        if int(t.max()) >= vocab_size or int(t.min()) < 0:
            raise ValueError(
                f"token id out of range [0, {vocab_size}): "
                f"saw min={int(t.min())}, max={int(t.max())}"
            )
        counts += torch.bincount(t, minlength=vocab_size)
    return counts


def common_mask_from_counts(counts, keep_fraction: float):
    """The threshold half of the mask pipeline: given a [vocab_size]
    count tensor, return a [vocab_size] BoolTensor marking the smallest
    prefix of frequency-sorted token ids whose cumulative occurrences
    cover (1 - keep_fraction) of the total. Separated from counting so
    diagnostics can sweep keep_fraction without re-tallying."""
    import torch

    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(
            f"keep_fraction must be in (0, 1], got {keep_fraction}"
        )
    vocab_size = int(counts.numel())
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    total = int(counts.sum().item())
    drop_budget = int(round((1.0 - keep_fraction) * total))
    # keep_fraction=1.0, empty corpus, or rounding to zero => mask nothing.
    if drop_budget <= 0:
        return mask

    sorted_counts, sorted_ids = torch.sort(counts, descending=True, stable=True)
    cum = sorted_counts.cumsum(0)
    budget_t = torch.tensor(drop_budget, dtype=cum.dtype)
    # First index where cum >= drop_budget; we drop ids 0..idx inclusive.
    idx = int(torch.searchsorted(cum, budget_t, right=False).item())
    n_dropped = min(idx + 1, vocab_size)
    mask[sorted_ids[:n_dropped]] = True
    return mask


def apply_loss_mask(ids, common_mask, first_tokens: int = 0):
    """Return a labels tensor with positions ignored by the CE loss
    where appropriate. Two independent masking sources, each applied
    only if active:

      1. common_mask -- token-id based. Catches the most-frequent token
         ids (function words, punctuation, generic ML-glue) so the
         gradient concentrates on content-bearing tokens. Caller passes
         None to disable.
      2. first_tokens -- position based. Sets the leading N positions
         of every sequence to ignore_index. Use when the corpus shares
         a near-identical opening that the model would otherwise burn
         capacity learning (e.g. arxiv papers all starting with
         "1. Introduction"). Caller passes 0 to disable, OR passes 0
         specifically for mid-paper slices where the position-based
         skip would discard real content.

    When BOTH disable signals are set (common_mask=None and
    first_tokens<=0), ids is returned identically (no clone, zero
    overhead), so the helper can be wired in unconditionally and
    toggled via config.

    Inputs are NOT mutated. common_mask must live on the same device
    as ids when supplied; first_tokens is a CPU int. Shape and dtype
    of the return value match ids."""
    if common_mask is None and first_tokens <= 0:
        return ids
    labels = ids.clone()
    if common_mask is not None:
        labels[common_mask[ids]] = -100
    if first_tokens > 0:
        # Mask the leading N positions of each sequence. Last axis is
        # the token dimension; arbitrary leading batch dims are fine.
        k = min(first_tokens, labels.shape[-1])
        labels[..., :k] = -100
    return labels


def apply_protect_passes(mask, tokenizer, protect_terms,
                         protect_numeric: bool, protect_symbols):
    """Apply all three protect overlays on top of a frequency-built
    mask. The two callers (train() and diagnose_loss_mask()) ran the
    same dance before this extraction; keeping it in one place makes
    the "what's safe to never mask" policy a single source of truth.

    Mutates `mask` in place. Returns (n_freed_terms, n_freed_numeric,
    n_freed_symbols) so the caller can log the deltas. Any input may
    be empty/None/False to disable that path -- each disabled overlay
    is a zero-cost no-op."""
    n_freed_terms = 0
    if protect_terms:
        touched = protect_token_ids(mask, tokenizer, protect_terms)
        n_freed_terms = sum(len(v) for v in touched.values())
    n_freed_numeric = 0
    if protect_numeric:
        n_freed_numeric = len(protect_numeric_tokens(mask, tokenizer))
    n_freed_symbols = 0
    if protect_symbols:
        n_freed_symbols = len(
            protect_symbol_tokens(mask, tokenizer, protect_symbols)
        )
    return n_freed_terms, n_freed_numeric, n_freed_symbols


def load_reference_counts(path: str, expected_vocab_size: int):
    """Load a precomputed reference unigram count tensor from disk
    (built by train_modal.py::build_reference_counts). Returns
    (counts, metadata_dict) or (None, None) if `path` is empty / the
    file is missing -- the caller can then fall back to in-corpus
    counting without an explicit feature flag.

    Raises RuntimeError on a vocab_size mismatch so a tokenizer
    change is caught at training start, not silently at the first
    bincount.

    Importing torch lazily (like the other train_utils helpers) keeps
    the module light on the path where the feature is unused."""
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


def protect_by_predicate(mask, tokenizer, predicate):
    """Force-unmask token ids whose decoded form (stripped) satisfies
    a user-supplied predicate. The base building block for any rule
    that needs to walk the masked set and selectively flip ids based
    on what the token decodes to.

    Iterates only over currently-masked indices, so the cost is
    O(|mask.sum()|) tokenizer.decode calls (~200 at kf=0.5) rather
    than O(vocab_size) (~150k). Mutates `mask` in place. Returns the
    sorted list of token ids flipped from masked to unmasked.

    predicate(decoded_stripped) -> bool. Receives the decoded token
    text with leading/trailing whitespace stripped; should return
    True to unmask. Pure whitespace tokens decode to '' (falsy after
    strip) -- if the predicate accepts '', that's the predicate's
    explicit choice."""
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    flipped = []
    for tid in indices:
        decoded = tokenizer.decode([int(tid)]).strip()
        if predicate(decoded):
            flipped.append(int(tid))
            mask[tid] = False
    return sorted(flipped)


def protect_numeric_tokens(mask, tokenizer):
    """Force-unmask token ids whose decoded form is a pure-digit
    string. Numbers in scientific corpora carry content --
    hyperparameter values, benchmark scores, model sizes -- so the
    loss should always see them. The term-based protect-list can't
    handle digits cleanly (single-character pieces fail its length
    gate), hence the predicate path.

    Catches '0'..'9' and multi-digit pieces ('100', '1024', etc.) in
    whatever form BPE chose to emit them, including the leading-space
    form (' 1', ' 100'). Tokens with any non-digit character (' 1.',
    '0)', 'e-4', '1st') are left untouched; treat those as
    number-adjacent rather than pure numerical content. str.isdigit()
    is True on non-empty all-digit strings AND on unicode
    superscript/subscript digits ('²', '₁'), which is the looser
    semantics we want."""
    return protect_by_predicate(
        mask, tokenizer, lambda s: bool(s) and s.isdigit(),
    )


def protect_symbol_tokens(mask, tokenizer, symbols):
    """Force-unmask token ids whose decoded form is exactly one of
    `symbols`. Math operators ('=', '@', '^', '_', '+', '-', '*',
    '/', '\\', '|') are content in scientific corpora -- equations,
    tensor ops, subscripts, superscripts, LaTeX -- and the loss
    should always see them. Like digits, they're single-character
    tokens that the term-based protect can't catch.

    `symbols` is any container supporting `in`; pass a set/tuple of
    decoded strings. Stripped match: ' =' and '=' both protect the
    same id if either is requested."""
    symbol_set = frozenset(symbols)
    return protect_by_predicate(
        mask, tokenizer, lambda s: s in symbol_set,
    )


def protect_token_ids(mask, tokenizer, terms, min_piece_chars: int = 3):
    """Force-unmask token ids reached by tokenizing any of `terms`, so
    the frequency mask cannot eat domain-critical vocabulary.

    Variant generation. For each term we try bare / leading-space /
    capitalized / all-caps, with and without leading space (8 variants
    after dedup).

    Per-variant gating. Each variant tokenizes to one or more BPE
    pieces. We unmask the variant's pieces only when EVERY piece has
    at least `min_piece_chars` non-whitespace characters when decoded.
    This admits multi-piece content terms (' diff' + 'usion',
    ' trans' + 'former') while rejecting acronym splits like
    ' VAE' -> [' V', 'AE']. Unmasking ' V' alone would leak to every
    mid-sentence capital-V word; better to leave the acronym
    unprotected (the second piece 'AE' is rare enough that the
    frequency mask spares it anyway, so the model still gets gradient
    on most of the acronym).

    Mutates `mask` in place. Returns a dict { term: [token_id, ...] }
    listing the token ids each term genuinely flipped from masked to
    unmasked. Empty list distinguishes 'no variant passed the gate'
    from 'every passing variant was already unmasked'; both cases are
    useful in the startup log.

    'add_special_tokens=False' is critical -- with it, BOS/EOS are not
    glued onto the variant and the encoded length reflects only the
    term itself."""
    touched = {}
    for term in terms:
        if not term:
            continue
        variants = set()
        for base in (term, term.lower(), term.upper(), term.capitalize()):
            variants.add(base)
            variants.add(" " + base)
        flipped = []
        # sorted() so the reported list is deterministic; the actual
        # mask flip is order-independent.
        for v in sorted(variants):
            ids = tokenizer.encode(v, add_special_tokens=False)
            if not ids:
                continue
            # Length gate: decode each piece, strip whitespace (the
            # leading-space convention shouldn't inflate length),
            # require every piece to clear the floor.
            pieces = [tokenizer.decode([int(tid)]) for tid in ids]
            if any(len(p.strip()) < min_piece_chars for p in pieces):
                continue
            for tid in ids:
                tid = int(tid)
                if 0 <= tid < mask.numel() and bool(mask[tid]):
                    flipped.append(tid)
                    mask[tid] = False
        # De-dup: the same token id can show up via several variants
        # (e.g. ' transformer' and ' Transformer' may map identically
        # after BPE normalization), but only the first flip is real.
        touched[term] = sorted(set(flipped))
    return touched
