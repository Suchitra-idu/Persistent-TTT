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
