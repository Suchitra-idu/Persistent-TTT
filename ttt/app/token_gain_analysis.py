"""Where in the target the carry acts: preregistration Addendum A (A1-A3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ttt.app.relatedness_eval import IN_CONTEXT, OTHER_SOURCE, RANDOM, RELATED, RESET, SELF

BINS = ((1, 8), (8, 32), (32, 128), (128, 1024))
ARMS = (SELF, RELATED, RANDOM, OTHER_SOURCE, IN_CONTEXT)
S32_POINT, S32_FLOOR = 0.40, 0.25


@dataclass(frozen=True)
class Interval:
    point: float
    lo95: float
    hi95: float
    lo_adj: float
    hi_adj: float


@dataclass(frozen=True)
class Analysis:
    n_cases: int
    n_clusters: int
    per_token: Mapping[str, tuple[Interval, ...]]
    share: Mapping[str, tuple[float, ...]]
    s32: Mapping[str, Interval]
    novel_advantage: Interval
    shared_advantage: Interval
    novel_fraction: float
    carry_over_context: tuple[float, ...]
    a1: bool
    a2: bool


def gains(rows: Sequence[Mapping], arms: Sequence[str] = ARMS) -> tuple[list[int], dict[str, np.ndarray]]:
    """Per-case, per-position gain over reset; row i of each array is case `order[i]`."""
    by = {(r["arm"], r["doc_idx"]): np.asarray(r["losses"], dtype=np.float64) for r in rows}
    order = sorted({d for a, d in by if a == RESET})
    reset = np.stack([by[(RESET, d)] for d in order])
    present = [a for a in arms if all((a, d) in by for d in order)]
    return order, {a: reset - np.stack([by[(a, d)] for d in order]) for a in present}


def clusters(cases: Sequence[Mapping]) -> list[int]:
    """Connected components of cases sharing a related or a random prime."""
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for i, case in enumerate(cases):
        for level in (RELATED, RANDOM):
            key = (level, case["prime_titles"][level])
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(cases))]


def novel_mask(cases: Sequence[Mapping]) -> np.ndarray:
    """True where a scored target token occurs in neither the related nor the random prime."""
    out = []
    for case in cases:
        seen = set(case["primes"][RELATED]) | set(case["primes"][RANDOM])
        out.append([t not in seen for t in case["target"][1:]])
    return np.asarray(out, dtype=bool)


def _bin_sums(g: np.ndarray) -> np.ndarray:
    return np.stack([g[:, a - 1 : b - 1].sum(axis=1) for a, b in BINS], axis=1)


def _interval(point: float, draws: np.ndarray, alpha_adj: float) -> Interval:
    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
    lo_adj, hi_adj = np.percentile(draws, [100 * alpha_adj / 2, 100 * (1 - alpha_adj / 2)])
    return Interval(float(point), float(lo95), float(hi95), float(lo_adj), float(hi_adj))


def analyse(
    rows: Sequence[Mapping], cases: Sequence[Mapping], *, n_boot: int = 10_000, seed: int = 0,
    n_hypotheses: int = 2,
) -> Analysis:
    """`cases[i]` must be the case with doc_idx i, as saved in the cases file."""
    order, g = gains(rows)
    cases = [cases[d] for d in order]
    cluster = np.asarray(clusters(cases))
    k = int(cluster.max()) + 1
    alpha_adj = 0.05 / n_hypotheses
    widths = np.asarray([b - a for a, b in BINS], dtype=np.float64)
    weights = widths / widths.sum()

    def by_cluster(per_case: np.ndarray) -> np.ndarray:
        out = np.zeros((k, *per_case.shape[1:]))
        np.add.at(out, cluster, per_case)
        return out

    sums = {a: by_cluster(_bin_sums(x)) for a, x in g.items()}
    counts = by_cluster(np.ones(len(cases)))
    novel = novel_mask(cases)
    diff = g[RELATED] - g[RANDOM]
    nov = (by_cluster(_bin_sums(diff * novel)), by_cluster(_bin_sums(novel.astype(float))))
    sha = (by_cluster(_bin_sums(diff * ~novel)), by_cluster(_bin_sums((~novel).astype(float))))

    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)
    ones = np.ones(k)

    def s32(s):
        return lambda c: (c @ s)[..., :2].sum(-1) / (c @ s).sum(-1)

    def stratified(pair):
        s, n = pair
        return lambda c: ((c @ s) / (c @ n)) @ weights

    def both(f):
        return f(ones), f(w)

    per_token_iv = {}
    for a, s in sums.items():
        point, draws = (s.T @ ones) / (counts.sum() * widths), (w @ s) / ((w @ counts)[:, None] * widths)
        per_token_iv[a] = tuple(_interval(point[b], draws[:, b], alpha_adj) for b in range(len(BINS)))
    share = {a: tuple(float(x) for x in (s.sum(0) / s.sum())) for a, s in sums.items()}
    s32_iv = {a: _interval(*both(s32(sums[a])), alpha_adj) for a in (SELF, RELATED)}
    novel_iv = _interval(*both(stratified(nov)), alpha_adj)
    shared_iv = _interval(*both(stratified(sha)), alpha_adj)
    ratio = ()
    if IN_CONTEXT in g:
        ratio = tuple(_ratio(per_token_iv[RELATED][b].point, per_token_iv[IN_CONTEXT][b].point) for b in range(len(BINS)))
    a1 = all(s32_iv[a].point >= S32_POINT and s32_iv[a].lo_adj >= S32_FLOOR for a in (SELF, RELATED))
    return Analysis(
        n_cases=len(cases), n_clusters=k, per_token=per_token_iv, share=share, s32=s32_iv,
        novel_advantage=novel_iv, shared_advantage=shared_iv, novel_fraction=float(novel.mean()),
        carry_over_context=ratio, a1=a1, a2=novel_iv.lo_adj > 0,
    )


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def profile(g: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Mean gain per token within each [edges[i], edges[i+1]) of target positions."""
    return np.asarray([g[:, a - 1 : b - 1].mean() for a, b in zip(edges[:-1], edges[1:])])
