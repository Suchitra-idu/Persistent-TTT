"""How the carry's effect falls with the relatedness of what built it.

Each target is the second block of an article, scored on exactly the same
tokens after the state is reset and after it is primed by an equal-length
block from four sources, ordered by relatedness: the article's own preceding
block, an article that the target article mentions by title, an unrelated
article, and a document from another source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ttt.app import evidence
from ttt.core import metrics
from ttt.core.stats import Estimate
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.rng import Rng

RESET = "reset"
SELF = "self"
RELATED = "related"
RANDOM = "random"
OTHER_SOURCE = "other_source"
LEVELS = (SELF, RELATED, RANDOM, OTHER_SOURCE)
SOURCE = "Wikipedia"

# (measure, reference arm, treated arm); positive means the treated arm is lower.
CONTRASTS = (
    ("self_vs_related", RELATED, SELF),
    ("related_vs_random", RANDOM, RELATED),
    ("random_vs_other_source", OTHER_SOURCE, RANDOM),
)

_WORD = re.compile(r"[\w'’-]+")
MAX_TITLE_WORDS = 4


@dataclass(frozen=True)
class Article:
    title: str
    text: str


@dataclass(frozen=True)
class Pick:
    target: int
    related: int
    unrelated: int
    mentions: int


@dataclass(frozen=True)
class Case:
    index: int
    target_title: str
    target: tuple[int, ...]
    primes: Mapping[str, tuple[int, ...]]
    prime_titles: Mapping[str, str]


@dataclass(frozen=True)
class RelatedRow:
    arm: str
    source: str
    doc_idx: int
    target_title: str
    prime_title: str
    n_tokens: int
    ppl: float


def usable_title(title: str) -> bool:
    """Multi-word, no qualifier: single words collide with ordinary capitalised
    words, and a parenthesised title never appears verbatim in running text."""
    words = title.split()
    return (
        2 <= len(words) <= MAX_TITLE_WORDS
        and "(" not in title
        and not title.startswith("List of")
        and title[0].isupper()
    )


def mentions(text: str, titles: Mapping[str, int]) -> dict[int, int]:
    """Article index -> how often its title occurs in `text` as a whole-word n-gram."""
    words = _WORD.findall(text)
    found: dict[int, int] = {}
    for start in range(len(words)):
        for length in range(2, MAX_TITLE_WORDS + 1):
            index = titles.get(" ".join(words[start : start + length]))
            if index is not None:
                found[index] = found.get(index, 0) + 1
    return found


def pick(
    articles: Sequence[Article],
    *,
    targets: Sequence[int],
    primes: Sequence[int],
    n: int,
    rng: Rng,
    min_mentions: int = 2,
) -> list[Pick]:
    """A related prime is the most-mentioned prime article; the unrelated one
    neither mentions nor is mentioned by the target. Visits targets in a
    seeded order, so the selection is reproducible."""
    titles = {articles[i].title: i for i in primes if usable_title(articles[i].title)}
    order = [targets[i] for i in rng.permutation(len(targets))]
    picks: list[Pick] = []
    for target in order:
        if len(picks) == n:
            break
        found = {i: c for i, c in mentions(articles[target].text, titles).items() if i != target}
        best = max(found.items(), key=lambda item: (item[1], -item[0]), default=None)
        if best is None or best[1] < min_mentions:
            continue
        unrelated = _unrelated(articles, primes, target, best[0], found, rng)
        if unrelated is not None:
            picks.append(Pick(target=target, related=best[0], unrelated=unrelated, mentions=best[1]))
    return picks


def _unrelated(articles, primes, target, related, found, rng, tries: int = 50) -> int | None:
    target_title = articles[target].title
    for _ in range(tries):
        candidate = primes[rng.integers(0, len(primes))]
        if candidate in (target, related) or candidate in found:
            continue
        if target_title in articles[candidate].text:
            continue
        return candidate
    return None


def measure(
    *, cases: Sequence[Case], compute: Compute, fast_weights: FastWeights
) -> tuple[RelatedRow, ...]:
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    rows: list[RelatedRow] = []
    for case in cases:
        fast_weights.reset_carry()
        rows.append(_row(RESET, case, "", compute.eval_loss(case.target)))
        for level in LEVELS:
            fast_weights.reset_carry()
            compute.eval_loss(case.primes[level])
            fast_weights.advance_carry()
            loss = compute.eval_loss(case.target)
            rows.append(_row(level, case, case.prime_titles[level], loss))
    fast_weights.reset_carry()
    return tuple(rows)


def _row(arm: str, case: Case, prime_title: str, loss: float) -> RelatedRow:
    return RelatedRow(
        arm=arm, source=SOURCE, doc_idx=case.index, target_title=case.target_title,
        prime_title=prime_title, n_tokens=len(case.target), ppl=metrics.perplexity(loss),
    )


def estimates(rows: Sequence[RelatedRow], *, rng: Rng, n_boot: int) -> tuple[Estimate, ...]:
    def arm(name: str):
        return evidence.per_doc(rows, where=lambda r: r.arm == name)

    kw = dict(rng=rng, n_boot=n_boot)
    out: tuple[Estimate, ...] = ()
    for level in LEVELS:
        out += evidence.paired_estimates(f"gain_{level}", arm(RESET), arm(level), **kw)
    for measure_name, reference, treated in CONTRASTS:
        out += evidence.paired_estimates(measure_name, arm(reference), arm(treated), **kw)
    return out
