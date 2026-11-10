"""How the carry's effect falls with the relatedness of what built it.

Each target is the second block of an article, scored on exactly the same
tokens after the state is reset and after it is primed by an equal-length
block from four sources, ordered by relatedness: the article's own preceding
block, an article that the target article mentions by title, an unrelated
article, and a document from another source.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ttt.app import evidence
from ttt.core import metrics
from ttt.core.stats import Estimate
from ttt.core.types import Carry
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.rng import Rng

RESET = "reset"
SELF = "self"
RELATED = "related"
RANDOM = "random"
OTHER_SOURCE = "other_source"
LEVELS = (SELF, RELATED, RANDOM, OTHER_SOURCE)
GENERIC = "generic"
IN_CONTEXT = "in_context"
CHAIN = "chain_"
SOURCE = "Wikipedia"
ADJUSTED_LEVEL = 1 - 0.05 / 4

# (measure, reference arm, treated arm); positive means the treated arm is lower.
CONTRASTS = (
    ("self_vs_related", RELATED, SELF),
    ("related_vs_random", RANDOM, RELATED),
    ("random_vs_other_source", OTHER_SOURCE, RANDOM),
)

# Preregistered headline contrasts (plan/PREREGISTRATION.md §6), each also at ADJUSTED_LEVEL.
HEADLINE = (
    ("self_vs_generic", GENERIC, SELF),
    ("related_vs_generic", GENERIC, RELATED),
    ("related_vs_random", RANDOM, RELATED),
    ("carry_vs_in_context", IN_CONTEXT, RELATED),
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
    chain: tuple[tuple[int, ...], ...] = ()


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
    unique_related: bool = False,
) -> list[Pick]:
    """A related prime is the most-mentioned prime article; the unrelated one
    neither mentions nor is mentioned by the target. Visits targets in a
    seeded order, so the selection is reproducible. With `unique_related`, a
    target whose related article is already taken is skipped, so cases stay
    independent for a document-level bootstrap."""
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
        if unique_related and best[0] in {p.related for p in picks}:
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
    *,
    cases: Sequence[Case],
    compute: Compute,
    fast_weights: FastWeights,
    generic: Carry | None = None,
    in_context: bool = False,
    chain_lengths: Sequence[int] = (),
) -> tuple[RelatedRow, ...]:
    """`generic` is installed as the carry for its arm; `in_context` puts the
    related prime in the same forward pass, unscored, with the carry reset;
    `chain_lengths` score the target after that many of `case.chain` carried."""
    fast_weights.set_mode(evolve=True, session=True)
    rows: list[RelatedRow] = []
    for case in cases:
        fast_weights.reset_carry()
        rows.append(make_row(RESET, case, "", compute.eval_loss(case.target)))
        for level in LEVELS:
            fast_weights.reset_carry()
            compute.eval_loss(case.primes[level])
            fast_weights.advance_carry()
            loss = compute.eval_loss(case.target)
            rows.append(make_row(level, case, case.prime_titles[level], loss))
        if generic is not None:
            fast_weights.reset_carry()
            fast_weights.install(generic)
            rows.append(make_row(GENERIC, case, GENERIC, compute.eval_loss(case.target)))
        if in_context:
            fast_weights.reset_carry()
            prime = case.primes[RELATED]
            loss = compute.eval_loss(prime + case.target, score_from=len(prime))
            rows.append(make_row(IN_CONTEXT, case, case.prime_titles[RELATED], loss))
        rows += _chain_rows(case, compute, fast_weights, chain_lengths)
    fast_weights.reset_carry()
    return tuple(rows)


def _chain_rows(case, compute, fast_weights, lengths) -> list[RelatedRow]:
    """One pass over the chain; the carry is snapshotted around each target so
    scoring it never feeds the rest of the chain."""
    if not lengths:
        return []
    if max(lengths) > len(case.chain):
        raise ValueError(f"case {case.index} has {len(case.chain)} chain articles, need {max(lengths)}")
    rows = []
    fast_weights.reset_carry()
    for read, article in enumerate(case.chain[: max(lengths)], start=1):
        compute.eval_loss(article)
        fast_weights.advance_carry()
        if read in lengths:
            state = fast_weights.snapshot()
            rows.append(make_row(f"{CHAIN}{read}", case, f"{read} articles", compute.eval_loss(case.target)))
            fast_weights.reset_carry()
            fast_weights.install(state)
    return rows


def generic_carry(carries: Sequence[Carry]) -> Carry:
    """Mean of `carries`, each layer rescaled to the median Frobenius norm of
    that layer across them, so averaging does not shrink the state."""
    if not carries:
        raise ValueError("a generic carry needs at least one carried state")
    deltas = {}
    for layer in carries[0].deltas:
        tensors = [c.deltas[layer] for c in carries]
        mean = sum(tensors[1:], tensors[0]) / len(tensors)
        norm = float(mean.norm())
        target = statistics.median(float(t.norm()) for t in tensors)
        deltas[layer] = mean * (target / norm) if norm > 0 else mean
    return Carry(deltas=deltas)


def shingles(text: str, *, n: int = 10, chars: int = 2000) -> set[str]:
    """Lower-cased n-word sequences of the opening text, for overlap checks."""
    words = [w.lower() for w in _WORD.findall(text[:chars])]
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlaps(text: str, seen: set[str], *, min_shared: int = 5) -> bool:
    return len(shingles(text) & seen) >= min_shared


def make_row(arm: str, case: Case, prime_title: str, loss: float) -> RelatedRow:
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
    extra = sorted({r.arm for r in rows} - {RESET, *LEVELS}, key=_arm_order)
    for name in extra:
        out += evidence.paired_estimates(f"gain_{name}", arm(RESET), arm(name), **kw)
    chains = [name for name in extra if name.startswith(CHAIN)]
    if len(chains) > 1:
        out += evidence.paired_estimates(f"{chains[-1]}_vs_{chains[0]}", arm(chains[0]), arm(chains[-1]), **kw)
    present = {r.arm for r in rows}
    for measure_name, reference, treated in HEADLINE:
        if {reference, treated} <= present:
            out += evidence.paired_estimates(measure_name, arm(reference), arm(treated), **kw)
            out += evidence.paired_estimates(
                f"{measure_name}_adjusted", arm(reference), arm(treated), level=ADJUSTED_LEVEL, **kw
            )
    return out


def gain_changes(
    candidate: Sequence[RelatedRow],
    baseline: Sequence[RelatedRow],
    *,
    arms: Sequence[str],
    rng: Rng,
    n_boot: int,
) -> tuple[Estimate, ...]:
    """Per arm, the candidate model's gain over its own reset minus the
    baseline model's, on the same cases (the mixed-session probe, §7)."""
    def arm(rows, name):
        return evidence.per_doc(rows, where=lambda r: r.arm == name)

    return tuple(
        evidence.gain_change(
            f"{name}_gain_change",
            (arm(candidate, RESET), arm(candidate, name)),
            (arm(baseline, RESET), arm(baseline, name)),
            rng=rng, n_boot=n_boot,
        )
        for name in arms
    )


def _arm_order(arm: str) -> tuple[int, int, str]:
    if arm.startswith(CHAIN):
        return (1, int(arm[len(CHAIN):]), "")
    return (0, 0, arm)
