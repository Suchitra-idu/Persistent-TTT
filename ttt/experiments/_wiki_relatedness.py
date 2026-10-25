"""Relatedness cases from English Wikipedia, for `relatedness_eval`.

The dump's parquet shards are read directly; character floors keep the pools
to articles long enough for the token blocks, before any tokenizing.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ttt.app.data_pipeline import Doc
from ttt.app.relatedness_eval import (
    OTHER_SOURCE,
    RANDOM,
    RELATED,
    SELF,
    Article,
    Case,
    Pick,
    pick,
)
from ttt.ports.rng import Rng
from ttt.ports.tokenizer import Tokenizer

SHARD_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.en/train-{index:05d}-of-00041.parquet"
)
PRIME_TOKENS = 1024
TARGET_TOKENS = 1024
# About 4 characters per token in English Wikipedia; generous so the token check rarely fails.
TARGET_MIN_CHARS = 12_000
PRIME_MIN_CHARS = 6_000


def load_articles(n_shards: int) -> list[Article]:
    from datasets import load_dataset

    files = [SHARD_URL.format(index=i) for i in range(n_shards)]
    table = load_dataset("parquet", data_files=files, split="train")
    table = table.filter(lambda row: len(row["text"]) >= PRIME_MIN_CHARS)
    return [Article(title=t, text=x) for t, x in zip(table["title"], table["text"])]


def build_cases(
    *,
    n_targets: int,
    n_shards: int,
    tokenizer: Tokenizer,
    other_docs: Sequence[Doc],
    rng: Rng,
    announce: Callable[[str], None] = print,
) -> tuple[list[Case], list[Pick]]:
    articles = load_articles(n_shards)
    targets = [i for i, a in enumerate(articles) if len(a.text) >= TARGET_MIN_CHARS]
    primes = list(range(len(articles)))
    announce(f"  wikipedia: {len(articles):,d} articles, {len(targets):,d} long enough as targets")
    long_docs = [d for d in other_docs if d.n_tokens >= PRIME_TOKENS]
    if not long_docs:
        raise ValueError("no other-source document is long enough to prime with")

    # Over-pick, since a few targets fall short of the token budget once tokenized.
    picked = pick(articles, targets=targets, primes=primes, n=n_targets + n_targets // 4, rng=rng)
    cases: list[Case] = []
    used: list[Pick] = []
    for p in picked:
        if len(cases) == n_targets:
            break
        target = tokenizer.encode(articles[p.target].text, max_length=PRIME_TOKENS + TARGET_TOKENS)
        related = tokenizer.encode(articles[p.related].text, max_length=PRIME_TOKENS)
        unrelated = tokenizer.encode(articles[p.unrelated].text, max_length=PRIME_TOKENS)
        if len(target) < PRIME_TOKENS + TARGET_TOKENS or min(len(related), len(unrelated)) < PRIME_TOKENS:
            continue
        other = long_docs[len(cases) % len(long_docs)]
        cases.append(Case(
            index=len(cases),
            target_title=articles[p.target].title,
            target=tuple(target[PRIME_TOKENS:]),
            primes={
                SELF: tuple(target[:PRIME_TOKENS]),
                RELATED: tuple(related),
                RANDOM: tuple(unrelated),
                OTHER_SOURCE: tuple(other.token_ids[:PRIME_TOKENS]),
            },
            prime_titles={
                SELF: articles[p.target].title,
                RELATED: articles[p.related].title,
                RANDOM: articles[p.unrelated].title,
                OTHER_SOURCE: f"{other.source}#{other.index}",
            },
        ))
        used.append(p)
    announce(f"  relatedness: {len(cases)} cases")
    return cases, used
