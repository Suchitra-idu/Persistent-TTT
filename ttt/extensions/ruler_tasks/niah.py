"""Needle-in-a-haystack, ported from NVIDIA/RULER (Apache-2.0) niah.py,
`type_haystack='essay'` / `type_needle_v='numbers'`.

Haystack words come from `RulerContext` (this repo's own corpus), not a
vendored Paul Graham essay file — RULER itself doesn't ship one either, and
`recall_score` doesn't care which real prose fills the haystack.

Four registered variants differ only in needle/query counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from ttt.core.ruler_scoring import recall_score
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks import _common
from ttt.extensions.ruler_tasks._registry import register

NEEDLE = "One of the special magic numbers for {key} is: {value}."
TEMPLATE = (
    "Some special magic numbers are hidden within the following text. Make "
    "sure to memorize them. I will quiz you about the numbers afterwards.\n"
    "{context}\n"
    "What are all the special magic numbers for {query} mentioned in the "
    "provided text?"
)
ANSWER_PREFIX = (
    " The special magic numbers for {query} mentioned in the provided text are"
)
CHUNK_WORDS = 20
TOKENS_TO_GENERATE = 128
NUM_DIGITS = 7


def _key(rng, ctx: RulerContext) -> str:
    pool = [w for w in ctx.haystack_words if w.isalpha() and len(w) >= 3]
    if len(pool) < 2:
        raise ValueError("haystack_words has too few usable words for a needle key")
    a, b = _common.sample(rng, pool, 2)
    return f"{a.lower()}-{b.lower()}"


def _value(rng) -> str:
    return str(rng.integers(10 ** (NUM_DIGITS - 1), 10**NUM_DIGITS))


def _chunks(words: list[str]) -> list[str]:
    return [
        " ".join(words[i : i + CHUNK_WORDS])
        for i in range(0, len(words), CHUNK_WORDS)
    ]


def _build(
    rng,
    tokenizer,
    seq_len: int,
    ctx: RulerContext,
    *,
    task_name: str,
    num_needle_k: int,
    num_needle_v: int,
    num_needle_q: int,
) -> RulerExample:
    keys = [_key(rng, ctx) for _ in range(num_needle_k)]
    values = [[_value(rng) for _ in range(num_needle_v)] for _ in keys]
    needles = [
        NEEDLE.format(key=key, value=value)
        for key, vals in zip(keys, values)
        for value in vals
    ]
    order = rng.permutation(len(needles))
    needles = [needles[i] for i in order]

    query_idx = sorted(_common.sample_indices(rng, len(keys), num_needle_q))
    queries = [keys[i] for i in query_idx]
    answers = tuple(v for i in query_idx for v in values[i])
    query = queries[0] if len(queries) == 1 else ", ".join(queries[:-1]) + f", and {queries[-1]}"

    def make(num_words: int) -> tuple[str, tuple[str, ...]]:
        chunks = _chunks(_common.repeated(ctx.haystack_words, num_words))
        context = _common.interleave(rng, chunks, needles)
        prompt = TEMPLATE.format(context=context, query=query)
        return prompt, answers

    prompt, targets = _common.fit_size(
        make,
        tokenizer=tokenizer,
        seq_len=seq_len,
        reserve=TOKENS_TO_GENERATE,
        start=CHUNK_WORDS * (len(needles) + 2),
    )
    return RulerExample(
        task=task_name,
        length_bucket=seq_len,
        prompt=prompt,
        answer_prefix=ANSWER_PREFIX.format(query=query),
        targets=targets,
    )


@dataclass(frozen=True)
class _Niah:
    name: str
    num_needle_k: int
    num_needle_v: int
    num_needle_q: int
    version: int = 1

    def build(self, rng, tokenizer, seq_len, ctx) -> RulerExample:
        return _build(
            rng,
            tokenizer,
            seq_len,
            ctx,
            task_name=self.name,
            num_needle_k=max(self.num_needle_k, self.num_needle_q),
            num_needle_v=self.num_needle_v,
            num_needle_q=self.num_needle_q,
        )

    def score(self, prediction: str, example: RulerExample) -> float:
        return recall_score(prediction, example.targets)


NIAH_SINGLE = register(_Niah("niah_single", 1, 1, 1))
NIAH_MULTIKEY = register(_Niah("niah_multikey", 4, 1, 1))
NIAH_MULTIVALUE = register(_Niah("niah_multivalue", 1, 4, 1))
NIAH_MULTIQUERY = register(_Niah("niah_multiquery", 4, 1, 4))
