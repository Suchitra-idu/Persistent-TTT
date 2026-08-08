"""Cheap base-model probe across candidate languages — not the trained
lang-transfer corpora, just a scout for which ones are worth adding to them
(docs/experiments-map.md)."""

from __future__ import annotations

import math
from typing import Callable, Protocol, Sequence

from ttt.core.types import LanguageScan
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.tokenizer import Tokenizer


class LangSource(Protocol):
    def load(self, code: str, snapshot: str, limit: int) -> list[dict]: ...


def scan(
    *,
    languages: Sequence[tuple[str, str]],
    snapshot: str,
    target_docs: int,
    max_chars: int,
    wiki_source: LangSource,
    tokenizer: Tokenizer,
    compute: Compute,
    fast_weights: FastWeights,
    announce: Callable[[str], None] = print,
) -> tuple[LanguageScan, ...]:
    results = []
    for code, name in languages:
        rows = wiki_source.load(code, snapshot, target_docs)
        measured = [_measure(row, max_chars, tokenizer, compute, fast_weights) for row in rows]
        measured = [m for m in measured if m is not None]
        if not measured:
            announce(f"{code} ({name}): no usable rows, skipped")
            continue
        total_nll = sum(nll for nll, _, _ in measured)
        n_tokens = sum(n for _, n, _ in measured)
        n_bytes = sum(b for _, _, b in measured)
        result = LanguageScan(
            code=code,
            name=name,
            n_docs=len(measured),
            n_tokens=n_tokens,
            n_bytes=n_bytes,
            ppl=math.exp(total_nll / n_tokens),
            bpb=total_nll / math.log(2) / n_bytes,
        )
        results.append(result)
        announce(f"{code} ({name}): {result.ppl:.1f} ppl, {result.bpb:.3f} bpb, {result.n_docs} docs")
    return tuple(results)


def _measure(row, max_chars, tokenizer, compute, fast_weights):
    text = row["text"][:max_chars]
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return None
    fast_weights.set_mode(evolve=False, stream=False, session=True)
    fast_weights.reset_carry()
    loss = compute.eval_loss(ids, lora=False)
    # `metrics.bits_per_byte`'s own convention: nats-per-token times token
    # count, not count-1 — kept consistent so this scan's numbers compose
    # with the rest of the eval machinery if it ever needs to.
    return loss * len(ids), len(ids), len(text.encode("utf-8"))
