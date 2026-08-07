"""RULER's two metrics: recall (NIAH/VT/CWE/FWE) and token-F1 (QA).

Both are RULER's own metrics, not this repo's invention — recall over exact
target strings for the synthetic tasks, SQuAD-style best-of-targets F1 for QA.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def recall_score(prediction: str, targets: Sequence[str]) -> float:
    """Fraction of `targets` present verbatim (case-insensitive) in `prediction`."""
    if not targets:
        raise ValueError("recall_score needs at least one target")
    lowered = prediction.lower()
    hits = sum(1 for target in targets if target.lower() in lowered)
    return hits / len(targets)


def qa_score(prediction: str, targets: Sequence[str]) -> float:
    """Best token-F1 of `prediction` against any of `targets`."""
    if not targets:
        raise ValueError("qa_score needs at least one target")
    return max(_token_f1(prediction, target) for target in targets)


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = _normalize(prediction).split()
    target_tokens = _normalize(target).split()
    if not pred_tokens or not target_tokens:
        return float(pred_tokens == target_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub("", text.lower())).strip()
