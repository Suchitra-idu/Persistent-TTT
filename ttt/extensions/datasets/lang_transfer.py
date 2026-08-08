"""TRAIN_LANGS / EVAL_LANGS — the language-transfer eval's two corpora, kept
separate so an eval-only language can never reach training (docs/
experiments-map.md). `/ckpt` duplicates `adapters.modal_runtime.CKPT_MOUNT`:
Ring 1 extensions import core only."""

from __future__ import annotations

from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.lang_transfer import (
    EVAL_LANGUAGES,
    TRAIN_HOLDOUT_LAST_N,
    TRAIN_LANGUAGES,
    combined_dir,
)
from ttt.extensions.datasets._registry import register

_CKPT_MOUNT = "/ckpt"

TRAIN_LANGS = register(
    DatasetSpec(
        name="lang-transfer-train",
        source=f"{_CKPT_MOUNT}/{combined_dir('train')}",
        source_meta_column="meta",
        source_meta_key="lang",
        include_sources=tuple(code for code, _ in TRAIN_LANGUAGES),
        default_source_preset="lang-transfer-balanced",
        # train_v1.py's periodic in-loop eval reads this spec's own holdout,
        # not EVAL_LANGS — TRAIN_HOLDOUT_LAST_N is sized off DEFAULT_TARGET_ROWS
        # so it can't silently go stale if that changes; shuffled by
        # `combine()`, so every language has a realistic chance of showing up.
        holdout_last_n=TRAIN_HOLDOUT_LAST_N,
    )
)

EVAL_LANGS = register(
    DatasetSpec(
        name="lang-transfer-eval",
        source=f"{_CKPT_MOUNT}/{combined_dir('eval')}",
        source_meta_column="meta",
        source_meta_key="lang",
        include_sources=tuple(code for code, _ in EVAL_LANGUAGES),
        # Larger than this corpus will ever be: this spec is read only through
        # `holdout()`, so every row it has should be eval-eligible, not just
        # the tail past some train/eval cut it will never be trained through.
        holdout_last_n=1_000_000,
    )
)
