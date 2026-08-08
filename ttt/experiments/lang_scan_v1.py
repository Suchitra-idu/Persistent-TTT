"""Base-model (no LoRA, no carry) perplexity across candidate languages —
which ones are actually weak spots, checked against the real model instead
of guessed (docs/experiments-map.md). CPU fetch, GPU forward pass.

    modal run ttt/experiments/lang_scan_v1.py
    modal run ttt/experiments/lang_scan_v1.py --languages "yi,ba,oc" --target-docs 10
"""

from __future__ import annotations

import contextlib
from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import lang_scan
from ttt.core import report
from ttt.core.config import lang_transfer
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-lang-scan-v1")
image = modal_runtime.build_image()

DEFAULT_TARGET_DOCS = 5
DEFAULT_MAX_CHARS = 3000


def run(
    *,
    engine: Engine,
    wiki_source,
    languages: tuple[tuple[str, str], ...] = lang_transfer.CANDIDATE_LANGUAGES,
    target_docs: int = DEFAULT_TARGET_DOCS,
    max_chars: int = DEFAULT_MAX_CHARS,
    announce: Callable[[str], None] = print,
):
    scans = lang_scan.scan(
        languages=languages,
        snapshot=lang_transfer.WIKI_SNAPSHOT,
        target_docs=target_docs,
        max_chars=max_chars,
        wiki_source=wiki_source,
        tokenizer=engine.tokenizer,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        announce=announce,
    )
    announce(report.render(report.language_scan_table(scans)))
    return scans


class _EvalCompute:
    """`eval_loss` only: a frozen model has no parameters to build an
    optimizer over (holdout_eval_v1's `_EvalCompute`, same shape)."""

    def __init__(self, model) -> None:
        import torch

        self._torch = torch
        self.model = model

    def eval_loss(self, token_ids, *, lora: bool = True) -> float:
        ids = self._torch.tensor([list(token_ids)], device="cuda", dtype=self._torch.long)
        ctx = contextlib.nullcontext() if lora else self.model.disable_adapter()
        with self._torch.no_grad(), ctx:
            return float(self.model(input_ids=ids, labels=ids).loss)


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=60 * 60,
)
@modal_runtime.caching
def lang_scan_fn(
    languages: str = "",
    target_docs: int = DEFAULT_TARGET_DOCS,
    max_chars: int = DEFAULT_MAX_CHARS,
    **flags,
):
    from ttt.adapters.wikipedia_lang_source import WikipediaLangSource

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    engine.compute = _EvalCompute(engine.model)
    picked = (
        tuple((c, c) for c in languages.split(",") if c)
        if languages
        else lang_transfer.CANDIDATE_LANGUAGES
    )
    run(
        engine=engine,
        wiki_source=WikipediaLangSource(),
        languages=picked,
        target_docs=target_docs,
        max_chars=max_chars,
    )


@app.local_entrypoint()
def main(
    languages: str = "",
    target_docs: int = DEFAULT_TARGET_DOCS,
    max_chars: int = DEFAULT_MAX_CHARS,
    flags: str = "",
):
    lang_scan_fn.remote(
        languages=languages,
        target_docs=target_docs,
        max_chars=max_chars,
        **cli.parse_flags(flags),
    )
