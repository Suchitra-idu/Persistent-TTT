"""Language-transfer eval's offline half: pull each language once, then
merge and shuffle them into the one file a DatasetSpec reads. CPU only, no GPU.

    modal run ttt/experiments/lang_corpus_prepare_v1.py --role train
    modal run ttt/experiments/lang_corpus_prepare_v1.py --role eval
"""

from __future__ import annotations

from typing import Callable

import modal

from ttt.adapters import modal_runtime
from ttt.core.config import lang_transfer
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage

app = modal.App("ttt-lang-corpus-prepare-v1")
image = modal_runtime.build_image()


def run(
    role: str,
    target_rows: int,
    *,
    wiki_source,
    storage: Storage,
    rng: Rng,
    announce: Callable[[str], None] = print,
) -> int:
    from ttt.app import lang_corpus_prepare

    languages = lang_transfer.languages_for(role)
    lang_corpus_prepare.prepare(
        languages=languages,
        snapshot=lang_transfer.WIKI_SNAPSHOT,
        wiki_source=wiki_source,
        storage=storage,
        role=role,
        target_rows=target_rows,
        announce=announce,
    )
    return lang_corpus_prepare.combine(
        languages=languages, storage=storage, role=role, rng=rng, announce=announce
    )


@app.function(
    image=image,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=60 * 60,
)
@modal_runtime.caching
def lang_corpus_prepare(role: str = "train", target_rows: int = lang_transfer.DEFAULT_TARGET_ROWS, seed: int = 0):
    from ttt.adapters.numpy_rng import NumpyRng
    from ttt.adapters.wikipedia_lang_source import WikipediaLangSource

    return run(
        role,
        target_rows,
        wiki_source=WikipediaLangSource(),
        storage=modal_runtime.checkpoint_storage(),
        rng=NumpyRng(seed),
    )


@app.local_entrypoint()
def main(role: str = "train", target_rows: int = lang_transfer.DEFAULT_TARGET_ROWS, seed: int = 0, flags: str = ""):
    # This phase needs no base_model/checkpoint resolution; flags exists only
    # because every entrypoint must accept it (tests/architecture/test_entrypoints.py).
    del flags
    lang_corpus_prepare.remote(role=role, target_rows=target_rows, seed=seed)
