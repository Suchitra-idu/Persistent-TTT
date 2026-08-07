"""RULER's offline half: synthesize fixed example sets once. CPU only, no GPU.

    modal run ttt/experiments/ruler_prepare_v1.py \
        --ruler-flags "tasks=niah_single,context_lengths=4096"
"""

from __future__ import annotations

from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.adapters.numpy_rng import NumpyRng
from ttt.core.config.ruler import RulerConfig, parse_ruler_flags
from ttt.ports.data_source import DataSource
from ttt.ports.storage import Storage
from ttt.ports.tokenizer import Tokenizer

app = modal.App("ttt-ruler-prepare-v1")
image = modal_runtime.build_image()


def run(
    resolved: cli.Resolved,
    cfg: RulerConfig,
    *,
    tokenizer: Tokenizer,
    source: DataSource,
    storage: Storage,
    qa_source=None,
    announce: Callable[[str], None] = print,
) -> dict[str, tuple]:
    from ttt.app import ruler_prepare
    from ttt.extensions.ruler_tasks import RULER_TASKS

    ctx = ruler_prepare.build_context(
        source=source,
        spec=resolved.spec,
        qa_source=qa_source if "qa" in cfg.tasks else None,
    )
    return ruler_prepare.prepare(
        cfg=cfg,
        tasks=RULER_TASKS,
        tokenizer=tokenizer,
        ctx=ctx,
        rng=NumpyRng(cfg.seed),
        storage=storage,
        announce=announce,
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
def ruler_prepare(ruler_flags: str = "", **flags):
    from ttt.adapters.hf_data_source import HfDataSource
    from ttt.adapters.hf_tokenizer import HfTokenizer
    from ttt.adapters.squad_qa_source import SquadQaSource

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    built = run(
        resolved,
        parse_ruler_flags(ruler_flags),
        tokenizer=HfTokenizer.from_pretrained(resolved.base_model),
        source=HfDataSource(),
        storage=modal_runtime.checkpoint_storage(),
        qa_source=SquadQaSource(),
    )
    return sorted(built)


@app.local_entrypoint()
def main(ruler_flags: str = "", flags: str = ""):
    ruler_prepare.remote(ruler_flags=ruler_flags, **cli.parse_flags(flags))
