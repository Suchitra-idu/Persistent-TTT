"""Does the carry compound with session length, and does the trained seed help?

Three regimes over one document sequence: cold (reset per document), persist,
and seeded. Emits JSON; plot it with `ttt/experiments/plot_pilot.py`.

    modal run ttt/experiments/compounding_pilot_v1.py --resume-from step_600
"""

from __future__ import annotations

import dataclasses
import json
from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, pilot
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-compounding-pilot-v1")
image = modal_runtime.build_image()

PILOT_DIR = "pilot"


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    pilot_source: str = "",
    n_docs: int = 10,
    announce: Callable[[str], None] = print,
) -> tuple[pilot.PilotRow, ...]:
    """One source at a time: compounding rates are what differ between domains,
    and a mixed sequence would average them away."""
    docs = _pool(resolved, engine, source, pilot_source, n_docs)
    if not docs:
        announce(f"no held-out documents for source {pilot_source!r}")
        return ()

    carries, meta = engine.carries()
    seed = carries.get(docs[0].source)
    if seed is None:
        announce(
            f"no trained carrier for {docs[0].source!r} "
            f"(checkpoint has {sorted(carries)}); the seeded regime is skipped"
        )

    rows = pilot.run(
        docs=docs, compute=engine.compute, fast_weights=engine.fast_weights, seed=seed
    )
    announce(f"{len(docs)} documents, {sum(d.n_tokens for d in docs):,d} tokens")
    for row in rows:
        announce(
            f"  [{row.regime:>7s}] pos {row.position:>2d} "
            f"nll {row.nll:.4f} ppl {row.ppl:.3f} state/W0 {row.state_ratio:.3e}"
        )
    if meta.get("n_updates"):
        announce(f"  carrier updates: {dict(sorted(meta['n_updates'].items()))}")
    return rows


def _pool(resolved, engine, source, pilot_source, n_docs):
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    matching = [
        doc for doc in holdout.docs if not pilot_source or doc.source == pilot_source
    ]
    return matching[:n_docs]


def payload(resolved: cli.Resolved, rows) -> bytes:
    """One record per (regime, position) — analysis happens off the container."""
    return json.dumps(
        {
            "config": dict(resolved.describe()),
            "resume_from": resolved.resume_from,
            "rows": [dataclasses.asdict(row) for row in rows],
        },
        indent=2,
    ).encode()


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
def compounding_pilot(pilot_source: str = "", n_docs: int = 10, out_name: str = "", **flags):
    from ttt.adapters.hf_data_source import HfDataSource
    from ttt.experiments.holdout_eval_v1 import _EvalCompute

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    storage = modal_runtime.checkpoint_storage()
    engine = _runtime.build(
        resolved, storage=storage, root=modal_runtime.CKPT_MOUNT, trainable=False
    )
    engine.compute = _EvalCompute(engine.model)

    rows = run(
        resolved,
        engine=engine,
        source=HfDataSource(),
        pilot_source=pilot_source,
        n_docs=n_docs,
    )
    path = f"{PILOT_DIR}/{out_name or _default_name(engine, pilot_source, len(rows))}"
    storage.write_bytes(path, payload(resolved, rows))
    storage.commit()
    print(f"saved -> {path}")
    return path


def _default_name(engine: Engine, pilot_source: str, n_rows: int) -> str:
    return f"compounding_{pilot_source or 'all'}_{n_rows}_{engine.clock.stamp()}.json"


@app.local_entrypoint()
def main(pilot_source: str = "", n_docs: int = 10, out_name: str = "", flags: str = ""):
    compounding_pilot.remote(
        pilot_source=pilot_source,
        n_docs=n_docs,
        out_name=out_name,
        **cli.parse_flags(flags),
    )
