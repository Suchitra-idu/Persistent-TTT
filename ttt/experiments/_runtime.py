"""The composition root: resolved config in, wired ports out.

Not an experiment — this is the infrastructure the `*_v1.py` files compose over,
and the only file here that may change. An `Engine` is a bag of ports plus the
two things only a real model can do: persist itself, and be resumed. Tests build
one out of fakes and drive a whole experiment with no GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from ttt import cli
from ttt.adapters import checkpoint_io
from ttt.adapters.numpy_rng import NumpyRng
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage
from ttt.ports.tokenizer import Tokenizer
from ttt.ports.tracker import Tracker

ADAPTER_DIR = "adapter"


def _no_save(**_: Any) -> None:
    """A run with nowhere to write is a smoke test, not a failure."""


@dataclass
class Engine:
    resolved: cli.Resolved
    tokenizer: Tokenizer
    compute: Compute | None = None
    fast_weights: FastWeights | None = None
    tracker: Tracker | None = None
    storage: Storage | None = None
    model: Any = None
    root: str = ""
    save: Callable[..., None] = _no_save

    def rng(self, seed: int | None = None) -> Rng:
        return NumpyRng(self.resolved.train.seed if seed is None else seed)


def build(
    resolved: cli.Resolved,
    *,
    storage: Storage,
    root: str,
    trainable: bool,
    announce: Callable[[str], None] = lambda _: None,
) -> Engine:
    """The real thing: transformers, PEFT, torch adapters. Needs a GPU.

    `root` is the storage adapter's mount path, passed separately because PEFT
    writes an adapter *directory* and the Storage port speaks in blobs.
    """
    from ttt.adapters.hf_tokenizer import HfTokenizer
    from ttt.adapters.model_builder import build_model
    from ttt.adapters.torch_compute import TorchCompute
    from ttt.adapters.torch_fast_weights import TorchFastWeights

    import torch

    # LoRA's initialisation and dropout draw from torch's global RNG.
    torch.manual_seed(resolved.train.seed)
    resume = _resume_dir(resolved)
    model, ttt_cfg = build_model(
        resolved.base_model,
        ttt_cfg=resolved.ttt,
        train_cfg=resolved.train,
        adapter_path=f"{root}/{resume}/{ADAPTER_DIR}" if resume else None,
        trainable=trainable,
    )
    if resume:
        checkpoint_io.load_ttt_params(
            model, storage, f"{resume}/{checkpoint_io.TTT_PARAMS}"
        )

    engine = Engine(
        resolved=dataclasses.replace(resolved, ttt=ttt_cfg),
        tokenizer=HfTokenizer.from_pretrained(resolved.base_model),
        compute=(
            TorchCompute(model, ttt_cfg=ttt_cfg, train_cfg=resolved.train)
            if trainable
            else None
        ),
        fast_weights=TorchFastWeights(model, ttt_cfg),
        storage=storage,
        model=model,
        root=root,
    )
    engine.save = _checkpointer(engine, announce)
    return engine


def _checkpointer(engine: Engine, announce: Callable[[str], None]) -> Callable[..., None]:
    """The TTT parameters, the adapter, then one commit — in that order, so a
    partially written step directory is never visible as a complete one."""

    def save(*, step: int):
        prefix = cli.step_path(engine.resolved.train.run_name, step)
        checkpoint_io.save_ttt_params(
            engine.model,
            engine.storage,
            f"{prefix}/{checkpoint_io.TTT_PARAMS}",
            layer_indices=engine.resolved.ttt.layer_indices or (),
        )
        engine.model.save_pretrained(f"{engine.root}/{prefix}/{ADAPTER_DIR}")
        engine.storage.commit()
        announce(f"  checkpoint saved: {prefix}")

    return save


def _resume_dir(resolved: cli.Resolved) -> str:
    return cli.resume_path(resolved.resume_from, resolved.train.run_name)


def tracker(resolved: cli.Resolved, *, job_type: str, invocation: str = "") -> Tracker:
    """Console when wandb is off, so a run without telemetry still reports.

    `invocation` is the exact command line that launched the run (`cli.invocation()`,
    read locally and threaded in — see its docstring), stored in the wandb
    run's config for provenance.
    """
    from ttt.adapters.console_tracker import ConsoleTracker

    if not resolved.train.wandb_enabled:
        return ConsoleTracker(every=resolved.train.log_every)

    from ttt.adapters.wandb_tracker import WandbTracker

    config = dict(resolved.describe())
    if invocation:
        config["invocation"] = invocation
    return WandbTracker.start(
        project=resolved.train.wandb_project,
        run_name=resolved.train.run_name,
        job_type=job_type,
        config=config,
    )


def announce_boot(engine: Engine, docs, announce: Callable[[str], None]) -> None:
    """The boot log: what was resolved, what the pool looks like, what will run."""
    resolved = engine.resolved
    announce(f"{resolved.base_model}  dataset={resolved.spec.name}")
    announce(resolved.strategy.describe())
    announce(
        f"{len(docs)} documents, {sum(d.n_tokens for d in docs):,d} tokens, "
        f"seed={resolved.train.seed}"
    )
    if engine.compute is not None:
        counts = engine.compute.parameter_counts()
        announce(
            "trainable "
            + ", ".join(f"{name} {counts[name] / 1e6:.1f}M" for name in sorted(counts))
        )
