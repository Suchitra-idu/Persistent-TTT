"""The composition root: resolved config in, wired ports out.

Not an experiment — this is the infrastructure the `*_v1.py` files compose over,
and the only file here that may change. An `Engine` is a bag of ports plus the
two things only a real model can do: persist itself, and be resumed. Tests build
one out of fakes and drive a whole experiment with no GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ttt import cli
from ttt.adapters import checkpoint_io
from ttt.adapters.numpy_rng import NumpyRng
from ttt.core.types import Carry
from ttt.ports.clock import Clock
from ttt.ports.compute import Compute
from ttt.ports.fast_weights import FastWeights
from ttt.ports.generation import Generation
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
    generation: Generation | None = None
    tracker: Tracker | None = None
    storage: Storage | None = None
    clock: Clock | None = None
    model: Any = None
    root: str = ""
    save: Callable[..., None] = _no_save

    def rng(self, seed: int | None = None) -> Rng:
        return NumpyRng(self.resolved.train.seed if seed is None else seed)

    def carries(self) -> tuple[dict[str, Carry], dict[str, Any]]:
        """The trained per-source carriers to resume or seed from, or ({}, {})."""
        resume = _resume_dir(self.resolved)
        if not resume or self.storage is None:
            return {}, {}
        return checkpoint_io.load_carries(
            self.storage, f"{resume}/{checkpoint_io.CARRIES}"
        )


def build(
    resolved: cli.Resolved, *, storage: Storage, root: str, trainable: bool
) -> Engine:
    """The real thing: transformers, PEFT, torch adapters. Needs a GPU.

    `root` is the storage adapter's mount path, passed separately because PEFT
    writes an adapter *directory* and the Storage port speaks in blobs.
    """
    from ttt.adapters.hf_tokenizer import HfTokenizer
    from ttt.adapters.model_builder import build_model
    from ttt.adapters.system_clock import SystemClock
    from ttt.adapters.torch_compute import TorchCompute
    from ttt.adapters.torch_fast_weights import TorchFastWeights
    from ttt.adapters.torch_generation import TorchGeneration

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
        generation=TorchGeneration(model),
        storage=storage,
        clock=SystemClock(),
        model=model,
        root=root,
    )
    engine.save = _checkpointer(engine)
    return engine


def _checkpointer(engine: Engine) -> Callable[..., None]:
    """Both artefacts, the adapter, then one commit — in that order, so a
    partially written step directory is never visible as a complete one."""

    def save(*, step: int, carries: Mapping[str, Carry], n_updates: Mapping[str, int]):
        prefix = cli.step_path(engine.resolved.train.run_name, step)
        checkpoint_io.save_ttt_params(
            engine.model,
            engine.storage,
            f"{prefix}/{checkpoint_io.TTT_PARAMS}",
            layer_indices=engine.resolved.ttt.layer_indices or (),
        )
        checkpoint_io.save_carries(
            carries,
            engine.storage,
            f"{prefix}/{checkpoint_io.CARRIES}",
            n_updates=n_updates,
        )
        engine.model.save_pretrained(f"{engine.root}/{prefix}/{ADAPTER_DIR}")
        engine.storage.commit()

    return save


def _resume_dir(resolved: cli.Resolved) -> str:
    return cli.resume_path(resolved.resume_from, resolved.train.run_name)


def tracker(resolved: cli.Resolved, *, job_type: str) -> Tracker:
    """Console when wandb is off, so a run without telemetry still reports."""
    from ttt.adapters.console_tracker import ConsoleTracker

    if not resolved.train.wandb_enabled:
        return ConsoleTracker(every=resolved.train.log_every)

    from ttt.adapters.wandb_tracker import WandbTracker

    return WandbTracker.start(
        project=resolved.train.wandb_project,
        run_name=resolved.train.run_name,
        job_type=job_type,
        config=dict(resolved.describe()),
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
