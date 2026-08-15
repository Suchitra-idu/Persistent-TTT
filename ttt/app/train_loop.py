"""The training loop: accumulate, clip, step, log, save, eval, carry.

Reaches the world only through ports, so the whole loop — the carry lifecycle
and the everlasting per-source carriers included — runs on fakes (D6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from ttt.app.data_pipeline import Doc
from ttt.core import lr_schedule, metrics
from ttt.core.config.train import TrainConfig
from ttt.core.types import Carry
from ttt.extensions.strategies import SOURCE, Strategy
from ttt.ports.compute import LORA, NEW, WDOWN, Compute
from ttt.ports.fast_weights import CARRY, FastWeights
from ttt.ports.tracker import MICRO_STEP, TRAIN_STEP, Tracker


@dataclass(frozen=True)
class TrainResult:
    steps: int
    micro_steps: int
    sessions: int
    nonfinite: int
    nonfinite_steps: int
    total_tokens: int
    total_steps: int
    carries: Mapping[str, Carry]
    n_updates: Mapping[str, int]


class Checkpointer(Protocol):
    def __call__(
        self,
        *,
        step: int,
        carries: Mapping[str, Carry],
        n_updates: Mapping[str, int],
    ) -> None: ...


class Evaluator(Protocol):
    def __call__(
        self, *, step: int, carries: Mapping[str, Carry]
    ) -> Mapping[str, float]: ...


def train(
    *,
    docs: Sequence[Doc],
    compute: Compute,
    fast_weights: FastWeights,
    tracker: Tracker,
    strategy: Strategy,
    cfg: TrainConfig,
    rng,
    carries: Mapping[str, Carry] | None = None,
    checkpoint: Checkpointer | None = None,
    evaluate: Evaluator | None = None,
    announce: Callable[[str], None] = lambda _: None,
) -> TrainResult:
    doc_lengths = [doc.n_tokens for doc in docs]
    total_steps = lr_schedule.total_optimizer_steps(
        strategy.count(doc_lengths), cfg.grad_accum_steps, cfg.num_epochs
    )
    warmup = lr_schedule.warmup_steps(
        total_steps, cfg.warmup_ratio, cfg.warmup_min_steps
    )
    per_source = strategy.carry_scope == SOURCE
    carried: dict[str, Carry] = dict(carries or {})
    n_updates: dict[str, int] = {}

    fast_weights.set_mode(evolve=True, stream=False, session=cfg.session_training)
    step = micro = sessions = nonfinite = nonfinite_steps = total_tokens = 0
    accumulated = 0
    step_loss = window_loss = 0.0

    for _ in range(cfg.num_epochs):
        for session in strategy.build(doc_lengths, rng):
            fast_weights.reset_carry()
            _seed(fast_weights, carried, docs, session, per_source)

            for item in session.items:
                doc = docs[item.doc_idx]
                token_ids = doc.token_ids[item.start : item.end]
                if micro == 0:
                    announce(f"  first micro-step: {len(token_ids):,d} tokens, starting forward")
                loss = compute.loss(token_ids)
                if micro == 0:
                    announce("  first micro-step: forward+backward-eligible loss returned")
                micro += 1
                total_tokens += len(token_ids)

                if math.isfinite(loss):
                    compute.backward(scale=1.0 / cfg.grad_accum_steps)
                    accumulated += 1
                    step_loss += loss / cfg.grad_accum_steps
                else:
                    nonfinite += 1

                fast_weights.advance_carry()
                if per_source:
                    carried[doc.source] = fast_weights.snapshot()
                    n_updates[doc.source] = n_updates.get(doc.source, 0) + 1

                tracker.log(
                    {
                        MICRO_STEP: micro,
                        "micro/doc_loss": loss,
                        "micro/state_ratio_mean": fast_weights.state_ratio(
                            family=CARRY
                        ),
                    }
                )

                if micro % cfg.grad_accum_steps:
                    continue

                # An all-nonfinite window has no gradient to step on, and AdamW
                # would still apply weight decay to every parameter.
                if not accumulated:
                    compute.zero_grad()
                    step_loss = 0.0
                    continue

                rates = _learning_rates(cfg, step, warmup, total_steps)
                stats = compute.clip_and_step(
                    max_grad_norm=cfg.max_grad_norm, learning_rates=rates
                )
                compute.zero_grad()
                if not math.isfinite(stats.total_norm):
                    # The optimizer never stepped (see TorchCompute.clip_and_step):
                    # a finite loss can still backward into a non-finite gradient,
                    # and applying that would make every parameter nan forever.
                    nonfinite_steps += 1
                    step_loss = 0.0
                    accumulated = 0
                    announce(f"  step {step + 1}: non-finite gradient, skipped")
                    continue
                step += 1
                accumulated = 0
                tracker.log(_step_metrics(step, step_loss, rates, stats, cfg))
                window_loss += step_loss
                step_loss = 0.0

                if step % cfg.log_every == 0:
                    announce(_progress(step, total_steps, window_loss, cfg, stats))
                    window_loss = 0.0
                if checkpoint is not None and step % cfg.save_every == 0:
                    checkpoint(step=step, carries=carried, n_updates=n_updates)
                if evaluate is not None and cfg.eval_every > 0:
                    _maybe_eval(step, cfg, evaluate, tracker, announce, carried)
            sessions += 1

    if checkpoint is not None:
        checkpoint(step=step, carries=carried, n_updates=n_updates)
    tracker.finish()
    return TrainResult(
        steps=step,
        micro_steps=micro,
        sessions=sessions,
        nonfinite=nonfinite,
        nonfinite_steps=nonfinite_steps,
        total_tokens=total_tokens,
        total_steps=total_steps,
        carries=carried,
        n_updates=n_updates,
    )


def _seed(fast_weights, carried, docs, session, per_source: bool) -> None:
    """The everlasting carrier for this session's source, if it has one. A
    source seen for the first time starts from zero, as training did."""
    if not per_source:
        return
    seed = carried.get(docs[session.items[0].doc_idx].source)
    if seed is not None:
        fast_weights.install(seed)


def _learning_rates(
    cfg: TrainConfig, step: int, warmup: int, total_steps: int
) -> dict[str, float]:
    """`step` is the count before this one, so the rate logged is the rate used."""
    scale = lr_schedule.multiplier(
        step, num_warmup_steps=warmup, num_training_steps=total_steps
    )
    return {
        LORA: cfg.lr_lora * scale,
        WDOWN: cfg.lr_wdown * scale,
        NEW: cfg.lr_new_modules * scale,
    }


def _step_metrics(step, step_loss, rates, stats, cfg) -> dict[str, float]:
    return {
        TRAIN_STEP: step,
        "train/loss": step_loss,
        "train/lr_lora": rates[LORA],
        "train/lr_wdown": rates[WDOWN],
        "train/lr_new": rates[NEW],
        "train/grad_clip_ratio": metrics.clip_ratio(
            stats.total_norm, cfg.max_grad_norm
        ),
        "grad/lora": stats.lora,
        "grad/wdown": stats.wdown,
        "grad/new": stats.new,
    }


def _maybe_eval(step, cfg, evaluate, tracker, announce, carried) -> None:
    if step % cfg.eval_every:
        return
    measured = evaluate(step=step, carries=carried)
    tracker.log({TRAIN_STEP: step, **measured})
    announce(_eval_line(measured))


def _progress(step, total_steps, window_loss, cfg, stats) -> str:
    return (
        f"step {step}/{total_steps} loss {window_loss / cfg.log_every:.4f} "
        f"|g| lora {stats.lora:.3f} wdown {stats.wdown:.3f} new {stats.new:.3f}"
    )


def _eval_line(measured: Mapping[str, float]) -> str:
    return "  [eval] " + " ".join(
        f"{key.split('/', 1)[1]} {value:.4g}"
        for key, value in sorted(measured.items())
    )
