"""RULER's GPU half: score prepared examples under each FastWeights regime.

`app/generate.py` runs unchanged; only the fast-weight mode around it differs.
`COLD_CARRY` (evolve=True) is the TTT-on arm, `FRESH` (evolve=False) is the
paper's implicit base-model arm — see `core/types.py`'s regime vocabulary.

Runs through the STREAM family (`stream=True`), not CARRY/`_scan_forward` —
`_scan_forward` materializes one `[num_chunks, d_model, d_ff]` tensor per
patched layer for the whole prompt at once (OOMs well before 32k tokens on
an H100); `_stream_forward` is chat's existing O(1)-in-length path, one
running `[d_model, d_ff]` state committed and discarded chunk by chunk.
Known gap versus the scan path: a trailing partial chunk (< chunk_size
tokens, at most) is never committed — negligible since the needle is never
in the last few tokens of a prompt (the query/answer_prefix is).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ttt.app import generate
from ttt.core import sampling_text
from ttt.core.config.ruler import BASE, INSTRUCT, RulerConfig
from ttt.core.ruler_types import RulerExample, RulerResult
from ttt.core.types import COLD_CARRY
from ttt.extensions.ruler_tasks._registry import RulerTask
from ttt.ports.fast_weights import FastWeights
from ttt.ports.generation import Generation
from ttt.ports.tokenizer import Tokenizer


@dataclass(frozen=True)
class RulerReport:
    results: tuple[RulerResult, ...]
    metrics: Mapping[str, float]


def evaluate(
    *,
    examples: Mapping[str, Sequence[RulerExample]],
    tasks: Mapping[str, RulerTask],
    cfg: RulerConfig,
    generation: Generation,
    fast_weights: FastWeights,
    tokenizer: Tokenizer,
    announce: Callable[[str], None] = print,
) -> RulerReport:
    stop_ids = sampling_text.stop_token_ids(eos_token_id=tokenizer.eos_token_id)
    restore = fast_weights.snapshot()
    results: list[RulerResult] = []
    try:
        for key, bucket in examples.items():
            task = tasks[bucket[0].task]
            for example in bucket:
                for regime in cfg.regimes:
                    results.append(
                        _run_one(
                            example,
                            task,
                            regime,
                            generation=generation,
                            fast_weights=fast_weights,
                            tokenizer=tokenizer,
                            stop_ids=stop_ids,
                            max_new_tokens=cfg.max_new_tokens,
                            prompt_style=cfg.prompt_style,
                        )
                    )
            announce(f"  {key}: {len(bucket)} examples x {len(cfg.regimes)} regimes")
    finally:
        # Not `reset_stream()` too: every `_run_one` already resets it before
        # running, so a lingering last-example state here is inert, and
        # wiping it here would erase what a caller might want to inspect.
        fast_weights.set_mode(evolve=True, stream=False, session=False)
        fast_weights.reset_carry()
        fast_weights.install(restore)
    return RulerReport(results=tuple(results), metrics=_aggregate(results))


def _run_one(
    example: RulerExample,
    task: RulerTask,
    regime: str,
    *,
    generation: Generation,
    fast_weights: FastWeights,
    tokenizer: Tokenizer,
    stop_ids,
    max_new_tokens: int,
    prompt_style: str,
) -> RulerResult:
    fast_weights.reset_stream()
    fast_weights.set_mode(evolve=regime == COLD_CARRY, stream=True, session=False)
    completion = generate.generate(
        generation=generation,
        prompt_ids=tokenizer.encode(_prompt_text(example, tokenizer, prompt_style)),
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        temperature=0.0,
    )
    prediction = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
    return RulerResult(
        task=example.task,
        length_bucket=example.length_bucket,
        regime=regime,
        score=task.score(prediction, example),
        n_new_tokens=len(completion.token_ids),
    )


def _prompt_text(example: RulerExample, tokenizer: Tokenizer, prompt_style: str) -> str:
    """`base`: the raw completion cue RULER was designed around — a base model
    has no instruction-following to lean on, so the prompt must end exactly
    where the answer starts. `instruct`: through the chat template instead,
    trusting the model to follow the question without a hand-written cue."""
    if prompt_style == BASE:
        return example.prompt + example.answer_prefix
    if prompt_style == INSTRUCT:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}], add_generation_prompt=True
        )
    raise ValueError(f"unknown ruler prompt_style {prompt_style!r}")


def _aggregate(results: Sequence[RulerResult]) -> dict[str, float]:
    by_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for result in results:
        by_key[(result.task, result.length_bucket, result.regime)].append(result.score)
    return {
        f"ruler/{task}_{length}_{regime}": sum(scores) / len(scores)
        for (task, length, regime), scores in by_key.items()
    }
