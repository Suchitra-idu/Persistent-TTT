"""Carry-on against carry-off continuation of the same held-out prefix.

Both arms draw from the same seeded generator, so the only difference between
them is whether the fast weight evolved while the prompt streamed.

    modal run ttt/experiments/holdout_generate_v1.py --resume-from step_600
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, generate
from ttt.core import sampling_text
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-holdout-generate-v1")
image = modal_runtime.build_image()

PREFIX_TOKENS = 320
MAX_NEW_TOKENS = 120


@dataclass(frozen=True)
class Continuation:
    doc_idx: int
    source: str
    prompt: str
    carry_on: str
    carry_off: str
    state_ratio: float


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    n_docs: int = 1,
    prefix_tokens: int = PREFIX_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    announce: Callable[[str], None] = print,
) -> tuple[Continuation, ...]:
    """Greedy by default: with temperature 0 the two arms cannot differ by luck."""
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    stop_ids = sampling_text.stop_token_ids(
        eos_token_id=engine.tokenizer.eos_token_id,
        pad_token_id=engine.tokenizer.pad_token_id,
        unk_token_id=engine.tokenizer.unk_token_id,
    )

    out = []
    for doc in holdout.docs[:n_docs]:
        prompt_ids = doc.token_ids[:prefix_tokens]
        arms = {
            evolve: _continue(
                engine,
                prompt_ids,
                evolve=evolve,
                stop_ids=stop_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=resolved.train.seed,
            )
            for evolve in (True, False)
        }
        out.append(
            Continuation(
                doc_idx=doc.index,
                source=doc.source,
                prompt=engine.tokenizer.decode(prompt_ids, skip_special_tokens=True),
                carry_on=arms[True][0],
                carry_off=arms[False][0],
                state_ratio=arms[True][1],
            )
        )
        _announce(out[-1], announce)
    return tuple(out)


def _continue(engine, prompt_ids, *, evolve, stop_ids, max_new_tokens, temperature, seed):
    engine.fast_weights.reset_stream()
    engine.fast_weights.set_mode(evolve=evolve, stream=True, session=False)
    engine.generation.reset_cache()
    completion = generate.generate(
        generation=engine.generation,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        temperature=temperature,
        generator=engine.rng().torch_generator(seed),
    )
    text = engine.tokenizer.decode(completion.token_ids, skip_special_tokens=True)
    return text, engine.fast_weights.state_ratio(family="stream")


def _announce(continuation: Continuation, announce) -> None:
    announce("=" * 78)
    announce(f"document {continuation.doc_idx} [{continuation.source}]")
    announce(f"PROMPT (tail): {continuation.prompt[-300:]}")
    announce(f"CARRY ON  (state/W0 {continuation.state_ratio:.2e}):")
    announce(continuation.carry_on)
    announce("CARRY OFF:")
    announce(continuation.carry_off)


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
def holdout_generate(n_docs: int = 1, temperature: float = 0.0, **flags):
    from ttt.adapters.hf_data_source import HfDataSource

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    run(
        resolved,
        engine=engine,
        source=HfDataSource(),
        n_docs=n_docs,
        temperature=temperature,
    )


@app.local_entrypoint()
def main(n_docs: int = 1, temperature: float = 0.0, **flags):
    holdout_generate.remote(n_docs=n_docs, temperature=temperature, **flags)
