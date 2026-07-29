"""The wiring check: with W_target zeroed, the TTT path must contribute exactly zero.

If this fails, every number the other experiments produce is meaningless — a
nonzero difference at W_target=0 means the TTT branch is altering the output
through some path that is not the fast weight.

    modal run ttt/experiments/sanity_check_v1.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.experiments import _runtime
from ttt.extensions.mechanism import iter_ttt_modules

app = modal.App("ttt-sanity-check-v1")
image = modal_runtime.build_image()

TOLERANCE = 1e-3


@dataclass(frozen=True)
class SanityResult:
    diff_at_init: float
    diff_at_zero: float
    n_tokens: int

    @property
    def passed(self) -> bool:
        return self.diff_at_zero < TOLERANCE


def input_ids(token_ids: Sequence[int], model):
    """On the model's device: a CPU batch against a CUDA model fails inside the
    embedding lookup, in a traceback that names neither."""
    import torch

    return torch.tensor(
        [list(token_ids)], device=next(model.parameters()).device, dtype=torch.long
    )


def run(
    model,
    token_ids: Sequence[int],
    *,
    fast_weights,
    announce: Callable[[str], None] = print,
) -> SanityResult:
    """Needs more tokens than chunk_size, or the scan path never fires and a
    broken scan would pass by never running."""
    import torch

    ids = input_ids(token_ids, model)
    # Walked, not indexed: PEFT wraps the model and `model.model.layers` stops
    # resolving the moment LoRA is attached.
    modules = list(iter_ttt_modules(model))

    fast_weights.set_mode(evolve=False, stream=False, session=False)
    with torch.no_grad():
        off = model(input_ids=ids).logits

    fast_weights.set_mode(evolve=True, stream=False, session=False)
    with torch.no_grad():
        at_init = model(input_ids=ids).logits

    with torch.no_grad():
        for module in modules:
            module.w_target.zero_()
        at_zero = model(input_ids=ids).logits

    result = SanityResult(
        diff_at_init=float((at_init - off).abs().max()),
        diff_at_zero=float((at_zero - off).abs().max()),
        n_tokens=len(token_ids),
    )
    announce(
        f"max |logit diff| at trained W_target = {result.diff_at_init:.4f}, "
        f"at W_target=0 = {result.diff_at_zero:.6f}"
    )
    announce("identity check passed" if result.passed else "IDENTITY CHECK FAILED")
    return result


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume()},
    secrets=modal_runtime.secrets(),
    timeout=60 * 20,
)
def sanity_check(**flags):
    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    text = "Test-time training updates a subset of weights during inference. " * 20
    result = run(
        engine.model,
        engine.tokenizer.encode(text),
        fast_weights=engine.fast_weights,
    )
    if not result.passed:
        raise AssertionError(
            f"TTT path is not exact-zero at W_target=0 "
            f"(max diff {result.diff_at_zero:.6f}); the wiring is broken"
        )
    return result.diff_at_zero


@app.local_entrypoint()
def main(**flags):
    sanity_check.remote(**flags)
