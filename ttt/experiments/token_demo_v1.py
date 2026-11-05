"""Token view of the carry on pasted text: the paper's relatedness protocol, live.

    MODAL_PROFILE=vihanganoob123 modal deploy ttt/experiments/token_demo_v1.py
    python -m ttt.experiments.token_demo_web
"""

# No `from __future__ import annotations`: `modal.parameter` reads raw annotations.
import dataclasses

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.experiments import _runtime

app = modal.App("ttt-token-demo-v1")
image = modal_runtime.build_image()

GPU = ["L4", "A10G", "L40S", "A100", "H100"]
SCALEDOWN_SECONDS = 300
CHECKPOINT = "hebbian-carry-600/step_600"
FLAGS = (
    "update_rule=hebbian,state_aware_gate=false,limit_docs=4000,carry_gate=true,"
    "carry_backprop=true,output_gate_bias_init=-6,run_name=hebbian-carry-600"
)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    timeout=30 * 60,
)
class TokenDemo:
    resume_from: str = modal.parameter(default=CHECKPOINT)
    flags: str = modal.parameter(default=FLAGS)

    @modal.enter()
    def load(self):
        from ttt.experiments.holdout_eval_v1 import _EvalCompute

        resolved = cli.from_flags(
            **{**cli.env_defaults(), **cli.parse_flags(self.flags), "resume_from": self.resume_from}
        )
        self.engine = _runtime.build(
            resolved, storage=modal_runtime.checkpoint_storage(), root=modal_runtime.CKPT_MOUNT,
            trainable=False,
        )
        self.engine.model.eval()
        self.compute = _EvalCompute(self.engine.model)

    @modal.method()
    def score(self, prime: str, target: str) -> dict:
        """Both texts are cut to their first 1024 tokens, as in the paper."""
        from ttt.app import token_view

        tokenizer = self.engine.tokenizer
        prime_ids = tokenizer.encode(prime)[: token_view.BLOCK]
        target_ids = tokenizer.encode(target)[: token_view.BLOCK]
        view = token_view.measure(
            prime=prime_ids, target=target_ids, compute=self.compute,
            fast_weights=self.engine.fast_weights,
        )
        return {
            **dataclasses.asdict(view),
            "tokens": [tokenizer.decode([t]) for t in target_ids],
            "prime_tokens": len(prime_ids),
            "summary": view.summary(),
            "checkpoint": self.resume_from,
        }
