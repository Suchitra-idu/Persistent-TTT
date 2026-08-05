"""The deployed chat engine the REPL talks to. One container, one live carry.

    modal deploy ttt/experiments/chat_v1.py
    python -m ttt.experiments.chat_repl --resume-from step_600

The container is what holds the fast weight between turns, so the session lives
here and `chat_repl` drives it remotely.
"""

# No `from __future__ import annotations` here: `modal.parameter` reads
# `__annotations__` without resolving it, and PEP 563 hands it the string "str".
import dataclasses

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app.chat import ChatSession, Sampling, Switches
from ttt.experiments import _runtime

app = modal.App("ttt-chat-v1")
image = modal_runtime.build_image()

SCALEDOWN_SECONDS = 300


def build_session(resolved: cli.Resolved, engine, *, switches, sampling, system_prompt):
    """The carry a run trained is what `switches.seed` names, so the carriers
    are loaded here and the switch picks one by source."""
    carries, _ = engine.carries()
    return ChatSession(
        generation=engine.generation,
        fast_weights=engine.fast_weights,
        tokenizer=engine.tokenizer,
        rng=engine.rng(),
        decay=resolved.ttt.carried_decay,
        switches=switches,
        sampling=sampling,
        carries=carries,
        system_prompt=system_prompt,
    )


def as_payload(turn) -> dict:
    return {
        **dataclasses.asdict(turn),
        "diagnostics": dataclasses.asdict(turn.diagnostics),
    }


@app.cls(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=60 * 60,
)
class ChatEngine:
    resume_from: str = modal.parameter(default="")
    flags: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        self.resolved = cli.from_flags(
            **{
                **cli.env_defaults(),
                **cli.parse_flags(self.flags),
                "resume_from": self.resume_from,
            }
        )
        self.engine = _runtime.build(
            self.resolved,
            storage=modal_runtime.checkpoint_storage(),
            root=modal_runtime.CKPT_MOUNT,
            trainable=False,
        )
        self.session = None

    @modal.method()
    def configure(self, switches: dict, sampling: dict, system_prompt: str = "") -> dict:
        """Also the reset: a seed switch only means anything from a clean state."""
        self.session = build_session(
            self.resolved,
            self.engine,
            switches=Switches(**switches),
            sampling=Sampling(**sampling),
            system_prompt=system_prompt,
        )
        return {"ready": True, **switches}

    @modal.method()
    def turn(self, message: str) -> dict:
        if self.session is None:
            raise RuntimeError("configure before turn; there is no session yet")
        return as_payload(self.session.turn(message))
