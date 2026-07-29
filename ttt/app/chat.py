"""Chat as a carry-ablation surface: three switches and a control (D14).

The model never sees a prior turn in context unless `context=full` says so, so
the fast weight is the only carrier of conversation memory — which is the
thing under test, not a limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ttt.app import generate as generate_mod
from ttt.core import sampling_text
from ttt.core.types import Carry
from ttt.ports.fast_weights import STREAM, FastWeights
from ttt.ports.generation import Generation
from ttt.ports.rng import Rng
from ttt.ports.tokenizer import Tokenizer

NONE = "none"
FULL = "full"
CONTEXTS = (NONE, FULL)


@dataclass(frozen=True)
class Switches:
    within_turn: bool = True
    cross_turn: bool = True
    seed: str = ""
    context: str = NONE

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(
                f"unknown context {self.context!r}; expected one of {list(CONTEXTS)}"
            )


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 512
    seed: int | None = None


@dataclass(frozen=True)
class Diagnostics:
    """`state_ratio` zero has two meanings; `pending_tokens` disambiguates."""

    state_ratio: float
    state_growth: float
    pending_tokens: int
    chunk_size: int
    gate_mean: float | None = None
    gate_std: float | None = None


@dataclass(frozen=True)
class Turn:
    text: str
    thinking: str
    raw: str
    token_ids: tuple[int, ...]
    stop_reason: str
    stop_token_id: int | None
    diagnostics: Diagnostics


@dataclass
class ChatSession:
    generation: Generation
    fast_weights: FastWeights
    tokenizer: Tokenizer
    rng: Rng
    decay: float
    switches: Switches = field(default_factory=Switches)
    sampling: Sampling = field(default_factory=Sampling)
    carries: Mapping[str, Carry] = field(default_factory=dict)
    system_prompt: str = ""
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        self._stop_ids = sampling_text.stop_token_ids(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            special_token_ids={
                token: self.tokenizer.token_id(token)
                for token in sampling_text.CHAT_SPECIAL_TOKENS
            },
        )
        self.reset()

    def reset(self) -> None:
        self.turns = 0
        self.history: list[dict[str, str]] = []
        self.generation.reset_cache()
        self.fast_weights.reset_stream()
        self._set_mode()
        seed = self.carries.get(self.switches.seed) if self.switches.seed else None
        if seed is not None:
            self.fast_weights.install(seed, family=STREAM)

    def turn(self, message: str) -> Turn:
        self._set_mode()
        before = self.fast_weights.state_ratio(family=STREAM)

        self.generation.reset_cache()
        prompt = self.tokenizer.apply_chat_template(
            self._messages(message),
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        completion = generate_mod.generate(
            generation=self.generation,
            prompt_ids=self.tokenizer.encode(prompt),
            max_new_tokens=self.sampling.max_new_tokens,
            stop_ids=self._stop_ids,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            generator=self._generator(),
        )

        raw = self.tokenizer.decode(completion.token_ids)
        thinking, answer = sampling_text.split_thinking(
            sampling_text.strip_chat_specials(raw)
        )
        diagnostics = self._diagnostics(before)
        self._close_turn(message, answer.strip())
        return Turn(
            text=answer.strip(),
            thinking=thinking,
            raw=raw,
            token_ids=completion.token_ids,
            stop_reason=completion.stop_reason,
            stop_token_id=completion.stop_token_id,
            diagnostics=diagnostics,
        )

    def _set_mode(self) -> None:
        self.fast_weights.set_mode(
            evolve=self.switches.within_turn, stream=True, session=False
        )

    def _messages(self, message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        if self.switches.context == FULL:
            messages.extend(self.history)
        messages.append({"role": "user", "content": message})
        return messages

    def _generator(self):
        if self.sampling.seed is None:
            return None
        return self.rng.torch_generator(self.sampling.seed + self.turns)

    def _diagnostics(self, before: float) -> Diagnostics:
        after = self.fast_weights.state_ratio(family=STREAM)
        pending, chunk = self.fast_weights.stream_progress()
        mean, std = self.fast_weights.gate_stats() or (None, None)
        return Diagnostics(
            state_ratio=after,
            state_growth=(after - before) / after if after else 0.0,
            pending_tokens=pending,
            chunk_size=chunk,
            gate_mean=mean,
            gate_std=std,
        )

    def _close_turn(self, message: str, answer: str) -> None:
        """Decay at the boundary, so chat drives the state through the regime
        training showed the model rather than an unbounded sum (D14 defect 3)."""
        if self.switches.cross_turn:
            self.fast_weights.reset_v_context()
            self.fast_weights.decay_stream(factor=self.decay)
        else:
            self.fast_weights.reset_stream()
        self.turns += 1
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})


def ab_turn(message: str, arms: Sequence[ChatSession]) -> tuple[Turn, ...]:
    """The same turn under two switch settings. Identical `sampling.seed` on
    both arms is what attributes a difference to the carry, not to sampling."""
    return tuple(arm.turn(message) for arm in arms)
