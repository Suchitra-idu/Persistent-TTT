"""RULER's vocabulary: one synthetic example, one scored result.

Plain data, no torch — same spirit as `EvalRow`/`PplRow` in `core/types.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QaPair:
    """One real question over one real document, for the `qa` task family."""

    query: str
    answers: tuple[str, ...]
    context: str

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("a QaPair needs at least one answer")


@dataclass(frozen=True)
class RulerContext:
    """Real-world material a generator needs but does not own, assembled once
    in Ring 4 so `extensions/ruler_tasks/` stays core-only."""

    haystack_words: tuple[str, ...] = ()
    qa_pairs: tuple[QaPair, ...] = ()


@dataclass(frozen=True)
class RulerExample:
    """One synthetic prompt + the answer(s) that count as correct.

    `answer_prefix` is RULER's own completion cue (e.g. " Answer:") — a base
    model has no instruction-following to lean on, so the prompt itself has to
    end exactly where the answer should start (`RulerConfig.prompt_style`).
    """

    task: str
    length_bucket: int
    prompt: str
    answer_prefix: str
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("a RulerExample needs a task name")
        if self.length_bucket <= 0:
            raise ValueError(f"length_bucket must be > 0, got {self.length_bucket}")
        if not self.prompt:
            raise ValueError("a RulerExample needs a non-empty prompt")
        if not self.answer_prefix:
            raise ValueError("a RulerExample needs a non-empty answer_prefix")
        if not self.targets:
            raise ValueError("a RulerExample needs at least one target")


@dataclass(frozen=True)
class RulerResult:
    """One scored generation: one example, under one FastWeights regime."""

    task: str
    length_bucket: int
    regime: str
    score: float
    n_new_tokens: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.n_new_tokens < 0:
            raise ValueError(f"n_new_tokens must be >= 0, got {self.n_new_tokens}")
