"""Warmup + cosine LR math, and the step count that sizes it.

Here rather than in the loop because Ring 4 may not import transformers.
"""

from __future__ import annotations

import math


def total_optimizer_steps(items: int, grad_accum_steps: int, epochs: int) -> int:
    """ceil(items / accum) per epoch: a partial final window still steps."""
    if items < 0:
        raise ValueError(f"items must be >= 0, got {items}")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if epochs < 0:
        raise ValueError(f"epochs must be >= 0, got {epochs}")
    return math.ceil(items / grad_accum_steps) * epochs


def warmup_steps(total_steps: int, warmup_ratio: float, warmup_min_steps: int) -> int:
    """The floor keeps a short run from reaching full LR on step 2."""
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
    if warmup_min_steps < 0:
        raise ValueError(f"warmup_min_steps must be >= 0, got {warmup_min_steps}")
    return max(warmup_min_steps, int(warmup_ratio * total_steps))


def multiplier(step: int, *, num_warmup_steps: int, num_training_steps: int) -> float:
    """LR multiplier in [0, 1]: linear warmup, then a half cosine to zero.

    Matches transformers' cosine schedule (num_cycles=0.5) across
    [0, num_training_steps]. Past the end it raises; HuggingFace keeps
    evaluating the cosine there and hands back a rising LR.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step > num_training_steps:
        raise ValueError(
            f"step {step} is past the end of the schedule "
            f"({num_training_steps} steps); the loop has stepped more times "
            "than it sized the run for"
        )
    if num_warmup_steps > 0 and step < num_warmup_steps:
        return step / num_warmup_steps
    remaining = num_training_steps - num_warmup_steps
    if remaining <= 0:
        return 0.0
    progress = (step - num_warmup_steps) / remaining
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
