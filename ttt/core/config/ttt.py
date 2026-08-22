"""TTTConfig — the mechanism's hyperparameters. Frozen (D3).

Changing chunk_size, eta, conv_kernel_size or normalize_delta_by_chunk
invalidates a checkpoint. v_source and gate_reg_weight are cut (D11).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LAYER_STRIDE = 2
DEFAULT_LAYER_START = 1


def derive_layer_indices(
    num_layers: int,
    stride: int = DEFAULT_LAYER_STRIDE,
    start: int = DEFAULT_LAYER_START,
) -> tuple[int, ...]:
    """Derived from model depth, so one config works across model sizes."""
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    return tuple(range(start, num_layers, stride))


@dataclass(frozen=True)
class TTTConfig:
    # None until model depth is known; derive_layer_indices fills it in.
    layer_indices: tuple[int, ...] | None = None

    chunk_size: int = 50
    eta: float = 2
    normalize_delta_by_chunk: bool = True
    conv_kernel_size: int = 8

    # "hebbian" is ttt_math.scan; "delta"/"delta_chunk" are error-corrected
    # alternatives, O(N)- and O(N/chunk_size)-sequential respectively. See
    # delta_scan's and chunked_delta_scan's docstrings.
    update_rule: str = "hebbian"

    # Breaks chunk-causality under next-token prediction; a knowingly invalid
    # ablation kept pending a research call (PLAN §9.2).
    v_bidirectional: bool = False

    output_gate: bool = True
    # -0.5 and +2.0 (less damped) were both tried and both hurt early
    # training — an undertrained carry let through at higher weight is
    # noise, not signal, this early. -2.0 is the value that's actually held up.
    output_gate_bias_init: float = -2.0

    clip_enabled: bool = True
    clip_tau: float = 20.0
    clip_at_inference_only: bool = False

    # 1.0 = pure sum (unbounded), 0.0 = last item only. Was tried at 0.98 for
    # RULER's long-range retention; reverted — over `everlasting`'s never-reset
    # accumulation that's slow enough forgetting to let the carry diverge, not
    # just retain a needle longer. Revisit with a cap on the stored carry
    # itself, not just clip_tau's cap on its use, before trying this again.
    carried_decay: float = 0.9

    # carried_decay's recurrence, one chunk at a time instead of one item at
    # a time (see ttt_math.exclusive_decayed_cumsum). Defaults to 1.0, the
    # prior unbounded-sum behaviour, so this is opt-in.
    chunk_decay: float = 1.0

    # delta/delta_chunk's truncated-BPTT window (see delta_scan's docstring);
    # 1 starves the target projection of gradient entirely.
    truncate_every: int = 5

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.conv_kernel_size < 1:
            raise ValueError(
                f"conv_kernel_size must be >= 1, got {self.conv_kernel_size}"
            )
        if self.clip_tau <= 0.0:
            raise ValueError(f"clip_tau must be > 0, got {self.clip_tau}")
        if not 0.0 <= self.carried_decay <= 1.0:
            raise ValueError(
                f"carried_decay must be in [0, 1], got {self.carried_decay}"
            )
        if not 0.0 <= self.chunk_decay <= 1.0:
            raise ValueError(f"chunk_decay must be in [0, 1], got {self.chunk_decay}")
        if self.truncate_every < 1:
            raise ValueError(
                f"truncate_every must be >= 1, got {self.truncate_every}"
            )
        if self.update_rule not in ("hebbian", "delta", "delta_chunk"):
            raise ValueError(
                "update_rule must be 'hebbian', 'delta', or 'delta_chunk', got "
                f"{self.update_rule!r}"
            )
        if self.layer_indices is not None:
            indices = tuple(self.layer_indices)
            if len(set(indices)) != len(indices):
                raise ValueError(f"layer_indices has duplicates: {indices}")
            if any(i < 0 for i in indices):
                raise ValueError(f"layer_indices must be non-negative: {indices}")
            object.__setattr__(self, "layer_indices", indices)

    @property
    def effective_clip_tau(self) -> float | None:
        """The tau for ttt_math.scan, or None. clip_at_inference_only is a
        training-mode question only the mechanism can answer."""
        return self.clip_tau if self.clip_enabled else None
