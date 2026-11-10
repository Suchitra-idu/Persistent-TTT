"""TTTConfig — the mechanism's hyperparameters. Frozen (D3).

Changing chunk_size, eta, conv_kernel_size or normalize_delta_by_chunk
invalidates a checkpoint.
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
    eta: float = 2.0
    normalize_delta_by_chunk: bool = True
    conv_kernel_size: int = 8
    update_rule: str = "hebbian"

    # Initialises both the main gate and the carry gate. The gated carry of the
    # paper uses -6 with carry_gate and carry_backprop; these defaults are the
    # ungated carry, which the recorded ungated runs relied on.
    output_gate_bias_init: float = -2.0

    clip_enabled: bool = True
    clip_tau: float = 20.0
    carried_decay: float = 0.9
    truncate_every: int = 5

    carry_gate: bool = False
    carry_backprop: bool = False

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
        if self.truncate_every < 1:
            raise ValueError(
                f"truncate_every must be >= 1, got {self.truncate_every}"
            )
        if self.update_rule not in ("hebbian", "delta_chunk"):
            raise ValueError(
                f"update_rule must be 'hebbian' or 'delta_chunk', got {self.update_rule!r}"
            )
        if self.carry_gate and self.update_rule != "hebbian":
            raise ValueError(
                "carry_gate is implemented for update_rule='hebbian' only, got "
                f"{self.update_rule!r}; the delta rules read the carry inside "
                "their residual, which a separate path would bypass"
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
        """The tau for ttt_math.scan, or None."""
        return self.clip_tau if self.clip_enabled else None
