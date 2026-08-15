"""TrainConfig — outer-loop hyperparameters. Frozen (D3).

No session-mode booleans (D4: one `strategy` name), no strategy knobs (spec §7:
plugins own their config), no slice_*/session_papers_* (D11), no
param_log_every (D12).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_STRATEGY = "hybrid"


@dataclass(frozen=True)
class TrainConfig:
    max_seq_len: int = 16384
    min_doc_tokens: int = 2048

    # Fixed at 1: the carry is per-stream.
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    num_epochs: int = 1

    # "" falls back to the spec's default; "none" is the explicit opt-out.
    source_preset: str = ""

    lr_lora: float = 1e-5
    lr_wdown: float = 3e-5
    lr_new_modules: float = 2e-5

    weight_decay_full: float = 0.1
    weight_decay_lora: float = 0.0
    warmup_ratio: float = 0.02
    warmup_min_steps: int = 10
    max_grad_norm: float = 10.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    strategy: str = DEFAULT_STRATEGY

    # False is the A1 ablation (every item starts from S_0 = 0), not a mode.
    session_training: bool = True

    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    run_name: str = "ttt-v1.1"
    # torch.autograd.set_detect_anomaly: a real slowdown, opt-in only, for
    # pinpointing exactly which backward op first produces a non-finite
    # gradient — not something to leave on for a real run.
    debug_anomaly: bool = False
    wandb_enabled: bool = True
    wandb_project: str = "inplace-ttt"

    eval_every: int = 100
    eval_n_docs: int = 3
    # In-loop default; standalone eval commands (holdout_eval_v1/v2) raise
    # this to 50 unless a flag overrides it — a checkpoint eval runs often and
    # cheaply, a standalone one runs once and wants the tighter estimate.
    eval_n_docs_per_source: int = 5
    eval_n_slices: int = 8
    eval_min_tokens: int = 2048
    eval_holdout_seed: int = 0

    def __post_init__(self) -> None:
        if self.micro_batch_size != 1:
            raise ValueError(
                f"micro_batch_size is fixed at 1 (the carry is per-stream), "
                f"got {self.micro_batch_size}"
            )
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if self.num_epochs < 0:
            raise ValueError(f"num_epochs must be >= 0, got {self.num_epochs}")
        if self.min_doc_tokens < 1:
            raise ValueError(f"min_doc_tokens must be >= 1, got {self.min_doc_tokens}")
        if self.max_seq_len < self.min_doc_tokens:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) is below min_doc_tokens "
                f"({self.min_doc_tokens}); every document would be truncated "
                "to shorter than the length it was kept for"
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.eval_every < 0:
            raise ValueError(f"eval_every must be >= 0, got {self.eval_every}")
        if self.eval_n_slices < 1:
            raise ValueError(f"eval_n_slices must be >= 1, got {self.eval_n_slices}")
        if not self.strategy:
            raise ValueError("strategy must name a registered session strategy")
