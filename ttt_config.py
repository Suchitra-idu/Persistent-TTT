"""Shared configuration for In-Place TTT continual pretraining on Modal.

Single source of truth for paths, model identity, and hyperparameters,
imported by both the training and inference apps.
"""

import os
from dataclasses import dataclass, field


CKPT_VOLUME_NAME = "ttt-checkpoints"
HF_CACHE_VOLUME_NAME = "hf-hub-cache"

CKPT_MOUNT = "/ckpt"
HF_CACHE_MOUNT = "/hf-cache"

# If the repo is PRIVATE: `modal secret create huggingface HF_TOKEN=hf_...`
# then uncomment the secrets=[...] lines in train_modal.py / infer_modal.py.
DATASET_SOURCE = "suchitraIdu/arxiv-ml-16k"

TEXT_COLUMN = "text"
TOKENS_EST_COLUMN = "tokens_est"

# Last n rows (newest arxiv ids) reserved as a contamination-free eval pool.
HOLDOUT_LAST_N = 200

# TTT_BASE_MODEL fully overrides if you need a non-Qwen3 path.
MODEL_SIZE = os.environ.get("TTT_MODEL_SIZE", "0.6B")
BASE_MODEL = os.environ.get("TTT_BASE_MODEL", f"Qwen/Qwen3-{MODEL_SIZE}")

LAYER_STRIDE = int(os.environ.get("TTT_LAYER_STRIDE", "2"))
LAYER_START = int(os.environ.get("TTT_LAYER_START", "1"))


def derive_ttt_layer_indices(num_layers: int,
                             stride: int = LAYER_STRIDE,
                             start: int = LAYER_START) -> tuple:
    """Every `stride`-th layer starting at `start`, capped at num_layers-1."""
    return tuple(range(start, num_layers, stride))


# Domain terms force-unmasked by the content-token loss mask. Expanded into
# multiple BPE variants at protect-time; only single-piece variants protect.
LOSS_MASK_DEFAULT_PROTECT_TERMS = (
    "transformer", "attention", "convolution", "convolutional",
    "embedding", "encoder", "decoder", "autoencoder",
    "perceptron", "mlp", "lstm", "rnn", "cnn", "gnn",
    "gan", "vae", "diffusion", "residual", "recurrent",
    "feedforward", "capsule",
    "softmax", "sigmoid", "relu", "gelu", "tanh",
    "activation", "normalization", "regularization", "dropout",
    "layernorm", "batchnorm", "groupnorm", "rmsnorm",
    "bert", "gpt", "llama", "qwen", "mistral", "claude",
    "gemini", "palm", "llm", "vit", "clip", "deepseek",
    "phi", "gemma", "falcon", "mixtral", "dalle",
    "policy", "reward", "agent", "action", "trajectory",
    "ppo", "dqn", "sac", "actor", "critic", "episode",
    "bandit", "exploration", "exploitation", "rollout",
    "bayesian", "markov", "gaussian", "kernel", "manifold",
    "lipschitz", "convex", "lagrangian", "hessian", "jacobian",
    "eigenvalue", "eigenvector", "tensor", "scalar",
    "entropy", "divergence", "kullback", "leibler",
    "wasserstein", "frobenius", "posterior", "prior",
    "likelihood", "mcmc", "mle",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega",
    "optimizer", "logits", "perplexity", "checkpoint",
    "pretraining", "finetuning", "adamw", "adam", "rmsprop",
    "sgd", "momentum", "lora", "qlora", "rlhf", "dpo",
    "classification", "regression", "segmentation",
    "detection", "translation", "summarization",
    "recognition", "captioning", "parsing",
    "imagenet", "mnist", "cifar", "coco", "glue",
    "squad", "bleu", "rouge", "wmt", "mmlu",
    "gsm8k", "humaneval", "bigbench",
)


@dataclass
class TTTConfig:
    """Hyperparameters of the In-Place TTT mechanism itself."""

    # Populated lazily from model.config.num_hidden_layers in
    # model_setup.build_model. Tests may pass an explicit tuple.
    layer_indices: tuple | None = None

    # Tokens per fast-weight update. Changing requires retraining.
    chunk_size: int = 512

    # Inner-loop learning rate for W <- W + eta * V^T Z.
    eta: float = 5e-2

    # Divide each chunk delta by chunk_size so eta is roughly C-independent.
    normalize_delta_by_chunk: bool = True

    # Causal Conv1D kernel width for V = Conv1D(source) @ W_target.
    conv_kernel_size: int = 8

    # Source tensor feeding target_conv: "embedding" (paper) or "hidden_state"
    # (per-layer input, more expressive; requires per-layer stream buffers).
    v_source: str = "hidden_state"

    # Symmetric pad (past+current+future) for target_conv instead of causal.
    # WARNING: breaks chunk-causality under standard NTP; carry can leak
    # ground-truth right-context. Streaming inference ignores this flag.
    v_bidirectional: bool = False

    # Per-position gate: output = base + sigmoid(W_g h) * eta * ttt_out.
    # Opens an extra gradient path for W_target vs raw linear attention.
    output_gate: bool = True

    # sigmoid(-2) ~= 0.12: gate starts mostly closed, must learn to open.
    output_gate_bias_init: float = -2.0

    # L2 on gate output (gate_reg_weight * mean(sigmoid(W_g h)^2)). 0 disables.
    gate_reg_weight: float = 0.0

    # Frobenius clip on ||eta * cumulative_state||_F, per-chunk in scan and
    # per-commit in stream. Bounds carry/W0 ratio; retune for other model
    # sizes (||W0||_F scales ~ sqrt(d * d_ff)).
    clip_enabled: bool = True
    clip_tau: float = 5.0
    clip_at_inference_only: bool = False

    def __post_init__(self):
        if self.v_source not in ("embedding", "hidden_state"):
            raise ValueError(
                "v_source must be 'embedding' or 'hidden_state', "
                f"got {self.v_source!r}"
            )


@dataclass
class TrainConfig:
    """Outer-loop (continual pretraining) hyperparameters."""

    max_seq_len: int = 16384
    min_doc_tokens: int = 2048
    micro_batch_size: int = 1      # fixed at 1; carry is per-stream
    grad_accum_steps: int = 16
    num_epochs: int = 1

    # Three parameter groups, three learning rates.
    lr_lora: float = 2e-5
    lr_wdown: float = 3e-5          # pretrained fast weight init, move gently
    lr_new_modules: float = 2e-5

    weight_decay_full: float = 0.1
    weight_decay_lora: float = 0.0
    warmup_ratio: float = 0.02
    warmup_min_steps: int = 10
    max_grad_norm: float = 10.0

    # LoRA. Alpha at 2x rank per current practice.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Session-persistent training: fast weight deltas carry across papers
    # within a session (detached, TBPTT-style) and reset at boundaries.
    session_training: bool = False
    session_papers_min: int = 2
    session_papers_max: int = 6

    # Random within-paper token-range slicing inside a session.
    # slice_prob=0 disables; slice_min_tokens guards against sub-chunk slices.
    slice_prob: float = 0
    slice_min: int = 2
    slice_max: int = 6
    slice_min_tokens: int = 1024

    # Single-paper sessions: one paper sliced into k random pieces.
    # When True, session_papers_* and slice_* above are IGNORED.
    single_paper_sessions: bool = False
    single_paper_slices_min: int = 2
    single_paper_slices_max: int = 6

    # Content-token loss masking. CE is computed only on positions whose
    # token_id is NOT among the most-frequent tokens accounting for
    # (1 - loss_mask_keep_fraction) of baseline occurrences.
    # 1.0 disables masking even when loss_mask_enabled=True.
    loss_mask_enabled: bool = False
    loss_mask_keep_fraction: float = 0.5

    # Force-unmask domain content. Pass () to disable the override.
    loss_mask_protect_terms: tuple = field(
        default_factory=lambda: LOSS_MASK_DEFAULT_PROTECT_TERMS
    )

    # External-reference frequency baseline; falls back to in-corpus
    # frequency (with a log line) when the file is absent.
    loss_mask_reference_counts_path: str = (
        "/ckpt/loss_mask/reference_wikitext103.pt"
    )

    # Predicate-protect: pure-digit tokens (scientific content).
    loss_mask_protect_numeric: bool = True

    # Predicate-protect: single-character math symbols (scientific content).
    loss_mask_protect_symbols: tuple = (
        "=", "@", "^", "_", "\\", "+", "-", "*", "/", "|", "<", ">",
    )

    # Mask the first N tokens of paper-START items (SessionItem.start == 0)
    # to skip boilerplate. Mid-paper slices are unaffected. 0 disables.
    loss_mask_first_tokens: int = 16

    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    run_name: str = "ttt-v1.1"

    wandb_enabled: bool = True
    wandb_project: str = "inplace-ttt"
    param_log_every: int = 50

    # In-loop holdout eval: per-paper ppl_fresh - ppl_carry gap on
    # eval_n_papers papers x eval_n_slices slices, twice (with/without
    # carry). eval_every=0 disables.
    eval_every: int = 100
    eval_n_papers: int = 3
    eval_n_slices: int = 8
    eval_holdout_seed: int = 0


TTT_CFG = TTTConfig()
TRAIN_CFG = TrainConfig()
