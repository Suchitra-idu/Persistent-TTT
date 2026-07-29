"""Shared configuration for In-Place TTT continual pretraining on Modal.

Single source of truth for paths, model identity, and hyperparameters,
imported by both the training and inference apps.
"""

import os
from dataclasses import dataclass


CKPT_VOLUME_NAME = "ttt-checkpoints"
HF_CACHE_VOLUME_NAME = "hf-hub-cache"

CKPT_MOUNT = "/ckpt"
HF_CACHE_MOUNT = "/hf-cache"


# --------------------------------------------------------------------------
# Dataset selection
# --------------------------------------------------------------------------
# Datasets are described by a DatasetSpec + a small registry. The active spec
# is chosen by the TTT_DATASET env var (default "arxiv"), so switching
# datasets is one env var, not a code change. Every downstream module reads
# DATASET_SPEC rather than the individual columns.
#
# If a repo is PRIVATE, `modal secret create huggingface HF_TOKEN=hf_...`
# then uncomment the secrets=[...] lines in train_modal.py / infer_modal.py.


@dataclass(frozen=True)
class DatasetSpec:
    """How to pull rows out of one dataset.

    - `source` is the HF repo id or a local dir of parquet/arrow shards.
    - `text_column` is the doc-text column name.
    - `tokens_est_column`, when set and present, lets the loader do a cheap
      pre-filter on token counts before tokenization.
    - `source_meta_column` + `source_meta_key`, when both set, extract a
      per-row source label into a top-level `source` column (see
      data_utils._annotate_source). Datasets like SlimPajama carry the
      RedPajama subset name inside a `meta` struct.
    - `include_sources`, when set, keeps only rows whose extracted source
      label is in this tuple (see data_utils.apply_source_filter). This is
      how we exclude CommonCrawl on SlimPajama.
    - `default_source_preset`, when set, is the SOURCE_PRESETS entry
      applied by default when `cfg.source_preset` is empty. Prevents the
      raw dataset ratio (heavily C4-skewed on DKYoon SlimPajama-6B) from
      being used silently. Pass `--source-preset none` to bypass.
    - `holdout_last_n` is the newest-row count reserved for eval only.
    """

    name: str
    source: str
    text_column: str = "text"
    tokens_est_column: str | None = None
    source_meta_column: str | None = None
    source_meta_key: str | None = None
    include_sources: tuple | None = None
    default_source_preset: str | None = None
    holdout_last_n: int = 200


DATASETS: dict = {
    "arxiv": DatasetSpec(
        name="arxiv",
        source="suchitraIdu/arxiv-ml-16k",
        text_column="text",
        tokens_est_column="tokens_est",
        holdout_last_n=200,
    ),
    # DKYoon/SlimPajama-6B is a 6B-token sample of SlimPajama-627B. The
    # `meta` column is a struct with the RedPajama subset name; excluding
    # CommonCrawl drops ~54% of rows but leaves ~2.5M docs with a diverse
    # C4/Github/Books/ArXiv/Wikipedia/StackExchange mix -- enough to
    # characterize per-domain TTT gaps at 0.6B scale.
    "slimpajama-6b": DatasetSpec(
        name="slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        text_column="text",
        source_meta_column="meta",
        source_meta_key="redpajama_set_name",
        include_sources=(
            "RedPajamaC4",
            "RedPajamaGithub",
            "RedPajamaBook",
            "RedPajamaArXiv",
            "RedPajamaWikipedia",
            "RedPajamaStackExchange",
        ),
        # Bigger dataset AND rare sources are severely under-represented
        # in the DKYoon subsample (Books ~0.1%, ArXiv ~0.9%) -- so
        # holdout needs to be wide enough that per-source eval sampling
        # still finds a few Books/ArXiv rows.
        holdout_last_n=5000,
        # Enforce a balanced training mix by default. Without this, --limit-docs
        # would just take the raw C4-heavy head (~77% C4) and the per-source
        # gap signal would be dominated by web text. Override with
        # --source-preset slim-paper or pass --source-preset none to disable.
        default_source_preset="slim-research",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(
            f"Unknown TTT_DATASET {name!r}. Known: {sorted(DATASETS)}"
        )
    return DATASETS[name]


DATASET_NAME = os.environ.get("TTT_DATASET", "arxiv")
DATASET_SPEC = get_dataset_spec(DATASET_NAME)


# --------------------------------------------------------------------------
# Training-time source-balancing presets
# --------------------------------------------------------------------------
# The DKYoon/SlimPajama-6B subsample is heavily skewed vs the parent
# SlimPajama-627B advertised proportions -- Books/ArXiv/Github are
# structurally rare, C4/StackExchange overrepresented. When a preset is
# selected via `--source-preset <name>`, load_token_dataset takes rows
# per source in the given ratio, so the training mix is what you
# actually want to study rather than what the subsample happens to
# contain.
#
# Values are unnormalized weights (they sum to 100 here for readability
# but the balancer normalizes at apply-time, so any positive numbers work).
# Sources not listed in a preset are dropped from the training pool for
# that preset.
SOURCE_PRESETS: dict = {
    # Restores SlimPajama-627B design proportions from the paper. Use
    # this for papers where you want to say "we trained on the
    # SlimPajama mix" without asterisks.
    "slim-paper": {
        "RedPajamaC4": 62,
        "RedPajamaGithub": 9,
        "RedPajamaBook": 8,
        "RedPajamaArXiv": 7,
        "RedPajamaWikipedia": 7,
        "RedPajamaStackExchange": 6,
    },
    # Tuned for the "which domain benefits most from TTT" story: C4
    # downweighted so rare/structured domains (Github, ArXiv, Books)
    # get enough per-domain training signal to produce a legible gap.
    "slim-research": {
        "RedPajamaC4": 25,
        "RedPajamaGithub": 20,
        "RedPajamaBook": 15,
        "RedPajamaArXiv": 20,
        "RedPajamaWikipedia": 10,
        "RedPajamaStackExchange": 10,
    },
}


def get_source_preset(name: str) -> dict:
    if name not in SOURCE_PRESETS:
        raise KeyError(
            f"Unknown source preset {name!r}. "
            f"Known: {sorted(SOURCE_PRESETS)}"
        )
    return SOURCE_PRESETS[name]


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------
# TTT_BASE_MODEL fully overrides if you need a non-Qwen3 path.
# 0.6B is the proven size; see docs/scaling.md for 1.7B / 4B / 8B recipes
# (bigger models need LR bumps, eta / chunk_size retuning, and LAYER_STRIDE=4
# above 4B to keep the scan tensors inside 80 GB).
MODEL_SIZE = os.environ.get("TTT_MODEL_SIZE", "0.6B")
BASE_MODEL = os.environ.get("TTT_BASE_MODEL", f"Qwen/Qwen3-{MODEL_SIZE}")

# LAYER_STRIDE=2 gives half the transformer layers a TTT MLP (proven at 0.6B
# and 1.7B). Bump to 4 at 8B for a ~2x memory reduction on the scan tensors.
LAYER_STRIDE = int(os.environ.get("TTT_LAYER_STRIDE", "2"))
LAYER_START = int(os.environ.get("TTT_LAYER_START", "1"))


def derive_ttt_layer_indices(num_layers: int,
                             stride: int = LAYER_STRIDE,
                             start: int = LAYER_START) -> tuple:
    """Every `stride`-th layer starting at `start`, capped at num_layers-1."""
    return tuple(range(start, num_layers, stride))


@dataclass
class TTTConfig:
    """Hyperparameters of the In-Place TTT mechanism itself."""

    # Populated lazily from model.config.num_hidden_layers in
    # model_setup.build_model. Tests may pass an explicit tuple.
    layer_indices: tuple | None = None

    # Tokens per fast-weight update. Changing requires retraining.
    chunk_size: int = 50

    # Inner-loop learning rate for W <- W + eta * V^T Z.
    eta: float = 7e-2

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

    # EMA decay on the session-persistent carry:
    #   carried_delta <- carried_decay * carried_delta + this_item_delta.
    # 1.0 = pure sum (unbounded magnitude growth); 0.0 = last-item only (no
    # session memory). 0.9-0.95 keeps state_ratio bounded across long sessions
    # while still averaging across items. Session-training only.
    carried_decay: float = 0.9

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
    lr_lora: float = 1e-5
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
    single_paper_slices_min: int = 5
    single_paper_slices_max: int = 10

    # Rebalance training rows by source at load time. When set, looks
    # up the preset in SOURCE_PRESETS and takes rows per source in that
    # ratio (see load_token_dataset). Empty string = keep the natural
    # dataset mix. Intended for wildly-skewed pretraining subsamples
    # like DKYoon/SlimPajama-6B where C4 is 77% and Books is 0.1%.
    source_preset: str = ""

    # Hybrid session mode (per-doc split by length):
    #   L < hybrid_carry_min_tokens  -> single-item session, no slicing.
    #     Fast weight is still reset at session start; the model sees the
    #     "S_0 = 0" case for these, which is the correct training signal
    #     for short docs that shouldn't accumulate.
    #   L >= hybrid_carry_min_tokens -> one-paper session sliced into
    #     k in [slices_min, slices_max] pieces of >= slice_min_tokens each,
    #     carry propagates across the k slices via TBPTT.
    # When True, single_paper_sessions and session_papers_* / slice_*
    # above are IGNORED. Intended for diverse pretraining mixes (SlimPajama)
    # where document length varies wildly; unnecessary for arxiv-style
    # corpora where every doc is long.
    hybrid_sessions: bool = False
    hybrid_carry_min_tokens: int = 2100
    hybrid_slices_min: int = 2
    hybrid_slices_max: int = 6
    hybrid_slice_min_tokens: int = 1000

    # Everlasting (per-source) carry: instead of per-session TBPTT, maintain
    # 6 persistent per-source fast-weight carriers that get loaded before a
    # doc's forward and snapshotted back after. Tests whether accumulated
    # per-domain fast weights give a domain-specific adaptation benefit that
    # per-session carry (Δbetween ≈ 0 in the standard mode) does not.
    # When True:
    #   - Every doc is a whole-doc, single-item session (no slicing).
    #   - session_training is forced True; carry lives in a dict keyed
    #     by source label, not in a single session-scoped slot.
    #   - Requires the active dataset to have a `source` column.
    #   - Recommend bumping carried_decay to 0.9 - 0.95 so the accumulated
    #     carry stays bounded across hundreds of doc updates.
    #   - Precedence over hybrid_sessions / single_paper_sessions.
    everlasting_carry: bool = False

    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    run_name: str = "ttt-v1.1"

    wandb_enabled: bool = True
    wandb_project: str = "inplace-ttt"
    param_log_every: int = 50

    # In-loop holdout eval: per-paper ppl_fresh - ppl_carry gap on
    # `n` papers x eval_n_slices slices, twice (with/without carry).
    # eval_every=0 disables.
    #
    # Effective paper count `n`:
    #   - When the active dataset has a `source` column (SlimPajama)
    #     AND eval_n_papers_per_source > 0:
    #         n = n_sources * eval_n_papers_per_source
    #     Sampling is stratified: exactly `eval_n_papers_per_source`
    #     papers per source. Ensures every domain is represented every
    #     eval -- what per-source eval metrics need.
    #   - Otherwise: n = eval_n_papers, sampled uniformly.
    #
    # eval_min_tokens filters holdout to docs with >= this many tokens
    # BEFORE sampling. Guards against picking tiny StackExchange posts
    # that can't be sliced into eval_n_slices meaningful pieces.
    eval_every: int = 100
    eval_n_papers: int = 3
    eval_n_papers_per_source: int = 1
    eval_n_slices: int = 8
    eval_min_tokens: int = 2048
    eval_holdout_seed: int = 0


TTT_CFG = TTTConfig()
TRAIN_CFG = TrainConfig()
