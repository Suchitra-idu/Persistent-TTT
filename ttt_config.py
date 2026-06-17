"""
Shared configuration for In-Place TTT continual pretraining on Modal.

Single source of truth for paths, model identity, TTT hyperparameters,
and training hyperparameters. Both the training app and the inference
app import from here so the two can never drift apart (DRY).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Modal resources. CHANGE THESE to match your actual names.
# ---------------------------------------------------------------------------
CKPT_VOLUME_NAME = "ttt-checkpoints"         # created automatically if missing
HF_CACHE_VOLUME_NAME = "hf-hub-cache"        # caches model + dataset downloads

CKPT_MOUNT = "/ckpt"
HF_CACHE_MOUNT = "/hf-cache"

# The dataset lives on the HuggingFace Hub as parquet shards and is
# pulled via load_dataset(repo_id). Downloads land in HF_HOME (the cache
# volume above), so the Hub is only hit on the first run.
# If the repo is PRIVATE: `modal secret create huggingface HF_TOKEN=hf_...`
# then uncomment the secrets=[...] lines in train_modal.py / infer_modal.py.
DATASET_SOURCE = "suchitraIdu/arxiv-ml-16k"   # <-- your HF repo id

TEXT_COLUMN = "text"               # confirmed from the dataset viewer
TOKENS_EST_COLUMN = "tokens_est"   # cheap length pre-filter, no tokenizing

# The LAST n rows (newest arxiv ids) never enter training; they are the
# contamination-free pool for session_eval. Slow weights must not have
# seen the papers used to measure fast weight memory.
HOLDOUT_LAST_N = 200

BASE_MODEL = "Qwen/Qwen3-8B"

# Qwen3-8B architecture facts used for layer selection and the LoRA regex.
NUM_LAYERS = 36

# Every 6th layer, matching the In-Place TTT paper's drop-in recipe.
# Sweep placement later; this is the v1 baseline.
TTT_LAYER_INDICES = (5, 11, 17, 23, 29, 35)


@dataclass
class TTTConfig:
    """Hyperparameters of the In-Place TTT mechanism itself."""

    layer_indices: tuple = TTT_LAYER_INDICES

    # Chunk size for the chunk-wise fast weight update.
    # The paper's ablation found 512 and 1024 both good; 1024 is more
    # compute-efficient (half as many cumsum steps, larger einsums).
    # WARNING: any old checkpoint trained with a different chunk_size
    # will exhibit subtly different within-paper dynamics when loaded
    # under this config -- retrain after changing.
    chunk_size: int = 1024

    # Inner-loop learning rate eta for the fast weight update
    # W <- W + eta * V^T Z. NOT verified against the official repo,
    # tune on the overfit-100-papers run before the full run.
    eta: float = 1e-3

    # Divide each chunk delta by chunk_size so eta's scale is roughly
    # independent of C. Set False to match a raw-sum formulation.
    normalize_delta_by_chunk: bool = True

    # Causal Conv1D kernel width for the LM-aligned target
    # V = Conv1D(X0) @ W_target. Verify against the official repo.
    conv_kernel_size: int = 4

    # Frobenius norm clipping of the accumulated fast weight update at
    # inference, from the paper appendix (tau = 1e-5). Implemented as an
    # absolute cap on ||eta * cumulative_update||_F, but the paper's
    # exact formula is UNVERIFIED and an absolute 1e-5 cap would zero
    # out the mechanism. Disabled by default; enable only after checking
    # the appendix / official repo for the precise semantics.
    clip_enabled: bool = False
    clip_tau: float = 1e-5
    clip_at_inference_only: bool = True


@dataclass
class TrainConfig:
    """Outer-loop (continual pretraining) hyperparameters."""

    max_seq_len: int = 16384       # one paper per sequence, no packing in v1
    min_doc_tokens: int = 2048     # drop docs too short to span multiple chunks
    micro_batch_size: int = 1      # fixed at 1 by the session loop; fast
                                   # weight carry is per-stream by design
    grad_accum_steps: int = 4     # effective batch ~ 256k tokens
    num_epochs: int = 1

    # Three parameter groups, three learning rates.
    lr_lora: float = 1e-4          # pretrained weights adapted via LoRA
    lr_wdown: float = 2e-5         # pretrained fast weight initial state, move gently
    lr_new_modules: float = 2e-4   # Conv1D + W_target, fresh and zero-init

    weight_decay_full: float = 0.1
    weight_decay_lora: float = 0.0
    warmup_ratio: float = 0.02
    warmup_min_steps: int = 10
    max_grad_norm: float = 1.0

    # LoRA. Alpha at 2x rank per current practice.
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Session-persistent training. A "session" is n consecutive papers,
    # n ~ Uniform[session_papers_min, session_papers_max]. Each paper is
    # its own forward/backward exactly as before; the ONLY difference is
    # that fast weight deltas carry across papers within a session
    # (detached, TBPTT-style) and reset at session boundaries.
    session_training: bool = True
    session_papers_min: int = 2
    session_papers_max: int = 6

    # Random within-paper slicing inside a session. With probability
    # slice_prob, a paper is split into k ~ Uniform[slice_min, slice_max]
    # consecutive token-range slices at random boundaries; each slice
    # becomes its own forward/backward inside the same session, so fast
    # weight carry now spans BOTH intra-paper slice boundaries and
    # inter-paper boundaries. No text parsing involved -- boundaries are
    # token positions. Set slice_prob=0 to disable; behavior then matches
    # the pre-slicing schedule exactly. slice_min_tokens guards against
    # slices smaller than one TTT chunk (where the within-slice scan is
    # a no-op).
    slice_prob: float = 0
    slice_min: int = 2
    slice_max: int = 6
    slice_min_tokens: int = 1024

    # Single-paper-session training. When True, each session is ONE
    # paper sliced into k ~ Uniform[single_paper_slices_min,
    # single_paper_slices_max] consecutive random pieces; the
    # session_papers_* and slice_* fields above are IGNORED. Every item
    # in a session is guaranteed to share content with the rest, so the
    # carry has a real signal to learn from and there's no risk of an
    # unrelated paper being silently "carried" between items as noise.
    # Trades the cross-paper memory training signal for a cleaner
    # intra-paper one.
    single_paper_sessions: bool = False
    single_paper_slices_min: int = 2
    single_paper_slices_max: int = 6

    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    run_name: str = "ttt-v1"

    # Observability. Aggregates go to wandb every optimizer step,
    # per-paper signals every micro step, heavier param-health metrics
    # every param_log_every optimizer steps.
    wandb_enabled: bool = True
    wandb_project: str = "inplace-ttt"
    param_log_every: int = 50


TTT_CFG = TTTConfig()
TRAIN_CFG = TrainConfig()
