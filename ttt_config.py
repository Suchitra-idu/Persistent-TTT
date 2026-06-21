"""
Shared configuration for In-Place TTT continual pretraining on Modal.

Single source of truth for paths, model identity, TTT hyperparameters,
and training hyperparameters. Both the training app and the inference
app import from here so the two can never drift apart (DRY).
"""

import os
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

# Model identity. Default is the small variant for cheap iteration;
# flip to 8B for the full run via the env var:
#   modal run ...                       # default, Qwen3-0.6B
#   TTT_MODEL_SIZE=8B modal run ...     # the real run
# TTT_BASE_MODEL fully overrides if you need a non-Qwen3 path (rare).
# The rest of the code is model-size invariant: NUM_LAYERS and the TTT
# layer schedule are derived from model.config.num_hidden_layers in
# model_setup.build_model, so changing this env var is the ONLY edit
# needed to swap sizes.
MODEL_SIZE = os.environ.get("TTT_MODEL_SIZE", "0.6B")
BASE_MODEL = os.environ.get("TTT_BASE_MODEL", f"Qwen/Qwen3-{MODEL_SIZE}")

# TTT layer schedule. stride=2 + start=1 yields 14 TTT layers on Qwen3-0.6B
# (28 transformer blocks), up from 9 at stride=3. Fast weight capacity is
# a bigger fraction of total model capacity on the small model, so we can
# afford denser placement. Override via env if you want the old every-3rd
# schedule or something else.
LAYER_STRIDE = int(os.environ.get("TTT_LAYER_STRIDE", "2"))
LAYER_START = int(os.environ.get("TTT_LAYER_START", "1"))


def derive_ttt_layer_indices(num_layers: int,
                             stride: int = LAYER_STRIDE,
                             start: int = LAYER_START) -> tuple:
    """Every `stride`-th layer starting at `start`, capped at num_layers-1.
    Default (stride=2, start=1) places TTT on alternating layers:
        Qwen3-0.6B (28 layers) -> 14 TTT layers (1, 3, 5, ..., 27)
        Qwen3-8B   (36 layers) -> 18 TTT layers (1, 3, 5, ..., 35)
    Quadruples fast-weight capacity vs the paper's every-6th baseline by
    distributing TTT sites across more depths."""
    return tuple(range(start, num_layers, stride))


# Default protect-list for the content-token loss mask. These terms
# carry document-specific signal in ML papers but are frequent enough
# that pure-frequency masking can incorrectly catch them. Each term is
# expanded into multiple BPE variants (bare / leading-space /
# capitalized / all-caps) at protect-time; only variants that tokenize
# to a single BPE piece protect that piece, so single-character prefix
# false positives are not a concern. Organized by category so the list
# stays auditable as the field moves.
LOSS_MASK_DEFAULT_PROTECT_TERMS = (
    # --- Modern architectures & layer types ---
    "transformer", "attention", "convolution", "convolutional",
    "embedding", "encoder", "decoder", "autoencoder",
    "perceptron", "mlp", "lstm", "rnn", "cnn", "gnn",
    "gan", "vae", "diffusion", "residual", "recurrent",
    "feedforward", "capsule",
    # --- Activations, normalization, regularization ---
    "softmax", "sigmoid", "relu", "gelu", "tanh",
    "activation", "normalization", "regularization", "dropout",
    "layernorm", "batchnorm", "groupnorm", "rmsnorm",
    # --- Modern foundation models / families ---
    "bert", "gpt", "llama", "qwen", "mistral", "claude",
    "gemini", "palm", "llm", "vit", "clip", "deepseek",
    "phi", "gemma", "falcon", "mixtral", "dalle",
    # --- Reinforcement learning vocabulary ---
    "policy", "reward", "agent", "action", "trajectory",
    "ppo", "dqn", "sac", "actor", "critic", "episode",
    "bandit", "exploration", "exploitation", "rollout",
    # --- Statistical / mathematical foundations ---
    "bayesian", "markov", "gaussian", "kernel", "manifold",
    "lipschitz", "convex", "lagrangian", "hessian", "jacobian",
    "eigenvalue", "eigenvector", "tensor", "scalar",
    "entropy", "divergence", "kullback", "leibler",
    "wasserstein", "frobenius", "posterior", "prior",
    "likelihood", "mcmc", "mle",
    # --- Greek letters (common as hyperparams and variables) ---
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega",
    # --- Optimization / training concepts ---
    "optimizer", "logits", "perplexity", "checkpoint",
    "pretraining", "finetuning", "adamw", "adam", "rmsprop",
    "sgd", "momentum", "lora", "qlora", "rlhf", "dpo",
    # --- Task / objective names ---
    "classification", "regression", "segmentation",
    "detection", "translation", "summarization",
    "recognition", "captioning", "parsing",
    # --- Common benchmarks / datasets ---
    "imagenet", "mnist", "cifar", "coco", "glue",
    "squad", "bleu", "rouge", "wmt", "mmlu",
    "gsm8k", "humaneval", "bigbench",
)


@dataclass
class TTTConfig:
    """Hyperparameters of the In-Place TTT mechanism itself."""

    # Populated lazily from model.config.num_hidden_layers in
    # model_setup.build_model so swapping BASE_MODEL needs no edits here.
    # Tests construct TTTConfig with an explicit tuple; runtime code must
    # go through build_model before reading this.
    layer_indices: tuple | None = None

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
    # Bumped from 1e-2 because eta gates the gradient on W_target
    # (output is linear in W_target with coefficient eta), so a small
    # eta starves W_target's gradient signal in a vicious cycle.
    eta: float = 1e-1

    # Divide each chunk delta by chunk_size so eta's scale is roughly
    # independent of C. Set False to match a raw-sum formulation.
    normalize_delta_by_chunk: bool = True

    # Causal Conv1D kernel width for the LM-aligned target
    # V = Conv1D(source) @ W_target. Larger kernel gives V more left context
    # to encode in each position -- helpful when the carry needs to store
    # multi-token structure (definitions, named entities, math notation)
    # rather than just single-token features.
    conv_kernel_size: int = 8

    # Source tensor that feeds target_conv. The paper uses the raw token
    # embedding X0 ("embedding"); each TTT layer then sees the same V up
    # to its own W_target. "hidden_state" instead feeds each TTT layer
    # its OWN input hidden_states (post-attention, post-prior-layers),
    # which is what every modern linear-attention paper (RWKV, RetNet,
    # Mamba) does. Trade-off: more expressive V, but streaming inference
    # needs per-layer rolling buffers of past hidden_states instead of
    # the shared embedding tap.
    v_source: str = "hidden_state"

    # When True, target_conv sees PAST + CURRENT + FUTURE tokens
    # (symmetric pad) instead of past + current only (causal left-pad).
    # Lets V at position n encode information about positions n+1..n+K/2,
    # which materially helps when the carry should remember multi-token
    # structure on the right of each position (e.g. an entity completed
    # several tokens later).
    #
    # WARNING: breaks the chunk-causality invariant that makes the parallel
    # scan equivalent to a sequential apply-then-update under standard
    # next-token-prediction training. Under CE loss, the carry now leaks
    # ground-truth tokens from the right into earlier positions' updates,
    # so the loss can decrease via shortcut rather than via learning. Use
    # this with a prefill+continuation training objective (or accept the
    # leakage for ablations). Streaming inference IGNORES this flag --
    # future tokens are not available across call boundaries -- so the
    # stream path remains strictly causal regardless of this setting.
    v_bidirectional: bool = False

    # Per-position output gating: output = base + sigmoid(W_g h) * eta * ttt_out
    # When False (default), the carry is added uniformly to every position --
    # matches the original paper formulation. When True, each TTT layer
    # learns a per-position scalar gate from the hidden state, so the model
    # decides where the carry should matter.
    #
    # Why this exists: raw linear-attention fast weights have a known
    # training-difficulty failure mode (cumsum averaging washes out the
    # gradient on W_target, the loss surface around any working W_target
    # is flat). Gating opens a NEW gradient path -- the gate's own learning
    # rewards W_target for being useful when gated on -- avoiding that
    # starvation. Off by default so flipping on is a clean A/B.
    output_gate: bool = False

    # Bias init for the output gate. At -2.0 the gate starts at
    # sigmoid(-2) ~= 0.12, so the carry is mostly closed by default and
    # the model has to learn to OPEN it where it's actually useful. This
    # is "regularization by init" -- pushes against the overfit failure
    # mode where the gate learned to be wide-open and amplified noisy V
    # directions on held-out data (single_paper_eval -3.9 PPL gap).
    output_gate_bias_init: float = -2.0

    # L2 regularization on the gate output, added to training loss as
    # gate_reg_weight * mean(sigmoid(W_g h)^2). Encourages "default closed"
    # gates more aggressively than just the bias init alone; useful when
    # the gate is consistently high after warmup. Set to 0 to disable.
    gate_reg_weight: float = 0.0

    # Frobenius norm clipping of the accumulated fast weight update at
    # inference, from the paper appendix (tau = 1e-5). Implemented as an
    # absolute cap on ||eta * cumulative_update||_F, but the paper's
    # exact formula is UNVERIFIED and an absolute 1e-5 cap would zero
    # out the mechanism. Disabled by default; enable only after checking
    # the appendix / official repo for the precise semantics.
    clip_enabled: bool = False
    clip_tau: float = 1e-5
    clip_at_inference_only: bool = True

    def __post_init__(self):
        if self.v_source not in ("embedding", "hidden_state"):
            raise ValueError(
                "v_source must be 'embedding' or 'hidden_state', "
                f"got {self.v_source!r}"
            )


@dataclass
class TrainConfig:
    """Outer-loop (continual pretraining) hyperparameters."""

    max_seq_len: int = 16384       # one paper per sequence, no packing in v1
    min_doc_tokens: int = 2048     # drop docs too short to span multiple chunks
    micro_batch_size: int = 1      # fixed at 1 by the session loop; fast
                                   # weight carry is per-stream by design
    grad_accum_steps: int = 16     # effective batch ~ 256k tokens
    num_epochs: int = 1

    # Three parameter groups, three learning rates. lr_lora was 1e-4 but
    # on small-data runs (~100 docs) that overfits the LoRA fast enough to
    # damage held-out base perplexity; the TTT mechanism shows a real gap
    # but on top of a corrupted base. 3e-5 lets LoRA adapt gently while
    # the TTT new modules (lr_new_modules) still move fast enough to
    # converge a useful update rule.
    lr_lora: float = 3e-5          # pretrained weights adapted via LoRA
    lr_wdown: float = 2e-5         # pretrained fast weight initial state, move gently
    lr_new_modules: float = 3e-4   # Reduced from 1e-3 after the randn-init +
                                   # 1e-3 combination produced a HARMFUL carry
                                   # (single_paper_eval -3.9 per-paper PPL,
                                   # monotonic with state_ratio). With zero
                                   # W_target init we want gentle, gradient-
                                   # respecting escape from the zero region;
                                   # 3e-4 is slow but avoids over-eager
                                   # learning of training-specific directions.

    weight_decay_full: float = 0.1
    weight_decay_lora: float = 0.0
    warmup_ratio: float = 0.02
    warmup_min_steps: int = 10
    max_grad_norm: float = 10.0   # Catches occasional spikes (observed |g| ~ 5.5
                                 # at random steps) without clipping the routine
                                 # ~0.5 gradient norms.

    # LoRA. Alpha at 2x rank per current practice.
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Session-persistent training. A "session" is n consecutive papers,
    # n ~ Uniform[session_papers_min, session_papers_max]. Each paper is
    # its own forward/backward exactly as before; the ONLY difference is
    # that fast weight deltas carry across papers within a session
    # (detached, TBPTT-style) and reset at session boundaries.
    session_training: bool = False
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

    # Content-token loss masking. When enabled, CE loss is computed only
    # on token positions whose token_id is NOT among the most-frequent
    # tokens that together account for (1 - loss_mask_keep_fraction) of
    # total training-corpus occurrences. Shifts the gradient toward
    # content-bearing tokens (entities, numbers, technical terms) and
    # away from syntactic glue. Two effects:
    #   1. LoRA overfits less on small corpora, preserving base ppl.
    #   2. TTT modules see a sharper retention-critical training signal.
    # 1.0 disables masking even when loss_mask_enabled=True.
    #
    # Disabled by default because masking starves the indirect-gradient
    # path that W_target depends on (gradient on V at masked positions
    # comes only via downstream unmasked positions through the carry,
    # and at keep_fraction=0.5 that's at least halved). Enable for runs
    # where you specifically want the LoRA-corruption protection on
    # small-data overfit experiments.
    #
    # keep_fraction semantics: the mask covers the top-K most-frequent
    # tokens whose cumulative occurrences in the *baseline distribution*
    # add up to (1 - keep_fraction). The baseline is either the
    # external reference (preferred, see loss_mask_reference_counts_path
    # below) or the training corpus itself (fallback). Under the
    # external reference, 0.5 still means "drop the top-N most-common-
    # in-general-English tokens accounting for half of reference
    # positions"; under in-corpus fallback it means "drop the top-N
    # most-common-in-our-papers tokens." Both modes use the same field.
    loss_mask_enabled: bool = False
    loss_mask_keep_fraction: float = 0.5

    # Domain-content protect-list. Token ids reached by tokenizing any of
    # these terms (multiple BPE variants per term) are force-unmasked
    # even when they sit above the frequency threshold. This is the
    # defense against the failure mode the frequency mask has on a
    # domain corpus: words that are "common in ML" but "content
    # everywhere else" (' transformer', ' attention', ' diffusion', etc.)
    # need to stay in the gradient. Pass () to disable the override.
    loss_mask_protect_terms: tuple = field(
        default_factory=lambda: LOSS_MASK_DEFAULT_PROTECT_TERMS
    )

    # External-reference frequency baseline. When this path is non-empty
    # AND the file exists at training start, the common-token mask is
    # built from a precomputed reference-corpus unigram distribution
    # (e.g. wikitext-103) instead of the training corpus's own
    # distribution. This is the structural fix to the "domain-glue
    # always looks common" failure mode: words like ' model',
    # ' training', ' data', ' attention' show up high in the ML-corpus
    # frequency table only because every ML paper uses them, but under
    # a general-English baseline they sit far below the threshold and
    # stay in the gradient signal. Function words (' the', ' of') are
    # still caught -- they dominate any English corpus.
    #
    # Default path matches build_reference_counts's default output;
    # run that Modal function once to populate it, then training
    # picks it up automatically. Falls back to in-corpus frequency
    # with a clear log line when the file is absent so first-run UX
    # is not blocked.
    loss_mask_reference_counts_path: str = (
        "/ckpt/loss_mask/reference_wikitext103.pt"
    )

    # Predicate-protect: numeric tokens. Numbers in scientific papers
    # carry content (hyperparameter values, model sizes, benchmark
    # scores, dataset statistics). Single-digit tokens 0-9 sit near the
    # top of the frequency table on this corpus so they would otherwise
    # always be masked; the term-based protect-list can't help because
    # single-character pieces fail its length gate. This sweep walks
    # the masked set after build and unmasks anything whose decoded
    # form is a pure-digit string. Set False for corpora where digits
    # are mostly enumeration glue ("1. Introduction") rather than
    # content -- but on arxiv-ml-16k they are content.
    loss_mask_protect_numeric: bool = True

    # Predicate-protect: math symbols. In a scientific corpus, single-
    # character math operators ARE content -- they signal equations
    # (=), tensor operations (@), exponents (^), subscripts (_), LaTeX
    # commands (\), and norms (|). The term-based protect can't catch
    # them (the length gate rejects single-character pieces); the
    # numeric predicate doesn't match (non-digit). This is the third
    # predicate path, exactly the same shape as protect_numeric but
    # with a configurable symbol set.
    #
    # Defaults to unambiguously math-leaning ASCII operators. NOT
    # included by default: parentheses/brackets/braces (heavy prose
    # use), periods/commas/colons/semicolons (pure punctuation), and
    # unicode math glyphs (often multi-byte BPE-split). Add to the
    # tuple if your corpus uses them as content.
    loss_mask_protect_symbols: tuple = (
        "=", "@", "^", "_", "\\", "+", "-", "*", "/", "|", "<", ">",
    )

    # Mask the first N tokens of every paper-START item (i.e. items
    # whose SessionItem.start == 0). Every arxiv paper opens with
    # near-identical boilerplate -- title formatting, author list, the
    # word "Abstract", "1. Introduction" -- whose next-token prediction
    # is trivial and dominated by corpus style. Training on these
    # positions burns gradient capacity learning that all papers look
    # the same; masking them concentrates the signal on the first
    # genuinely informative tokens. Mid-paper slices (start > 0) are
    # unaffected -- those positions are real content. Set to 0 to
    # disable.
    loss_mask_first_tokens: int = 16

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
