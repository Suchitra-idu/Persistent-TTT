# In-Place TTT with Cross-Session Persistent Domain Memory

Continual test-time adaptation of Qwen3 (0.6B → 8B) via In-Place
fast-weight TTT on streaming arXiv ML papers. Builds on
"In-Place Test-Time Training" (ByteDance Seed, arXiv 2604.06169). The
research contribution is replacing the paper's per-document fast weight
resets with **session-level persistence** — domain memory written into
the weights by one paper survives into the next paper (and across
sessions), with no shared context window — plus an **EMA staging
decay** so the carry stays in a useful magnitude regime instead of
saturating the clip.

Everything runs on Modal. Nothing touches the local machine except
`modal run` commands, the chat REPL (`python chat_client.py`), and the
local math test.

---

## Folder structure

```
Our-TTT/
├── README.md
├── ttt_config.py        TTTConfig + TrainConfig + module-level constants
├── inplace_ttt.py       the TTT mechanism (pure PyTorch, no Modal)
├── ttt_wiring.py        LoRA regex, param groups, ckpt I/O (extracted from inplace_ttt)
├── model_setup.py       model assembly shared by train and inference
├── data_utils.py        dataset loading + the holdout split
├── observability.py     wandb telemetry and metric collectors
├── train_utils.py       session schedule, slicing, loss-mask helpers, grad norms
├── chat_utils.py        pure chat helpers (sampling, prompt format, stop ids)
├── train_modal.py       Modal app: training + sanity_check + build_reference_counts
│                        + diagnose_loss_mask
├── infer_modal.py       Modal app: TTTInference + eval + generation entrypoints
├── chat_client.py       local REPL that talks to the deployed TTTInference class
└── tests/               local CPU suite, no Modal/GPU/downloads (~3s)
    ├── conftest.py            shared tiny-module fixtures
    ├── test_scan_math.py      scan vs sequential reference, both modes
    ├── test_mechanism.py      identity, causality, stream/scan, evolve, clip
    ├── test_wiring.py         LoRA regex, param groups, checkpoint I/O
    ├── test_session.py        carry lifecycle, staging idempotence, schedule, slicing
    ├── test_loss_mask.py      loss-mask build, protect helpers, reference-count loader
    ├── test_chat_utils.py     sampling, prompt format, stop-token assembly
    └── test_observability.py  telemetry safety, metric collectors
```

Run the suite with `python -m pytest tests/ -q` (needs only torch,
numpy, pytest). Run it after ANY change to `inplace_ttt.py`,
`ttt_wiring.py`, `train_utils.py`, or `observability.py`, and before
every training launch.

The Python modules MUST stay flat at the project root. Modal ships them
into containers via `image.add_local_python_source("ttt_config", ...)`,
which imports by module name from the directory where `modal run`
executes. Moving them into `src/` or a package breaks both apps.

### Who imports whom

```
train_modal.py ─┬─> model_setup.py ──> inplace_ttt.py, ttt_wiring.py ──> ttt_config.py
                ├─> data_utils.py  ──────────────────────────────────> ttt_config.py
                ├─> train_utils.py  (loss mask, session schedule)
                └─> observability.py

infer_modal.py ─┬─> model_setup.py, data_utils.py, inplace_ttt.py, chat_utils.py

chat_client.py ─── modal.Cls.from_name("inplace-ttt-infer", "TTTInference")

tests/          └─> inplace_ttt.py, ttt_wiring.py, train_utils.py, ttt_config.py
```

`inplace_ttt.py` and `ttt_wiring.py` have zero Modal dependencies on
purpose, so the mechanism is unit-testable locally and reusable outside
Modal.

---

## What each file owns

**ttt_config.py.** Single source of truth. `TTTConfig` holds the
mechanism (layer indices, chunk size, `eta`, conv kernel, output gate,
clipping, `carried_decay`). `TrainConfig` holds the outer loop
(learning rates per parameter group, LoRA shape, session sizes, loss
mask, in-loop eval, wandb settings). Module-level constants hold Modal
volume names, the HF dataset repo id, the holdout size, and the model
selection env vars.

**Model + TTT layer selection is env-var driven:**
- `TTT_MODEL_SIZE` (default `"8B"`) → `BASE_MODEL = "Qwen/Qwen3-{SIZE}"`
- `TTT_LAYER_STRIDE` (default `2`) — every stride-th layer becomes a TTT layer
- `TTT_LAYER_START` (default `1`) — first TTT layer index
- `TTT_BASE_MODEL` — full override for non-Qwen3 paths

**inplace_ttt.py.** The mechanism. `InPlaceTTTMLP` replaces the gated
MLP on the TTT layers and implements both execution paths: a parallel
chunk scan for training and whole-sequence eval, and a stateful stream
for autoregressive generation. Owns `patch_model_with_ttt`, session
lifecycle (`reset_session_state`, `advance_session_state`,
`session_state_norms`), stream lifecycle (`reset_fast_weights`,
`stream_pending_progress`), gate stats (`gate_stats`,
`stateful_state_norms`), and fast-weight export/import.

**ttt_wiring.py.** LoRA target regex, param-group construction,
TTT-parameter grad unfreezing, and TTT checkpoint save/load. Extracted
from `inplace_ttt.py` so the mechanism module has zero LoRA/PEFT deps.

**model_setup.py.** One function, `build_model`, assembles base model
→ TTT patch → LoRA wrap → grad unfreezing → checkpoint loading in the
single correct order. Train and inference both call it, so they can
never assemble the model differently.

**data_utils.py.** `open_dataset` pulls the parquet dataset from the
HF Hub (cached on a Modal volume). `split_holdout` reserves the newest
`HOLDOUT_LAST_N` papers (default 200) as a contamination-free eval
pool.

**train_utils.py.** Pure functions: the session scheduler,
`SessionItem` / `slice_doc` / `build_session_items` slicing,
`make_single_paper_sessions`, per-group gradient norms,
`expected_items_per_doc` LR-schedule sizing, and the loss-mask
helpers (`build_common_token_mask`, `apply_loss_mask`,
`apply_protect_passes`, `load_reference_counts`, `protect_by_predicate`
and its numeric/symbol/term specializations).

**observability.py.** `Telemetry` wraps wandb and can never crash or
stall a run. Collectors: `gpu_stats`, `param_health` (heavier drift +
gate mean/std), `session_metrics`, `snapshot_wdown`.

**train_modal.py.** The training app. Session-scheduled loop, three
optimizer param groups, 8-bit AdamW, gradient checkpointing,
nonfinite-loss guard, checkpointing to a Modal volume, full telemetry,
in-loop eval every `eval_every` steps. Also exposes `sanity_check`,
`build_reference_counts` (builds the wikitext-103 unigram reference for
loss masking), and `diagnose_loss_mask` (dumps mask stats + spot
checks).

**infer_modal.py.** The inference app. `TTTInference` class exposes
`perplexity`, `session_perplexity`, `generate`, `save_session`,
`chat_reset`, `chat_turn`, plus `fetch_holdout_texts`. Local
entrypoints: `holdout_eval`, `single_paper_eval`, `session_eval`,
`generate_cli`, `holdout_generate`, `compare_ppl`.

Class parameters: `ckpt` (checkpoint name or empty for base model),
`load_ttt` (bool; when False, loads the trained LoRA but skips
`ttt_params.pt` so W_target stays zero — the "LoRA-only" ablation
point).

**chat_utils.py.** Pure helpers for chat: top-p sampling, Qwen3 chat
template application, stop-token assembly, `<think>...</think>`
splitting, special-token stripping. No Modal / GPU deps.

**chat_client.py.** Local REPL that connects to the deployed
`TTTInference` class via `modal.Cls.from_name(...)`. `modal run`
doesn't forward stdin, so the interactive loop has to live in a plain
Python process.

---

## The mechanism in six lines

For TTT layers, with activations `Z = silu(gate(H)) * up(H)` and
LM-aligned targets `V = CausalConv1D(source) @ W_target` chunked into
chunks of size C:

```
apply:    O_[i] = Z_[i] @ (W_down + eta * S_i)^T
gate:     O_[i] = base_out + sigmoid(W_g h) * (O_[i] - base_out)
update:   S_{i+1} = S_i + V_[i]^T @ Z_[i] / C          (S_0 = 0)
```

Chunk `i` is processed with updates from strictly earlier chunks
(exclusive cumsum + causal conv). **Session mode** changes two things:
`S_0` starts from `carried_delta` (previous items' final state, fp32,
detached — truncated BPTT), and at the boundary
`carried_delta ← carried_decay · carried_delta + this_item_total`
(EMA staging). Set `carried_decay = 1.0` to recover pure accumulation.

**Output gate** (`TTTConfig.output_gate=True`): a per-position sigmoid
gate modulates the TTT contribution. Bias initialized to −2 so the
gate starts mostly closed (sigmoid ≈ 0.12), forcing the mechanism to
learn to open. `gate_reg_weight` optionally regularizes the gate's L2.

**v_source** (`TTTConfig.v_source`): either `"embedding"` (raw
token embeddings, kept for the paper reference) or `"hidden_state"`
(per-layer input, more expressive; default). Adds one small per-layer
context buffer under streaming inference.

### Session shape

Training composes sessions in one of two shapes:

- **Multi-paper** (default): each session contains `k ∼ U[session_papers_min,
  session_papers_max]` papers (default 2..6). Papers may be randomly sliced
  into token-range sub-papers per `slice_prob` (default 0 → disabled).
  Fast-weight carry threads through every paper (and slice) in the
  session, decayed at each boundary by `carried_decay`.
- **Single-paper** (`single_paper_sessions=True`, or `--single-paper 1`
  on the CLI): each session is ONE paper cut into `k ∼ U[single_paper_slices_min,
  single_paper_slices_max]` consecutive pieces (default 2..6). Every
  item shares content with the rest of the session, so the carry has
  a signal it can actually learn from without cross-paper noise.

`session_training` (or `--session 1`) turns session mode on for the
model itself. Without it, `session_mode=False` and the carry never
staged (each item is independent).

### Trainable parameters — three groups, three learning rates

| group  | what                                                                | LR default (0.6B) | why |
|--------|---------------------------------------------------------------------|-------------------|-----|
| lora   | attention + gate/up (all layers), down_proj on **non-TTT** layers  | `1e-5`            | pretrained; adapted via LoRA `r=16`, α=32 |
| wdown  | down_proj on TTT layers (full)                                     | `3e-5`            | pretrained fast weight init; move gently |
| new    | W_target + target_conv (full)                                      | `2e-5`            | fresh, zero/passthrough init |

At bigger model sizes (4B, 8B), all three LRs are typically bumped
~10× to compensate for smaller per-parameter gradients at scale (see
notes below).

W_target is zero-initialized, so the whole model is exact-identity to
base Qwen3 at step 0 (verified by `sanity_check`). LoRA never touches
down_proj on TTT layers — the LoRA regex enforces this.

---

## Setup

1. Install Modal locally and authenticate (`pip install modal`,
   `modal setup`).
2. Edit `ttt_config.py`:
   * `DATASET_SOURCE` — your HF dataset repo id
   * Volume names if you want different ones (both auto-create)
3. Create secrets:
   ```
   modal secret create wandb WANDB_API_KEY=...
   ```
   If the dataset repo is private, also:
   ```
   modal secret create huggingface HF_TOKEN=hf_...
   ```
   and append `modal.Secret.from_name("huggingface")` to `SECRETS` in
   both app files.
4. Run the test suite once:
   ```
   pip install pytest
   python -m pytest tests/ -q
   ```

---

## Run protocol (in order)

### 0. Wiring check

Must print a max logit diff near zero and pass the assert. A failure
means the TTT patch broke bit-exact identity; do not train.
```
modal run train_modal.py::sanity_check
```

### 1. (Optional) Build the wikitext-103 loss-mask reference

Content-token loss masking is off by default (`loss_mask_enabled=False`).
If you enable it, the reference-counts file makes the mask
domain-agnostic — otherwise it falls back to in-corpus frequency (which
tends to mask ML-specific glue words). Build once per tokenizer change:
```
modal run train_modal.py::build_reference_counts
```
Saved to `/ckpt/loss_mask/reference_wikitext103.pt` on the checkpoint
volume; `TRAIN_CFG.loss_mask_reference_counts_path` already points
there.

Inspect what a mask would keep/drop for a given corpus:
```
modal run train_modal.py::diagnose_loss_mask --limit-docs 100 --use-reference
```

### 2. Overfit smoke test

Loss must fall fast, `grad/new` must be nonzero from early on, and
`session/state_ratio_*` must grow smoothly (not jump to huge values).
```
modal run --detach train_modal.py::train \
    --limit-docs 100 --num-epochs 5 --session 1 --single-paper 1
```

### 3. Real run

Pick model size and layer stride via env vars. Example (0.6B, every
2nd layer):
```
TTT_MODEL_SIZE=0.6B modal run --detach train_modal.py::train \
    --limit-docs 2000 --num-epochs 1 --session 1 --single-paper 1
```

8B needs a smaller stride (`TTT_LAYER_STRIDE=4`) and a bigger
`chunk_size` (400 in config) to fit in 80 GB. See "Memory notes at
scale" below.

Training CLI flags:
- `--limit-docs N`: cap training documents (0 = full training split)
- `--num-epochs N`: override `TrainConfig.num_epochs`
- `--grad-accum N`: override `grad_accum_steps`
- `--session 0|1`: force `session_training` on/off
- `--single-paper 0|1`: force `single_paper_sessions` on/off

Checkpoints save every `TrainConfig.save_every` steps (default 200)
under `<CKPT_MOUNT>/<run_name>/step_<N>/`.

### 4. Evaluate

**Three-way comparison** — one command runs BASE (untouched Qwen3),
LORA-ONLY (trained LoRA, `W_target=0`), and FULL (LoRA + TTT):
```
modal run infer_modal.py::holdout_eval --n-papers 5 --ckpt step_400
```
- Omit `--ckpt` to see only BASE (nothing to compare against).
- `holdout_eval` samples from the newest `HOLDOUT_LAST_N` arxiv IDs.

For BASE and LORA-ONLY: `state` should be exactly `0.00e+00` at every
row and `ppl carry == ppl fresh`. If not, wiring is broken (same signal
as `sanity_check`). For FULL: `state` grows over the session, and the
`gap = fresh - carry` column is the TTT signal.

**Single-paper eval** — clean within-paper carry signal:
```
modal run infer_modal.py::single_paper_eval --n-slices 8 --ckpt step_400
```
One held-out paper cut into N equal-token slices, run as one session.
Because content distribution is constant across positions, per-slice
`gap` isolates the carry contribution from paper-to-paper ppl variance.

**Qualitative check** — see what the model actually produces:
```
modal run infer_modal.py::holdout_generate --n-papers 1 --ckpt step_400 --greedy
```
Prints prompt + carry-on continuation + carry-off continuation
side-by-side. Pass `--greedy` so the only differing factor is the
carry (otherwise sampling adds noise).

Other eval entrypoints:
```
# Single-text TTT on/off ppl gap
modal run infer_modal.py::compare_ppl --text-path paper.txt --ckpt step_400

# Sampled generation, fast weights evolving over prompt + output
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_400

# Custom local .txt files as one session
modal run infer_modal.py::session_eval --papers-dir ./papers --ckpt step_400
```

### 5. Chat (interactive REPL)

Chat runs as a local Python process so stdin isn't swallowed by
`modal run`. Deploy the app once, then run the client:

```
modal deploy infer_modal.py
python chat_client.py --ckpt step_400
```

Options: `--evolve / --no-evolve`, `--enable-thinking` (Qwen3 thinking
mode — OFF by default because our LoRA/TTT was trained on raw papers,
not `<think>` traces), `--system "..."`, `--max-new-tokens`,
`--temperature`, `--top-p`, `--top-k`, `--from-snapshot NAME`,
`--debug`.

REPL commands:
- `/m` — multiline / paste mode; ends on a blank line
- `/save <name>` — persist current fast weights to
  `<run_name>/sessions/<name>.pt`
- `/reset` — drop fast-weight state, keep model loaded
- `/quit` — exit

**Invariant (do NOT relax):** each turn the model sees only the
current turn's prompt (optional static system message + the current
user message via Qwen3's chat template). Prior turns are NOT re-fed as
context, and `past_key_values` from earlier turns is discarded at the
turn boundary. The TTT fast-weight state (`state.delta` + the buffered
partial chunk) is the SOLE cross-turn memory channel. Re-feeding
history would turn this into a context-window memory test, not a
mechanism test.

Resume from a saved fast-weight snapshot on a later run (slow weights
from `--ckpt` plus fast-weight delta from the snapshot):
```
python chat_client.py --ckpt step_400 --from-snapshot mychat
```

---

## The `evolve` and `load_ttt` switches

`evolve=True` lets fast weights update chunk by chunk as text streams
in. `evolve=False` freezes evolution; previously accumulated or
imported state is still applied. With no state loaded, evolve=False is
plain slow weights only. Programmatic: `set_ttt_evolve(model, bool)`.

`load_ttt=False` on `TTTInference` skips loading `ttt_params.pt`, so
`W_target` and the TTT-layer `W_down` stay at their pretrained-plus-
init values. Combined with `--ckpt step_N`, this isolates LoRA's
contribution from TTT's — the LORA-ONLY column of `holdout_eval`.

Three configurations, three points of comparison:

| ckpt         | load_ttt | column       | what it measures |
|--------------|----------|--------------|------------------|
| `""`         | *       | BASE         | untouched Qwen3 pretraining |
| `step_N`     | False    | LORA-ONLY    | LoRA-only slow-weight adaptation |
| `step_N`     | True     | FULL         | LoRA + TTT (the full trained model) |

---

## Checkpoints layout

```
ttt-checkpoints volume
├── <run_name>/                       (TrainConfig.run_name, default "ttt-v1.1")
│   ├── step_200/
│   │   ├── adapter/                  PEFT LoRA adapter (save_pretrained)
│   │   └── ttt_params.pt             W_down (TTT layers), W_target, target_conv
│   ├── step_400/ ...
│   └── sessions/
│       └── <name>.pt                 persisted fast weight snapshots
└── loss_mask/
    └── reference_wikitext103.pt      unigram counts from wikitext-103
```

A fast weight snapshot is only valid for the exact slow weights it was
created under. Loading a snapshot after further training silently
applies a delta against a `W0` that no longer exists.

---

## Observability, metric → failure-mode map

Every optimizer step logs to wandb (project `inplace-ttt`). Heavier
`health/*` metrics log every `param_log_every` (50) steps.

| metric | healthy | failure it exposes |
|---|---|---|
| `train/loss` | monotone decrease early, plateaus late | flat from step 0 = broken tap or bad LR |
| `grad/new` | initial spike then decays to steady state | ~0 throughout while `grad/lora` healthy = X0 tap or target computation broken |
| `grad/lora`, `grad/wdown` | stable, bounded | dead or exploding groups |
| `train/grad_clip_ratio` | ~1.0 most of the time | persistently below 1 = clipping is eating updates |
| `session/state_ratio_L*` | grows within a session, bounded across sessions | steady climb past ~10 with `carried_decay=1.0` = unbounded fast-weight growth; use decay |
| `health/w_target_L*`, `health/conv_L*` | rising then flattening | flat from the start = new modules stuck in dead basin (try non-zero init or freeze slow weights briefly) |
| `health/wdown_drift_L*` | small (< 0.1) | TTT layers' W_down leaving the pretrained basin |
| `health/gate_mean_L*` | in (0.1, 0.9) | stuck at ~0 (gate closed → TTT suppressed) or ~1 (gate open → gate learning nothing) |
| `health/gate_std_L*` | > 0.05 | near-zero = gate is a constant, not modulating |
| `micro/unmasked_token_frac` | as configured (~0.5) | drift = loss-mask build is stale |
| `eval/gap` | grows over training | flat or negative late = TTT not contributing |
| `anomaly/nonfinite_count` | 0 | NaN/inf losses (guarded, skipped, alerted) |
| `gpu/mem_*`, `perf/*` | flat | OOM creep, throughput regressions |

---

## Key hyperparameters and their sensitivity

| knob | default | tune when |
|---|---|---|
| `eta` (TTTConfig) | `7e-2` | state saturates the clip → lower; too small applied delta → raise. Scale ~inversely with model width. |
| `chunk_size` | `400` | smaller = more updates per forward (finer resolution, more memory); at 8B use ≥400 to fit |
| `conv_kernel_size` | `8` | with `v_source="hidden_state"` (default), 4-8 is sweet spot. With `"embedding"`, larger (16-32). |
| `clip_tau` | `5.0` | tight bottleneck. Raising it usually hurts unless retrained; direction-only is often the useful regime. |
| `carried_decay` | `0.95` | 1.0 = pure accumulation (state grows unbounded on long sessions). 0.9 = ~10-item half-life. 0.8 = 5-item. Bounded plateau ≈ per_item_delta / (1 - decay). |
| `output_gate` | `True` | disable if the gate is stuck near 1 or 0 across training and adding param count isn't worth it |
| `carried_decay=0.9-0.95` + `clip_tau=5` | | the calibrated combo that works at 0.6B |
| Loss masking | disabled | enable when training on a domain-heavy corpus that has boilerplate; requires wikitext-103 reference build |

### Notes at scale (0.6B → 4B → 8B)

- **Per-param gradient shrinks with model size** (~6-11× smaller at 4B
  vs 0.6B in observed runs). Compensate by scaling **all three LRs
  ~10×** at bigger models.
- **`eta` bump is a trap.** It's the fast-weight update magnitude, not
  an LR. Scaling `eta` with LR causes state to saturate the clip and
  makes gradient collapse worse.
- **`state_ratio` scales with activation magnitudes** (bigger hidden =
  bigger per-chunk delta). At 4B, `state_ratio` around 5 is roughly
  equivalent to 0.6B's ~2. If it climbs to 20+, either lower `eta` or
  lower `carried_decay`.
- **Memory:** TTT scan materializes `deltas[B, k, d, d_ff]` and
  `cum[B, k, d, d_ff]`, each `(seq_len / chunk_size) × hidden × ffn ×
  2 bytes` per TTT layer. At 8B this is ~16 GB per layer at `k=164`.
  Use `TTT_LAYER_STRIDE=4` (halves TTT layer count) + `chunk_size=400`
  (fourths `k`) to fit.

---

## Known limitations (v1 scope)

- Papers truncate at `max_seq_len=16384` tokens; the long tail loses
  content.
- Batch size is fixed at 1 by the session loop (fast weight carry is
  per-stream; the code rejects B changes mid-session).
- Optimizer steps landing mid-session leave the carried state slightly
  stale vs the freshly updated slow weights. Standard TBPTT, benign.
- Chat mode is fundamentally OOD (LoRA/TTT trained on raw papers, not
  chat traces). Expect wandering / paper-flavored responses rather than
  chatbot behavior.
- Dataset cleaning leaks some LaTeX artifacts (mangled accents, macro
  fragments, author-list residue). Fix in the arxiv pipeline before a
  publication run.

---

## Troubleshooting

- **`sanity_check` assert fails.** The TTT wiring broke bit-exact
  identity at init. Check `W_target` is zero, LoRA's target regex still
  excludes TTT-layer `down_proj`, and gate bias init is still `-2.0`
  (near-closed sigmoid).
- **`RuntimeError: reference counts vocab_size ... != expected`.** The
  reference file was built against a different tokenizer/model.
  Rerun `modal run train_modal.py::build_reference_counts`.
- **`TTT checkpoint mismatch, saved N tensors, loaded M`.** The ckpt
  was trained with different `TTT_LAYER_STRIDE`/`LAYER_START`/
  `MODEL_SIZE` than the current config. Match the env vars used at
  training time, or re-save the ckpt.
- **`CUDA out of memory` at 8B.** See "Memory notes at scale". Bump
  `TTT_LAYER_STRIDE=4`, raise `chunk_size` to 400+, or drop
  `max_seq_len` to 8192.
- **Chat wanders / hallucinates.** The model is paper-trained, not
  chat-tuned. Use `--no-evolve` to isolate whether the carry is the
  source of drift. `/reset` between turns clears fast-weight state.
- **`grad/new` collapses to zero early.** New modules stuck in the
  dead basin. Options: freeze slow weights for the first ~100-200
  steps so all gradient flows through the new modules, or non-zero init
  `W_target` with small random (breaks `sanity_check` bit-exactness).
- **401/403 on dataset load.** Private repo without the `huggingface`
  secret attached to both app functions.
- **wandb charts empty.** The `wandb` Modal secret is missing;
  training continues console-only by design.
- **`Unclassified trainable parameter`.** A new trainable param
  appeared that `build_param_groups` (in `ttt_wiring.py`) doesn't
  recognize; classify it explicitly.
