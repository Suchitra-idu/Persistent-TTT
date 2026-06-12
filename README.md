# In-Place TTT with Cross-Session Persistent Domain Memory

Continual test-time adaptation of Qwen3-8B via In-Place fast-weight TTT
on streaming arXiv ML papers. Builds on "In-Place Test-Time Training"
(ByteDance Seed, arXiv 2604.06169). The research contribution is
replacing the paper's per-document fast weight resets with
session-level persistence, so domain memory written into the weights by
one paper survives into the next paper and across sessions, with no
shared context window.

Everything runs on Modal. Nothing touches the local machine except
`modal run` commands and the local math test.

---

## Folder structure

```
inplace-ttt/
├── README.md
├── ttt_config.py        all constants and hyperparameters (edit here first)
├── inplace_ttt.py       the TTT mechanism (pure PyTorch, no Modal)
├── model_setup.py       model assembly shared by train and inference
├── data_utils.py        dataset loading and the holdout split
├── observability.py     wandb telemetry and metric collectors
├── train_utils.py       pure training logic (session schedule, grad norms)
├── train_modal.py       Modal app, training + sanity check
├── infer_modal.py       Modal app, inference + evaluation
├── pipeline/          arXiv data pipeline (separate, already run)
└── tests/               local CPU suite, no Modal/GPU/downloads (~3s)
    ├── conftest.py            shared tiny-module fixtures
    ├── test_scan_math.py      scan vs sequential reference, both modes
    ├── test_mechanism.py      identity, causality, stream/scan, evolve, clip
    ├── test_wiring.py         LoRA regex, param groups, checkpoint I/O
    ├── test_session.py        carry lifecycle, staging idempotence, schedule
    └── test_observability.py  telemetry safety, metric collectors
```

Run the suite with `python -m pytest tests/ -q` (needs only torch,
numpy, pytest). Run it after ANY change to inplace_ttt.py,
train_utils.py, or observability.py, and before every training launch.

The Python modules MUST stay flat at the project root. Modal ships them
into containers via `image.add_local_python_source("ttt_config", ...)`,
which imports by module name from the directory where `modal run`
executes. Moving them into `src/` or a package breaks both apps.

### Who imports whom

```
train_modal.py ─┬─> model_setup.py ──> inplace_ttt.py ──> ttt_config.py
                ├─> data_utils.py  ──────────────────────> ttt_config.py
                └─> observability.py
infer_modal.py ─┬─> model_setup.py, data_utils.py, inplace_ttt.py
tests/          └─> inplace_ttt.py, ttt_config.py
```

`inplace_ttt.py` has zero Modal dependencies on purpose, so the
mechanism is unit-testable locally and reusable outside Modal.

---

## What each file owns

**ttt_config.py.** Single source of truth. `TTTConfig` holds the
mechanism (TTT layer indices, chunk size, eta, conv kernel, clipping).
`TrainConfig` holds the outer loop (learning rates per parameter group,
LoRA shape, session sizes, wandb settings). Module-level constants hold
Modal volume names, the HF dataset repo id, and the holdout size.

**inplace_ttt.py.** The mechanism. `InPlaceTTTMLP` replaces the gated
MLP on the TTT layers and implements both execution paths, a parallel
chunk scan for training and whole-sequence eval, and a stateful stream
for autoregressive generation. Also owns the model-tree helpers
(`patch_model_with_ttt`, `set_ttt_evolve`, session lifecycle functions,
fast weight export/import, LoRA config builder, parameter grouping,
TTT checkpoint save/load).

**model_setup.py.** One function, `build_model`, that assembles
base model, TTT patch, LoRA wrap, grad unfreezing, and checkpoint
loading in the single correct order. Train and inference both call it,
so they can never assemble the model differently.

**data_utils.py.** `open_dataset` pulls the parquet dataset from the
HF Hub (cached on a Modal volume after the first run). `split_holdout`
defines the one train/eval boundary, the newest `HOLDOUT_LAST_N` papers
never enter training and are the contamination-free pool for session
evaluation.

**train_utils.py.** Pure functions used by the training loop, the
session scheduler and per-group gradient norms. Kept Modal-free so the
schedule, which shapes every run, is unit-tested.

**observability.py.** `Telemetry` wraps wandb and can never crash or
stall a run (missing key or network failure degrades to console). The
metric collectors map every known failure mode of this project to a
chart, see the table below.

**train_modal.py.** The training app. Session-scheduled loop, three
optimizer parameter groups, 8-bit AdamW, gradient checkpointing,
nonfinite-loss guard, checkpointing to a Modal volume, full telemetry.
Also `sanity_check`, the identity test that must pass before anything.

**infer_modal.py.** The inference app. A warm `TTTInference` class with
perplexity scoring, session perplexity (fast weights persisting across
papers), generation with streaming fast weight updates, fast weight
snapshot save/load for cross-session persistence, and holdout
evaluation entrypoints.

---

## The mechanism in five lines

For TTT layers, with activations `Z = silu(gate(H)) * up(H)` and
LM-aligned targets `V = CausalConv1D(X0) @ W_target` chunked into
chunks of size C

```
apply:   O_[i] = Z_[i] @ (W_down + eta * S_i)^T
update:  S_{i+1} = S_i + V_[i]^T @ Z_[i] / C        (S_0 = 0)
```

Chunk i is processed with updates from strictly earlier chunks. Session
mode changes exactly one thing, `S_0` starts from the previous paper's
final state instead of zero, and gradients never cross the paper
boundary (truncated BPTT).

Trainable parameters, three groups with separate learning rates

| group  | what                                   | LR    | why |
|--------|----------------------------------------|-------|-----|
| lora   | attention + gate/up (all layers), down_proj (non-TTT layers) | 1e-4 | pretrained, adapted via LoRA r=32 |
| wdown  | down_proj on the 6 TTT layers (full)   | 2e-5  | the fast weight initial state, move gently |
| new    | W_target + Conv1D (full)               | 2e-4  | fresh, zero/passthrough init |

W_target is zero-initialized so the whole model is exactly base
Qwen3-8B at step 0. LoRA never touches down_proj on TTT layers
(regex-enforced), because that weight is used functionally as the fast
weight.

---

## Setup

1. Install Modal locally and authenticate (`pip install modal`,
   `modal setup`).
2. Edit `ttt_config.py`
   * `DATASET_SOURCE` to your HF dataset repo id.
   * Volume names if you want different ones (both auto-create).
3. Create secrets
   ```
   modal secret create wandb WANDB_API_KEY=...
   ```
   If the dataset repo is private, also
   ```
   modal secret create huggingface HF_TOKEN=hf_...
   ```
   and append `modal.Secret.from_name("huggingface")` to `SECRETS` in
   both app files.
4. Run the test suite once
   ```
   pip install pytest
   python -m pytest tests/ -q
   ```

---

## Run protocol (in this order, no skipping)

**0. Wiring check.** Must print a max logit diff near zero and pass the
assert. A failure means the model is broken, do not train.
```
modal run train_modal.py::sanity_check
```

**1. Overfit smoke test.** Loss must fall fast. Watch `grad/new` in
wandb, it must be nonzero from early on.
```
modal run --detach train_modal.py::train --limit-docs 100 --num-epochs 5
```

**2. Real run.**
```
modal run --detach train_modal.py::train
```

**3. Evaluate.** Contamination-free, zero local files.
```
modal run infer_modal.py::holdout_eval --n-papers 5 --ckpt step_600
```
Per-paper perplexity with fast weight carry vs without. The carry vs
fresh gap on papers 2..n is the cross-session memory signal.

Other evaluation commands
```
# single-text TTT on/off perplexity gap
modal run infer_modal.py::compare_ppl --text-path paper.txt --ckpt step_600

# generation, fast weights evolving over prompt + output
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_600

# same with evolution frozen
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_600 --no-evolve
```

---

## The evolve switch

`evolve=True` lets fast weights update chunk by chunk as text streams
in. `evolve=False` freezes evolution; previously accumulated or
imported state is still applied, and with no state loaded the model
behaves as plain Qwen3-8B + LoRA. This separates "stop learning" from
"forget everything" (`reset_fast_weights` does the latter), and gives
the eta-ablation in one flag. Programmatic access is
`set_ttt_evolve(model, bool)`; every inference method takes `evolve`.

Cross-session persistence primitives, `export_fast_weights` /
`import_fast_weights` snapshot and restore the accumulated deltas;
`TTTInference.save_session(name)` persists them to the checkpoint
volume under `sessions/`.

---

## Checkpoints layout

```
ttt-checkpoints volume
└── ttt-v1/                      (TrainConfig.run_name)
    ├── step_200/
    │   ├── adapter/             PEFT LoRA adapter (save_pretrained)
    │   └── ttt_params.pt        W_down (TTT layers), W_target, Conv1D
    ├── step_400/ ...
    └── sessions/
        └── <name>.pt            persisted fast weight snapshots
```

A fast weight snapshot is only valid for the exact slow weights it was
created under. Loading a snapshot after further training silently
applies a delta against a W0 that no longer exists.

---

## Observability, metric to failure-mode map

Every optimizer step logs to wandb (project `inplace-ttt`). The point
of each chart

| metric | healthy | failure it exposes |
|---|---|---|
| `grad/new` | nonzero, growing early | ~0 while `grad/lora` healthy means the X0 tap or target computation is broken |
| `grad/lora`, `grad/wdown` | stable | dead or exploding groups |
| `session/state_ratio_*` | small, bounded | steady climb past ~1e-1 means unbounded fast weight growth, add forgetting |
| `health/wdown_drift_L*` | small | TTT layers' W_down leaving the pretrained basin |
| `health/w_target_L*`, `health/conv_L*` | rising then flattening | flat from the start means new components not learning |
| `train/grad_clip_ratio` | ~1.0 | persistently below 1 means clipping is eating updates |
| `anomaly/nonfinite_count` | 0 | NaN/inf losses (guarded, skipped, alerted) |
| `micro/paper_loss` vs `micro/session_pos` | later positions cheaper | session carry not helping during training |
| `gpu/mem_*`, `perf/*` | flat | OOM creep, throughput regressions |

Heavier `health/*` metrics log every `param_log_every` (50) steps.
Per-paper `micro/*` metrics use their own x-axis.

---

## Known unverified items (check against the official repo before the real run)

* **eta** (inner learning rate), default 1e-3 with per-chunk
  normalization. Tune on the overfit run.
* **Conv1D kernel size**, default 4.
* **Frobenius clipping formula** (paper appendix, tau = 1e-5). The
  literal absolute-cap reading would zero the mechanism at inference,
  so clipping is DISABLED by default (`TTTConfig.clip_enabled`).
  Verify the exact semantics before enabling.
* **Exact init scheme** for Conv1D / W_target. Current scheme
  (W_target zero, conv passthrough) guarantees exact identity at step 0
  and is verified by `sanity_check`, but may differ from the paper.

## Known limitations (deliberate v1 scope)

* Papers truncate at 16,384 tokens; the ~75k-token tail loses content.
* Batch size is fixed at 1 by the session loop (fast weight carry is
  per-stream by design; the code rejects B changes mid-session).
* Optimizer steps landing mid-session leave the carried state slightly
  stale vs the freshly updated slow weights. Standard TBPTT, benign.
* No forgetting mechanism yet. Intentional, the
  `session/state_ratio_*` curves from the first runs decide which one
  (decay, norm cap, or L2 weight normalization with Muon).
* No general-data (PG19) mixing yet, planned to counter domain drift.
* Dataset cleaning leaks some LaTeX artifacts (mangled accents, macro
  fragments, author-list residue). Fix in pipeline.py before the real
  run.

## Troubleshooting

* `sanity_check` assert fails. The TTT wiring changed the model at
  init. Check W_target is zero-init and LoRA's target regex still
  excludes TTT-layer down_proj.
* `RuntimeError: Set DATASET_SOURCE`. Edit ttt_config.py.
* 401/403 on dataset load. Private repo without the huggingface secret
  attached.
* `Unclassified trainable parameter`. A new trainable param appeared
  that `build_param_groups` does not recognize; classify it explicitly.
* `TTT checkpoint mismatch` on load. The checkpoint was trained with
  different `TTT_LAYER_INDICES` than the current config.
* wandb charts empty. The `wandb` Modal secret is missing; training
  continues console-only by design.
