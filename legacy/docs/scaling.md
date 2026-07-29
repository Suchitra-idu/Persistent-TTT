# Scaling Considerations (0.6B → 1.7B → 4B → 8B)

Empirical guide to what changes as you scale the base model.

## Model-size table

Qwen3 dense sizes used with this code:

| size | layers | hidden | ffn | vocab (padded) | TTT layers @ stride 2 |
|---|---|---|---|---|---|
| 0.6B | 28 | 1024 | 3072 | 151936 | 14 |
| 1.7B | 28 | 2048 | 6144 | 151936 | 14 |
| 4B | 36 | 2560 | 9728 | 151936 | 18 |
| 8B | 36 | 4096 | 12288 | 151936 | 18 |

Set with `TTT_MODEL_SIZE=0.6B` etc. Env vars are read at
`ttt_config.py` import time.

## Three scaling effects that bite

### 1. Per-parameter gradient dilution

Gradient magnitude at model scale, empirically observed:

| group | 0.6B | 4B | ratio |
|---|---|---|---|
| `lora` | ~1.0 / √3.9M ≈ 5e-4 | ~0.2 / √19M ≈ 4.6e-5 | ~11× smaller |
| `wdown` | ~2.5 / √35M ≈ 4.2e-4 | ~1.5 / √500M ≈ 6.7e-5 | ~6× smaller |
| `new` | ~0.4 / √14M ≈ 1e-4 | ~0.09 / √59M ≈ 1.2e-5 | ~8× smaller |

**Compensation:** bump all three LRs ~10× at 4B and 8B. This matches
per-parameter update size to what was working at 0.6B. Slower
convergence remains a problem because 4B has more parameters to move —
plan for 3-5× more training steps than 0.6B needed.

**Do NOT bump `eta`** as part of the "10× everything." `eta` is the
inner-loop update magnitude, not an LR. Bumping it makes state
saturate the clip early → gradient collapse. See below.

### 2. State magnitude scales with activation scale

At bigger models, activations `Z` and `V` are naturally bigger
(larger `hidden`, `ffn`). Per-chunk delta `V^T Z / C` scales roughly
with `hidden × sqrt(ffn)` or similar. Same `eta` produces bigger
`||eta · delta||`.

Empirical: at 0.6B, `state_ratio` sits around 1-6 during productive
training. At 4B with same `eta=7e-2`, it's 2-15. At 8B, 5-20.

**Consequence:** `clip_tau=5` is more constantly-binding at bigger
scales. The "direction-only" regime is where most of your applied
signal lives.

**Compensation options:**
- Lower `eta` at bigger scales (`7e-2` → `3e-2` for 4B, `2e-2` for 8B).
  This directly halves per-chunk delta magnitude.
- Lower `carried_decay` (0.95 → 0.9 or 0.85). Bounds stored magnitude
  more tightly.
- Do NOT raise `clip_tau` — the model calibrates its expectations to
  it during training. Change requires retraining, and empirically
  doesn't help.

### 3. LoRA covers more ground at scale

`fresh_ppl` from LoRA-only after equivalent training:
- 0.6B: BASE 37 → LoRA-only 31 (LoRA saves 6 ppl)
- 4B (estimated): BASE ~13 → LoRA-only ~11 (LoRA saves 2 ppl)

The base model at 4B is much stronger on ML papers to start with, and
LoRA quickly captures whatever domain adaptation is easy. That means
TTT has **less absolute headroom** at larger models.

**Consequence:** absolute TTT gap likely shrinks with model size.

**Relative gap** (gap / fresh_ppl) may be more stable — at 0.6B best,
it's ~15%. If 4B lands around 8-15% relative, the mechanism is
scaling proportionally.

**Where TTT wins bigger at scale:**
- Out-of-distribution data (private, novel domain) where LoRA can't
  adapt as broadly
- Very long sessions where the fast-weight capacity (bigger W_target)
  actually stores more useful info
- Specialized notation / per-paper novelty

## Memory considerations

TTT scan materializes `deltas[B, k, d, d_ff]` and `cum[B, k, d, d_ff]`.

`k = seq_len / chunk_size`. Per layer: `k × d × d_ff × 2 bytes` (bf16).

| model | k @ seq=16384, chunk=100 | k × d × d_ff × 2 bytes |
|---|---|---|
| 0.6B | 164 | 164 × 1024 × 3072 × 2 = **1.0 GB** |
| 1.7B | 164 | 164 × 2048 × 6144 × 2 = **4.1 GB** |
| 4B | 164 | 164 × 2560 × 9728 × 2 = **8.2 GB** |
| 8B | 164 | 164 × 4096 × 12288 × 2 = **16 GB** |

Per layer. Times 2 (deltas + cum). Times num_ttt_layers.

At 8B with `LAYER_STRIDE=2` (18 layers), that's `16 × 2 × 18 = 576 GB`
nominal. Gradient checkpointing lets one layer's activations live at
a time, so realistic peak is ~30-40 GB just for these tensors during
backward. Add ~30 GB static (params + optim states + backbone
activations) → doesn't fit in 80 GB.

**Fixes at 8B:**
- `TTT_LAYER_STRIDE=4` → 9 TTT layers → half the memory.
- `chunk_size=400` in config → `k=41` → 4× less memory.
- Combined → 8× reduction, comfortable fit.

## Recommended config per scale

Starting points that have worked or should work:

### 0.6B (proven, +4.5 gap achieved)

```
TTT_MODEL_SIZE=0.6B, TTT_LAYER_STRIDE=2

eta=7e-2, chunk_size=100, conv_kernel_size=8
clip_tau=5.0, carried_decay=0.90-0.95

lr_lora=1e-5, lr_wdown=3e-5, lr_new_modules=2e-5

modal run --detach train_modal.py::train \
    --limit-docs 2000 --num-epochs 1 --session 1 --single-paper 1
```

### 1.7B (should work with minor tweaks)

```
TTT_MODEL_SIZE=1.7B, TTT_LAYER_STRIDE=2

eta=4e-2 or 5e-2 (compensating for bigger activations)
chunk_size=100-200
carried_decay=0.92

LRs ~3× bumped: lr_lora=3e-5, lr_wdown=9e-5, lr_new_modules=6e-5

modal run --detach train_modal.py::train \
    --limit-docs 2000 --num-epochs 1 --session 1 --single-paper 1
```

### 4B (memory borderline)

```
TTT_MODEL_SIZE=4B, TTT_LAYER_STRIDE=2 (may need 3 for memory)

eta=2e-2 or 3e-2
chunk_size=200-400
carried_decay=0.88-0.92

LRs ~10× bumped: lr_lora=1e-4, lr_wdown=3e-4, lr_new_modules=2e-4

modal run --detach train_modal.py::train \
    --limit-docs 3000 --num-epochs 1 --session 1 --single-paper 1
```

### 8B (memory-limited)

```
TTT_MODEL_SIZE=8B, TTT_LAYER_STRIDE=4

eta=2e-2, chunk_size=400
carried_decay=0.88

LRs ~10× bumped: lr_lora=1e-4, lr_wdown=3e-4, lr_new_modules=2e-4

modal run --detach train_modal.py::train \
    --limit-docs 3000 --num-epochs 1 --session 1 --single-paper 1
```

## What experiments to run when scaling up

1. **Sanity check first.** Model-size change breaks `sanity_check` if
   any wiring is size-dependent. Run it:
   ```
   TTT_MODEL_SIZE=<size> modal run train_modal.py::sanity_check
   ```
2. **Overfit smoke test** with 100 docs, 5 epochs. Loss should fall,
   `grad/new` should spike then decay, `state/W0` should grow smoothly
   (not jump to giant values or stay flat).
3. **In-loop eval on step_100.** If `eval/gap` is flat or negative,
   something is off with `eta` or `carried_decay` — don't proceed
   until this shows a positive signal.
4. **Full run.** Only after (3) looks healthy.

## Empirical benchmarks (this codebase)

At 0.6B, step 400+, decay=0.95, session_training=True,
single_paper_sessions=True:
- `holdout_eval --n-papers 5`: per-paper gap +3 to +7, avg +4.5
- `single_paper_eval --n-slices 18`: per-paper gap +3.5 to +4.5, positive at every slice
- State plateau: ~1.7 (with decay=0.9); ~5-7 (with decay=1.0 pre-decay era)

At 4B, step 100 with eta=7e-2 and 10× LRs:
- `eval/gap ≈ +0.03` (nearly flat)
- `state/W0 ≈ 20-50` (heavy clip saturation)

At 4B, step 100 with `eta=7e-2` and standard LRs (not bumped):
- `grad/new` dilution → basin escape barely happens
- Recommendation: bump LRs, retrain

## Related docs

- [config.md](config.md) — all knobs
- [failure-modes.md](failure-modes.md) — dead basin, saturation, OOM
- [observability.md](observability.md) — which metrics to watch during
  a scale-up run
