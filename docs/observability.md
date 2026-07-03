# Observability

Full metric reference. Everything logged from
[`train_modal.py`](../train_modal.py) via
[`observability.py`](../observability.py).

## Telemetry backend

`Telemetry` in [`observability.py`](../observability.py) is a thin,
failure-proof wandb wrapper. Wandb project: `inplace-ttt`. Run name:
`f"{run_name}-{MMDD-HHMM}"`.

**Failure modes handled:**
- `wandb_enabled=False` in config → no-op.
- `WANDB_API_KEY` unset → prints a warning, continues console-only.
- `wandb.init()` throws → prints the error, continues console-only.
- `wandb.log()` throws → prints the error, keeps training.

Rule: telemetry can NEVER crash or stall a run.

## Two x-axes

- `train/step` — optimizer steps (post-grad-accum). Used for
  aggregates (`train/*`, `grad/*`, `session/*`, `health/*`,
  `perf/*`, `gpu/*`, `anomaly/*`, `eval/*`).
- `micro/step` — micro-batch steps (one per SessionItem forward).
  Used for `micro/*` (per-item signals).

Configure with:
```python
run.define_metric("train/step")
run.define_metric("micro/step")
run.define_metric("micro/*", step_metric="micro/step")
for ns in ("train/*", "grad/*", "session/*", "health/*",
           "perf/*", "gpu/*", "anomaly/*", "eval/*"):
    run.define_metric(ns, step_metric="train/step")
```

## Full metric reference

### `train/*` — every optimizer step

| metric | healthy | failure mode |
|---|---|---|
| `train/loss` | monotone decrease early, plateaus late | flat from step 0 → broken tap or bad LR / model init issue |
| `train/lr` | matches cosine schedule (warmup up, decay down) | flat / discontinuous → scheduler misconfigured |
| `train/grad_clip_ratio` | ~1.0 most of the time; occasional dips | persistently < 1 → clipping is eating updates (raise `max_grad_norm`) |
| `train/tokens_per_sec` | steady | drops → memory pressure / disk contention |
| `train/tokens_total` | monotone | (sanity) |

### `grad/*` — every optimizer step

Per-group gradient L2 norms (concatenated across params in the group).

| metric | healthy | failure mode |
|---|---|---|
| `grad/lora` | stable, bounded | dead (LoRA not learning) or exploding (LR too high) |
| `grad/wdown` | stable, bounded | dead → W_down frozen at init; exploding → LR too high |
| `grad/new` | initial spike, then decays to steady state | ~0 throughout while `grad/lora` healthy → **W_target stuck in dead basin** (see [failure-modes.md](failure-modes.md#dead-basin)) |

### `session/*` — every optimizer step (from `session_metrics`)

Per-layer `||eta · carried_delta||_F / ||W_down||_F` at end of the
current session.

| metric | healthy | failure mode |
|---|---|---|
| `session/state_ratio_L<i>` | grows within a session, bounded across | steady climb without bound → EMA disabled or too soft (`carried_decay ≥ 0.99`) |
| `session/state_ratio_mean` | scale-appropriate (0.6B: ~1-6; 4B: ~2-15) | > 20 → clip is saturating; direction is being wasted |

### `health/*` — every `param_log_every` (default 50) steps

Heavier metrics: parameter drift, gate stats.

| metric | healthy | failure mode |
|---|---|---|
| `health/wdown_drift_L<i>` | small (< 0.1) | TTT layers' W_down leaving the pretrained basin — LR too high |
| `health/w_target_L<i>` | rising then flattening | flat from step 0 → new modules stuck; growing without bound → unstable |
| `health/conv_L<i>` | small rise from init | flat → conv unmoved from pass-through init; big → conv taking over W_target's role |
| `health/gate_mean_L<i>` | in (0.1, 0.9) | stuck at ~0 (gate closed → TTT term suppressed) or ~1 (gate not modulating, equivalent to no gate) |
| `health/gate_std_L<i>` | > 0.05 | near-zero std → gate is a constant across positions, learning nothing |
| `health/lora_norm` | rising then flattening | flat → LoRA not learning; explosive → LR too high |

### `micro/*` — every micro-step (own axis)

Per-SessionItem signals.

| metric | healthy | failure mode |
|---|---|---|
| `micro/paper_loss` | reasonable range for the model size (bf16 arithmetic can spike briefly) | NaN/Inf → guarded, skipped, alerted |
| `micro/session_pos` | 0 to session_length-1 | (sanity) |
| `micro/n_tokens` | matches slice size | (sanity) |
| `micro/paper_loss` vs `micro/session_pos` | later positions have lower loss on average (carry is helping) | later positions consistently higher → carry is hurting during training (see [failure-modes.md](failure-modes.md#carry-hurts-training)) |
| `micro/unmasked_token_frac` | as configured (~0.5 with `loss_mask_keep_fraction=0.5`) | drift → loss-mask build is stale |

### `gpu/*` — every log step

From `gpu_stats()`:
- `gpu/mem_alloc_gb` — currently allocated by PyTorch
- `gpu/mem_reserved_gb` — reserved by allocator (may exceed alloc)
- `gpu/mem_peak_gb` — peak in the last window (reset periodically)

**Healthy:** flat with tolerable peak. Creeping upward → memory leak
(state.delta not being reset between sessions somewhere).

### `perf/*` — every log step

- `perf/step_sec` — wall-clock per optimizer step
- `perf/tokens_per_sec` — throughput

**Healthy:** flat. Drops → contention, IO, or a slower path being
triggered by the current data.

### `anomaly/*`

- `anomaly/nonfinite_count` — count of skipped micro-batches due to
  NaN/Inf loss. Healthy: 0. Nonzero triggers a wandb `WARN` alert.

### `eval/*` — every `eval_every` optimizer steps

From the in-loop eval (3 papers × 8 slices by default):
- `eval/carry_ppl` — token-weighted ppl with carry on
- `eval/fresh_ppl` — token-weighted ppl with carry off
- `eval/gap` = `fresh_ppl - carry_ppl`. Positive = carry is helping.
- `eval/state_ratio_mean` — mean across TTT layers at end of eval

**Healthy trajectory:** `gap` starts near zero, grows over training.
At 0.6B best runs, converges to +4-5.

## Metric → failure mode map (quick reference)

Ordered by "what should I check when things look wrong":

| symptom | likely cause | doc |
|---|---|---|
| `grad/new` ~= 0 from start | Dead basin, W_target isn't escaping | [failure-modes.md](failure-modes.md#dead-basin) |
| `session/state_ratio_*` climbing past 20 | Clip saturating, no useful magnitude reaching model | [failure-modes.md](failure-modes.md#state-saturation) |
| `train/loss` flat from step 0 | Broken tap, bad model init, or scheduler bug | check `sanity_check` first |
| `train/grad_clip_ratio` < 1 persistently | Grad clip too tight, killing training | raise `max_grad_norm` |
| `eval/gap` stays flat at 0 late in training | Mechanism trained but not contributing | try longer run, non-zero W_target init |
| `health/wdown_drift_L*` > 0.5 | W_down being trained too aggressively | lower `lr_wdown` |
| `health/gate_mean_*` stuck at 0 or 1 | Gate not learning to modulate | try `gate_reg_weight > 0` or disable gate |
| `anomaly/nonfinite_count > 0` | NaN in loss — guard fired | check for bad tokenization / data corruption |
| `gpu/mem_*` creeping up | Memory leak somewhere | audit session state resets |
| `perf/tokens_per_sec` dropping | Throughput regression | check for oversized items, disk contention |

## Custom charts to build in wandb

The auto-defined metrics are the raw signal. Useful derived charts:

1. **`eval/gap` over `train/step`** — the headline "is TTT helping"
   curve. Line chart with a horizontal reference at 0.
2. **`grad/new / grad/lora` ratio** — should be roughly stable after
   the initial spike. Big divergence means one group is being trained
   much harder than the other.
3. **`session/state_ratio_mean` vs `clip_tau`** — how close to
   saturation the carry is running.
4. **Per-layer `session/state_ratio_L<i>`** — heatmap across layers
   and steps. Uneven layers → some TTT layers doing most of the work.

## Related docs

- [training.md](training.md) — when each metric is logged
- [failure-modes.md](failure-modes.md) — diagnostic patterns in the metrics
