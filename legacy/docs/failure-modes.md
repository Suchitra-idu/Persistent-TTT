# Failure Modes

Observed failure patterns, root causes, and fixes.

## Dead basin

**Symptom:** `grad/new` collapses to ~0 within the first ~50 steps
and stays flat. `health/w_target_L*` stays close to init magnitude.
`eval/gap` stays at 0 throughout training.

**Cause:** `W_target = 0` at init. This makes V = 0, delta = 0, and
therefore no signal reaches the model. Gradient to W_target is
proportional to activation magnitudes, which are moderate — so there's
a chicken-and-egg problem: W_target needs to move to be useful, but
gradient signal is small until it moves.

At 0.6B, activation scale is enough that gradient can spike early
(see 0.6B's `grad/new` peak at ~2500 in the first few steps). W_target
escapes zero in ~10 steps.

At 4B and 8B, per-parameter gradients are ~7-10× smaller (more params
sharing the loss reduction budget), and the same activation-driven
spike is proportionally weaker. W_target may fail to escape at all.

**Fixes** (in order of ease):

1. **Bump LRs proportionally.** All three groups need ~10× LR at 4B
   and 8B to compensate for gradient dilution. See [scaling.md](scaling.md).
   This alone may not fix the basin — LR scales update size, not
   gradient magnitude.

2. **Non-zero W_target init.** Break out of the basin geometrically:
   ```python
   # In inplace_ttt.py __init__:
   w = torch.empty(hidden_size, hidden_size)
   nn.init.normal_(w, std=1e-4)   # or 5e-5
   self.w_target = nn.Parameter(w)
   ```
   Tradeoff: `sanity_check` no longer passes (bit-exact identity to
   base Qwen3 is lost at step 0). This is a diagnostic property, not
   a training requirement.

3. **Freeze slow weights for the first N steps.** All gradient flows
   into new modules, concentrating the initial spike energy. Requires
   a small train loop change (skip optimizer step on `lora` + `wdown`
   groups for the warmup phase, then unfreeze).

## State saturation

**Symptom:** `session/state_ratio_*` grows past ~10 and keeps
climbing. Eval `gap` may still be positive but is bounded — usually
in a range of +2 to +5 regardless of how much more you train.

**Cause:** clip is active on nearly every applied delta. `clip_tau=5`
caps `||eta · cum||_F` at 5, regardless of stored magnitude. Once
stored state ≫ clip_tau, only direction survives. Gradient to
W_target for "make delta bigger" is orthogonal-projected away —
model can only refine direction, and refinement slows as W_target
grows.

**Fixes:**

1. **Enable `carried_decay < 1.0`.** Bounds stored magnitude:
   ```
   plateau ≈ per_item_delta / (1 - carried_decay)
   ```
   `0.9` gives ~10-item half-life, plateau at ~10× per-item.
   `0.95` gives ~20-item, plateau at ~20×.

2. **Lower `eta`.** Reduces per-chunk delta magnitude → state grows
   slower → clip binds later. `eta=3e-2` instead of `7e-2` halves
   growth rate.

3. **Do NOT raise `clip_tau`.** Experimentally verified at 0.6B: the
   model calibrates its direction quality to the clip during training.
   Raising `clip_tau` at inference exposes downstream layers to
   unclipped magnitudes they weren't trained for → catastrophic ppl
   collapse. Training with a bigger `clip_tau` didn't help either —
   state saturation just happens at bigger magnitudes without
   improving direction quality.

## Carry hurts training

**Symptom:** later-position items in a session have HIGHER loss than
earlier ones (opposite of what you want). `micro/paper_loss` vs
`micro/session_pos` shows a positive slope.

**Cause:** the carry from prior items is corrupting the residual
stream. Common when `session_training=True` is enabled early in
training when W_target isn't in a useful direction yet — the
"carry" is essentially structured noise that compounds.

**Fixes:**

1. **Wait it out.** After ~50-100 steps, W_target should have moved
   into a useful direction and later positions will start being
   cheaper.

2. **Increase `carried_decay`** (closer to 1.0, but not exactly 1.0)
   so early-training noise doesn't compound as fast.

3. **Enable `session_training` only after a warmup period.** Train
   with `session_mode=False` for the first N steps to let W_target
   escape the basin without noisy carry, then enable session mode.
   Requires a train loop change.

## Cross-paper carry cliff

**Symptom:** `holdout_eval` with 5+ papers shows carry ppl exploding
after position ~3 (values like 1000+). Fresh ppl stays healthy.
Within-paper (`single_paper_eval`) is fine.

**Cause:** `session_training=False` at training time → the model
literally never saw non-zero carry during training. At eval, the
first paper's carry is now "input the model hasn't been trained for."
Slow weights don't know how to absorb non-zero carry input → the
whole model destabilizes.

**Fix:** train with `--session 1` so the model sees accumulated carry.
Observed at 0.6B: this alone eliminated the cliff, and gap stayed
positive across 8-paper sessions.

## LoRA overfitting

**Symptom:** training loss keeps dropping but `eval/carry_ppl` and
`eval/fresh_ppl` both climb.

**Cause:** overfitting the training slice. Common on small `--limit-docs`
runs.

**Fix:** more data. Bump `--limit-docs` or use the full training
split. Optionally add `lora_dropout > 0`.

## Chat wandering / hallucinated scaffolding

**Symptom:** in the chat REPL, responses ignore the current user
message and continue whatever narrative the model was on. Or the model
emits literal `you>` / `bot>` tokens in its output.

**Cause 1:** the fast-weight carry from prior turns dominates the
current turn's input. Common at high `pending_tokens` + high
`state_ratio`. The mechanism was working as designed — the carry has
compressed the prior context — but the model wasn't trained for chat,
so it doesn't know to attend to the new user message over the carry.

**Cause 2:** LoRA + TTT trained on raw paper text, never on chat
templates. The chat template is OOD for the trained model, so it can
generate tokens that look like chat scaffolding.

**Fixes:**

1. `/reset` between turns to drop the fast-weight state.
2. `--no-evolve` — freeze fast weights, keep only the slow model.
3. Long-term: second-stage chat-format fine-tune. Even 500-1000
   hand-constructed Q&A pairs over training papers would meaningfully
   improve chat behavior.

## `sanity_check` fails

**Symptom:** `modal run train_modal.py::sanity_check` prints a nonzero
max logit diff and the assert fires.

**Cause:** something in the TTT patch is not bit-exact identity at
init. Common breakages:

- `W_target` is not exactly zero (e.g., someone added a non-zero init)
- Output gate bias changed from `-2.0` such that `sigmoid(bias) *
  ttt_out` is nonzero even when `ttt_out = 0`. Actually this is fine
  because ttt_out is zero when W_target is zero. But if you change
  init logic and W_target ≠ 0, gate open would leak.
- LoRA regex includes a TTT-layer `down_proj` (double-tap: LoRA on
  the same weight that's used as W0). Check
  `build_lora_target_regex` in `ttt_wiring.py`.
- `target_conv` init is not pass-through (last position 1, rest 0).

**Fix:** revert the offending init/change, or explicitly account for
non-identity init.

## OOM at scale

**Symptom:** `CUDA out of memory` early in training. Common at 8B.

**Cause:** TTT scan materializes `deltas[B, k, d, d_ff]` and
`cum[B, k, d, d_ff]` per layer. At 8B (`d=4096`, `d_ff=12288`,
`seq=16384`, `chunk=100`), that's ~16 GB per layer × 18 layers × 2 =
576 GB nominal.

Gradient checkpointing lets you recompute one segment at a time, so
only ~30-35 GB is live at any moment during backward. But at 8B this
still bumps the ceiling.

**Fixes** (in order):

1. `TTT_LAYER_STRIDE=4` — halves TTT layer count.
2. Bump `chunk_size` in config (100 → 400+). `k = seq_len / chunk_size`
   drops 4× → deltas/cum drop 4×.
3. Drop `max_seq_len` (16384 → 8192 or 4096). Same effect as #2.
4. If still OOM: reduce LAYER_STRIDE further or drop to 4B.

Set the env var `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if
the error message mentions large reserved-but-unallocated memory
(fragmentation).

## Reference counts vocab mismatch

**Symptom:**
```
RuntimeError: reference counts vocab_size 151669 != expected 151936;
rebuild the reference after a tokenizer change via build_reference_counts
```

**Cause:** the reference was built against a different tokenizer,
usually because `TTT_MODEL_SIZE` changed.

**Fix:**
```
modal run train_modal.py::build_reference_counts
```

## TTT ckpt mismatch

**Symptom:**
```
RuntimeError: TTT checkpoint mismatch, saved 162 tensors, loaded 97.
Check layer indices match the checkpoint.
```

**Cause:** the ckpt was saved with a different `TTT_LAYER_STRIDE`,
`TTT_LAYER_START`, `TTT_MODEL_SIZE`, or `output_gate` setting than
the current config.

**Fix:** match the training-time env vars at inference time:
```
TTT_MODEL_SIZE=0.6B TTT_LAYER_STRIDE=2 modal run infer_modal.py::...
```

Or retrain with the new config.

## Snapshot silent misapplication

**Symptom:** loading a `sessions/<name>.pt` snapshot from an older
checkpoint gives degraded outputs. No error raised.

**Cause:** a fast-weight snapshot is a delta against `W_down`. If
`W_down` has moved (further training), the snapshot no longer aligns
with the current `W0`.

**Fix:** re-snapshot on the current slow weights. Snapshots are only
valid for the exact ckpt they were created under. There's no explicit
guard — treat this as a discipline issue.

## `grad/new` spike then collapse (early)

**Symptom:** `grad/new` spikes to a large value (0.6B: ~2500; 4B:
~500) at step 5-10, then decays to a small steady state within 30-50
steps.

**Cause:** normal escape from the dead basin. The spike is the
initial gradient signal driving W_target from zero to a useful
direction. Once escaped, gradient collapses because the loss can't
improve further from magnitude alone — only direction refinement.

**Not a failure — this is expected.** Look for the spike happening
at all; if it doesn't, you're stuck in the basin. See "Dead basin"
above.

## Related docs

- [observability.md](observability.md) — how to spot these in wandb
- [scaling.md](scaling.md) — scale-specific failure patterns
- [mechanism.md](mechanism.md) — the mechanics behind the modes
