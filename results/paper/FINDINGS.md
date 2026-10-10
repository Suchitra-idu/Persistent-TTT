# Phase 2 findings

Hand-written interpretation of the generated tables and figures in this folder.
Every number below is a perplexity gain (Δppl, positive = lower perplexity) on
the same 106 held-out documents, as a paired bootstrap 95% CI. `*` = the CI
excludes zero. Base model: Qwen3-0.6B.

Runs compared, all at steps 200 / 400 / 600:

| label | checkpoint | what it is |
|---|---|---|
| lora-only-matched | `lora-only-matched` | LoRA alone, same flags/LR/seed/docs as the TTT runs, no TTT layers |
| hebbian | `ttt-v1.1` | TTT layers with the Hebbian update (`update_rule=hebbian`) |
| delta_chunk | `ttt-v1.1-clean-deltachunk` | TTT layers with the chunked delta rule |

## 1. TTT training beats a matched LoRA baseline at every step

`tables/T1_main_gaps` (`vs_matched_lora`), `figures/F1_gains_vs_step` left panel.

| step | hebbian vs. LoRA | delta_chunk vs. LoRA |
|---|---|---|
| 200 | +0.073 * (91% of docs) | +0.232 * (100%) |
| 400 | +0.080 * (99%) | +0.191 * (99%) |
| 600 | +0.106 * (100%) | +0.203 * (100%) |

The lead holds on every source (`T2_per_source`). It is largest on C4, Wikipedia and Book,
and smallest on Github and ArXiv.

**Caveat, to state as a limitation:** a TTT run trains the full `W_down` of 14
layers on top of LoRA, so it has far more trainable parameters than the
control. Most of the lead is that extra capacity. The `training` rows of T1 show it
(base → TTT path switched off): at step 600 that gap alone is +1.928 for hebbian and
+1.992 for delta_chunk, against +1.859 for LoRA. The part that is clean evidence for
the mechanism is the `within` measure below.

## 2. Fast-weight updates help within a document, and the update rule matters

`within` = the same trained model with fast-weight updates on vs. off, inside one slice.

| step | hebbian | delta_chunk |
|---|---|---|
| 200 | +0.016 * | +0.144 * |
| 400 | +0.014 * | +0.059 * |
| 600 | +0.034 * | +0.058 * |

- Delta_chunk is larger than hebbian at every step: 9× at step 200, 4× at 400, 1.7× at 600.
- Delta_chunk's gain is largest early and settles around +0.06. This is consistent with the
  slow weights absorbing part of what the fast weights did at first. That is an
  interpretation, not something measured.
- The gain does not build up along a document (`T4_by_position`, `F2`). It is
  already present in slice 0, dips in the middle slices, and has no upward trend. TTT here adapts
  locally, inside each slice, rather than accumulating knowledge of the document.

**The hebbian output gate is almost shut.** Its median is 0.002 at every chain
position (`T5_state_and_gate`), so the TTT term reaches the output at about 0.2%
strength. Hebbian's small within-document gain is what leaks through that gate.
Delta_chunk's gate stays open (about 0.78). Training learned to suppress the hebbian
fast weights and to use the delta-rule ones.

## 3. Carrying state across documents does nothing, at every step, for both rules

`T1` `cross_doc`, `T3_persistence`, `F1` right panel, `F3`.

| measure (step 600) | hebbian | delta_chunk |
|---|---|---|
| cross_doc: carry vs. reset at document boundaries | −0.004 [−0.015, +0.006] | −0.002 [−0.006, +0.002] |
| repeat_9_vs_1: the same document re-read 9 times vs. once | +0.000 [−0.001, +0.001] | +0.000 [−0.001, +0.001] |
| same_vs_other_source: primed by the same vs. another source | +0.010 * | +0.002 |

`cross_doc` sits inside ±0.008 with a CI that includes zero at 200, 400 and 600,
for both rules. Three times more training does not move it, which answers
"it's just undertrained" within this compute budget.

Carrying state *within* a document (from earlier slices to later ones) does help a little with
delta_chunk: `between_slices` is +0.012 * at step 600, about +0.015 per
slice from slice 1 on (`F2` right panel). So persistence helps over a short range and
does not carry across a document boundary.

Two per-source `cross_doc` estimates are significantly negative (hebbian Book,
delta_chunk Wikipedia). Each rests on 4 documents, among 12 per-source tests;
treat them as noise unless they replicate.

## 4. Why: the read-time clip is saturated, so only the direction of the state reaches the output

Every fast-weight read is Frobenius-clipped so that ‖η·S‖ ≤ τ, with τ = 20
(`ttt/core/ttt_math.py`, `frobenius_clip`). With the pretrained ‖W₀‖ averaging
46.6 across the TTT layers (`inputs/w0_norms.json`), the clip binds whenever the
state ratio ‖η·S‖/‖W₀‖ exceeds **0.43**.

`T6_clip_saturation`:

| | rows checked | above the clip threshold | smallest margin |
|---|---|---|---|
| hebbian, steps 200/400/600 | 826 / 826 / 2096 | 100% | 843× / 292× / 255× |
| delta_chunk, steps 200/400/600 | 826 / 826 / 2096 | 100% | 2478× / 1900× / 1597× |

The carried state's magnitude never reaches the output. What reaches it is a vector
of fixed norm τ whose direction is that of the accumulated state. Two observations
follow and are both in the data (`F3`):

- **Re-reading a document grows the state 6.5× and changes nothing.** From
  the first reading to the 10th, the delta_chunk state ratio rises from about 14,600 to
  about 95,000 (hebbian: about 2,600 to about 16,900). Perplexity is flat to ±0.001: once the
  direction has settled, more of the same signal is invisible.
- **Carrying a state into a new document doubles to quadruples its size** (`T5`: `persist` vs.
  `doc_reset` at chain positions 1–4) **with no effect on perplexity.** The carried
  direction is an average over earlier documents (carried_decay = 0.9). Once it
  is normalized to τ, it holds nothing specific to the next document.

**What this does not yet show:** the saturation is measured, but its role as the
cause is inferred. The direct test is an eval-only ablation on existing checkpoints:
re-run the boundary and repeat suites with the clip disabled or τ raised. If
`cross_doc` and the repeat curve move, the clip is the bottleneck. If they stay
flat, the carried state is uninformative no matter how it's read. That costs two GPU evals
(one per rule) at step 600 and no training.

## 5. Sanity checks that hold

- The LoRA-only checkpoint gives exactly 0.000 for `within` and `between_slices` at all three
  steps. The eval pipeline does not invent a TTT effect.
- All bundles share the same 106 documents. The true base perplexities are identical
  across the three LoRA-only bundles.

## 6. Corrections to earlier results

- `fresh` in bundles from TTT checkpoints written before the fix measured the base model
  with the *trained* `W_down`, not the true base. This analysis takes `fresh` from
  the LoRA-only bundle at the same step (`paper_analysis.rebased`). The code now
  restores the pretrained `W_down` for `fresh` (`pretrained_mlps`), so new bundles
  are correct as written.
- Hebbian checkpoints after step 650 on the `suchitra-idu` volume come from a
  different run and are excluded. Delta_chunk was evaluated from the
  `aiteera-llc` workspace's volume. Both volumes contain a folder called `ttt-v1.1`
  holding different runs.

## Open items

1. **Confirm the LR schedules match:** in W&B, the `step N/<total>` progress line should show
   the same `<total>` in all three runs.
2. **The clip ablation** in §4: two evals, recommended. It turns the explanation
   from correlational into causal.
3. **Limitations to state:** the capacity confound (§1), one seed, one model size,
   at most 600 training steps, and 6 Book documents in the holdout.
