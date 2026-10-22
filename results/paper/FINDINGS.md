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

The lead holds on every source (`T2_per_source`). In absolute perplexity it is largest on C4,
Wikipedia and Book and smallest on Github and ArXiv. In relative terms
StackExchange leads for both rules.

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

The `same_vs_other_source` +0.010 for hebbian is a foreign state hurting less,
not a same-source state helping. Priming on a different source raises hebbian
perplexity in 15 of 30 source pairs (mean −0.022, down to −0.07), and
same-source priming is about neutral (−0.008). Delta_chunk priming is neutral
(3 of 30 pairs significant, mixed directions).

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

## 4. The clip discards the state's size, but removing it does not help

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

**Whether the clip can be removed depends on the update rule.** Two short
training probes without the clip (`clip_enabled=false`, otherwise the original
run's flags; gradient clipping at `max_grad_norm=10` stays on in both):

| probe | steps | result |
|---|---|---|
| hebbian, no clip (`hebbian-noclip`, W&B `ygka2rdc`) | 200 | **Collapses.** Loss 25.8 averaged over the first 10 steps; at step 10 the `W_down` gradient norm is 132, 13× the gradient clip. It settles at a loss of about 8.1 (perplexity about 3,300) and does not recover within 200 steps. The unclipped delta_chunk probe, on the same data order, stays at 1.8–2.8. |
| delta_chunk, no clip (`delta-noclip-probe`, `delta-noclip`) | 105, 100 | **Stable loss** (1.8–2.8, gradient norms 0.3–1.6, the gradient clip never kicked in), **because the TTT path is shut off** (below). |

- **Hebbian:** the clip is necessary. Its update has no error term and a fixed step,
  so the state grows with every chunk and wrecks the forward pass within 10
  steps.
- **Delta_chunk:** training survives without the clip, but only by switching
  the TTT path off. Evaluated at step 100 against the clipped run at step 100 (same
  106 documents; bundles `delta-noclip/step_100` and
  `ttt-v1.1-clean-deltachunk/step_100`):

  | step 100 | with clip | without clip |
  |---|---|---|
  | within | +0.276 * | +0.003 |
  | between_slices | +0.051 * | +0.002 |
  | cross_doc | −0.002 | −0.005 |
  | cross_slice | +0.062 * | +0.004 |
  | output gate, median | 0.92 | **0.00** |
  | state ratio, median | ≈ 73,600 | ≈ 890 |
  | full model, without vs. with clip | | −0.167 * (worse on 106/106 documents) |

  Without the clip, the output gate closes completely, so every TTT effect is
  zero, and the model ends up worse overall. Even unclipped, the state is about 2,000×
  above the clip threshold. That is far too large to use, and training shuts
  it out.

**What this establishes.** The clip is what makes the TTT term usable. It brings
the state down to a fixed scale the model can use, and when it is absent (delta_chunk)
or insufficient (hebbian's gate stays at 0.002 even with it), training
shuts the gate. Removing the clip does not let the state's size carry information
across documents: nothing reaches the output at all. So the clip does not explain the
cross-document null result. That result holds with the clip, without it, and for both update rules.
What a document passes to the next is a fixed-norm direction. Measured directly,
that direction carries nothing, not even when it was built from the same document
(`repeat_9_vs_1` = +0.000).

Each time the TTT term becomes unusable, the optimizer responds the same
way: hebbian with the clip, delta_chunk without it, both end with a closed gate.

## 4b. A redesigned carry works on related text

Hebbian only, run `hebbian-carry-b`, bundles `results/hebbian-carry-b/step_{100,200}`,
table `tables/T11_carry_design.csv`, figures `F13`, `F14`, training logs in
`inputs/carry_design_training/`. Changes against `ttt-v1.1`: separate carry path
with its own gate and a zero-initialised scale (`carry_gate`), training through
the carry in windows of 2 slices (`carry_backprop`, `carry_window=2`), main gate
bias −6.

| measure | Hebbian @200 | carry design @100 | carry design @200 |
|---|---|---|---|
| between slices, same document | +0.014 * | +0.135 * | +0.099 * |
| cross-slice, boundary suite | +0.014 * | +0.151 * | +0.119 * |
| cross-document | +0.008 | +0.011 * | +0.002 |
| within a slice | +0.016 * | +0.047 * | +0.018 * |
| vs matched LoRA | +0.073 * | | +0.215 * |

Carry from earlier slices of the same document helps 7–10× more than the
original design, from the second slice on and flat after that (no accumulation).
Main gate median 0.0007, so the benefit flows through the carry path.

Trained to 600 (`hebbian-carry-600`, bundles `results/hebbian-carry-600/`, same
config and seed; boundary chains of 20 documents from step 200 on):

| measure | @200 | @400 | @600 |
|---|---|---|---|
| between slices, same document | +0.099 * | +0.062 * | +0.097 * |
| cross-document (100 docs) | +0.010 * | −0.003 | −0.020 * (hurts on 71%) |
| vs matched LoRA | +0.215 * | +0.188 * | +0.234 * |
| carry share of previous-slice context | 3.9% [3.1, 4.6] | | 4.0% [3.3, 4.7] |

So: same-document carry survives full training; cross-document carry turns
harmful (4 of 6 sources significant). At 600 the design leads matched LoRA by
more than plain Hebbian (+0.106) or delta (+0.203). Carried state = 44.0M values,
176 MB fp32, about the bf16 attention cache of 1,536 tokens
(`inputs/param_counts.json`). Bundled changes, one seed.

Two failed attempts first (`inputs/carry_design_training/README.md`): without
the zero-initialised scale the carry doubled perplexity (τ = 20 is 4–5× the
largest singular value of W0, `inputs/w0_spectral.json`); with the scale but the
gate at −2, the in-document Hebbian fast weights wrecked the model until about
step 140, and training through every slice ran out of memory near step 195.

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
2. **The no-clip probes are short and single-seed** (§4): 100–200 steps, one
   seed each. Report them as probes, not converged runs.
3. **Limitations to state:** the capacity confound (§1), one seed, one model size,
   at most 600 training steps, and 6 Book documents in the holdout.
