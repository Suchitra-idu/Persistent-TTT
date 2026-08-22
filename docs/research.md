# Research reference

[mechanism.md](mechanism.md) is the mechanism as it exists. This page is the
research built on top of it: the proposal's objectives mapped onto the code,
the config knobs that actually matter for the write-up, and what running it
has actually found. Source: [proposal.md](../proposal.md).

**Current research path: `hebbian` only.** `delta`/`delta_chunk` were
explored in depth (findings 1, 2, 3, 5, and 8 below are from that work) and
the code stays in the repo, working and tested, but the active line of
research no longer uses them — nothing here should be read as a
recommendation to run them. Finding 4 (the output gate) and 6 (an early,
unconfirmed growth number) aren't rule-specific. `TTTConfig`'s
hebbian-relevant defaults (`clip_tau`, `carried_decay`,
`output_gate_bias_init`) are pinned to match the last git commit exactly,
undoing an in-session `clip_tau` retune that was tried and rejected — see
finding 7.

## The three components

| Component | Objective | Lives in |
|---|---|---|
| Within-document adaptation | baseline | `chunk_size`, `eta`, one item |
| Cross-document persistence | O1 | the carry, `carry_scope`, `carried_decay` |
| Meta-learned per-source init | O2 | `everlasting`'s trained carriers |

Cross-domain characterisation (O3) is every eval broken out per source.

## Research questions → instrument

| Q | Question | Instrument |
|---|---|---|
| Q1 | Does carry help at all? | `Δbetween` |
| Q2 | Does the trained per-source seed beat cold accumulation? | `Δseed` |
| Q3 | Which domains benefit most, from carry vs. seed? | per-source `Δbetween` / `Δseed` |

Formulas: [mechanism.md#the-result](mechanism.md#the-result).

## `update_rule` — hebbian only, and why that has real consequences

The full mechanism and math for all three rules —
[mechanism.md#the-update-rules](mechanism.md#the-update-rules) — is the
detailed reference; this section is only the research framing on top of it.

`TTTConfig.update_rule` defaults to `"hebbian"`, and that's the whole active
research path now. The consequence worth being explicit about: `hebbian` is
a pure outer-product accumulator with no error-correction (see
[mechanism.md](mechanism.md#the-update-rules)) — nothing in its own math
bounds its growth. `delta`/`delta_chunk` (not in use) have that boundedness
built in via error-correction; `hebbian` doesn't, so external bounding
(`clip_tau`, `carried_decay`) is load-bearing in a way it never had to be for
the rule this investigation isn't using. Finding 7 is a direct, measured
instance of that risk.

## The output gate

`out = base_out + sigmoid(W_gate·h + b_gate) · ttt_out`. `W_gate` is
zero-initialised, so the gate starts *uniform*: `sigmoid(output_gate_bias_init)`.
Current default `-2.0` → `≈0.12` at step 0 — the TTT term starts almost
entirely suppressed and has to earn its way open.

`gate_mean` / `gate_std`, captured under `no_grad` every forward, are on
`session_eval.ItemRow` and every chained-eval table's `gate` column. A gate
pinned near zero means the TTT term cannot be reaching the output, whatever
the carry's own state is doing — check this before concluding the carry is
inert.

## `minilasting` — the third strategy

Not in [extensions-map.md](extensions-map.md)'s table yet. `carry_scope =
"session"`, but a session is `docs_per_session` (default 5) documents of
*one* source, round-robin picked across sources, chained and sliced like
`hybrid`, then reset. The middle ground between `hybrid` (dies every doc) and
`everlasting` (never dies): carry outlives one document but not the epoch —
"a user leaving it on for a session of use, not forever."

## Config reference — research-relevant knobs

| Field | Default | Note |
|---|---|---|
| `update_rule` | `"hebbian"` | the sole active research path; `delta`/`delta_chunk` exist in code but are not in use |
| `output_gate_bias_init` | `-2.0` | `-0.5` and `+2.0` (less damped) both tried, both hurt early training |
| `carried_decay` | `0.9` | applies every item, not just cross-session — see finding 1. Steady state `1/(1-decay) = 10×`. **Do not raise above this without also addressing `clip_tau` — see finding 7** |
| `clip_tau` | `2,000,000` | matches the last git commit exactly, by decision. Confirmed unsafe for `hebbian` at `carried_decay=0.98` (finding 7) — a tighter value was tried and reverted, so this risk is currently live, not fixed |
| `truncate_every` | `5` | `delta`/`delta_chunk`-only, not consulted by `hebbian` at all |
| `eval_n_docs_per_source` | `5` | caps the holdout pool *before* any downstream eval script's own doc-count flag sees it — see Pitfalls |

## Findings

Real results from running this system, not spec or intent. Cite the source
files directly; this is a pointer, not a replacement.

1. **`delta_chunk`'s carry is bounded, not compounding, across long chained
   sessions.** `state/W0` oscillates within a per-source band (roughly
   300–900) rather than trending up with chain length, confirmed at 5, 15,
   and 25 documents chained — ruling out "not enough documents yet." Two
   independent forces point the same way: the update rule's own convergence
   (above), and `carried_decay` — which is **not** cross-session-only, it
   fires every item, including every slice inside one continuous session.
   With `decay=0.9` (ceiling `10×`) fighting the error-correction every item,
   the two settle into a dynamic equilibrium, not growth.

2. **The benefit is real and doesn't decay over a long session — it just
   doesn't grow either.** Per-slice `Δbetween` stays mostly positive across
   an entire 25-document chain, every source tested. "Persists without
   decaying" is the claim the data supports; "compounds" is not.

3. **Effect size varies enormously by domain (direct Q3 evidence).** Mean
   `Δbetween` over a 5-doc chain, one real run: C4 **+0.74**, StackExchange
   +0.21, Book +0.20, Github +0.14, Wikipedia +0.10, ArXiv **+0.03**. Every
   source net-positive, ~25× spread top to bottom. ArXiv was also the only
   source with a small negative dip at every one of 4 document boundaries
   tested, before recovering — a plausible "topic-switch cost" specific to
   long, internally-coherent documents that the noisier sources didn't show.

4. **The output gate: two numbers that look contradictory and aren't.**
   Raising `output_gate_bias_init` to `+2.0` caused wild early gradients and
   flipped every source's `Δbetween` negative by step 25. The gate *did*
   close back down after (eval-time `gate_mean` ≈0.02–0.05) — but a direct
   checkpoint read (`ttt/experiments/gate_report_v1.py`, reads
   `output_gate.bias`/`.weight` off the loaded model directly) showed the
   *bias* had barely moved from its `+2.0` init. The *weight* did all the
   closing — it grew a large norm and learned a direction that dominates the
   bias for real hidden-state magnitudes. Verify any single computed
   `gate_mean` against the raw parameters before trusting it.

5. **The nan-gradient crash, root-caused.** Backward chain through repeated
   `frobenius_clip` calls compounding across untruncated steps, combined with
   a fixed `eta` exceeding the delta rule's own stability bound once real
   activations hit outlier-scale dims. Not reproducible in isolation on
   CPU/toy inputs — found via `torch.autograd.set_detect_anomaly` on a live
   GPU run. Fixed permanently by `_adaptive_eta` + `truncate_every`.

6. **An earlier, much larger carry-growth number (342→2230) was very likely
   `hebbian`'s, not `delta_chunk`'s** — never fully confirmed which
   `update_rule` that run used, but every controlled `delta_chunk` run since
   has stayed under 1000 regardless of chain length. Resolve with a pinned,
   logged rerun before citing either number.

7. **`hebbian` + a raised `carried_decay` reproduces catastrophic runaway
   growth at the default `clip_tau` — confirmed, and currently unfixed by
   choice.** A real training run, `update_rule=hebbian,carried_decay=0.98`
   (133 steps, from scratch): `state_ratio_final` reached **1.93e6** — right
   up against the `2,000,000` clip ceiling, meaning the clip was the *only*
   thing standing between this and a non-finite crash, not evidence the
   config was fine. Perplexities collapsed accordingly (`cold_carry` ppl in
   the tens of thousands — 114,027 for Wikipedia; `Δbetween` −29,306). Root
   cause: `hebbian` has no error-correction (finding 1) and `carried_decay=0.98`
   gives it a `50×` steady-state ceiling instead of `0.9`'s `10×`, so nothing
   slows the accumulator down before it hits the clip boundary — and
   `clip_tau=2,000,000` was never validated against `hebbian`'s own growth
   rate; it's simply the value that's been in the config the whole time.
   A tighter `clip_tau=2,000` was tried and **did** stop this specific
   blowup in isolated testing, but was **deliberately reverted** — the
   decision was to keep `hebbian`'s config identical to the last git commit
   rather than carry an ad hoc, unvalidated retune forward. Practical
   consequence: **`carried_decay` above its default `0.9` is not safe with
   the current `clip_tau` and hasn't been re-addressed.** The `scan()`
   kernel itself is unchanged throughout all of this (verified via `git
   diff` against the pre-investigation commit) — this finding is entirely
   about the config surrounding it, not the update rule's own math.

8. **Widening `truncate_every` is expensive, measured directly.** ~70 MB of
   retained backward-graph memory per TTT layer per chunk in the window,
   measured at Qwen3-0.6B dims (hidden=1024, d_ff=3072) on real GPU hardware.
   At 14 TTT layers, reaching one document's worth of chunks (~41 at
   `chunk_size=50`) costs ~40 GB; a full 5-document chained session's worth
   costs ~200 GB — not feasible on an 80 GB H100 without activation
   checkpointing on the scan itself. This is what makes "just widen
   `truncate_every` for full credit assignment" a real engineering project,
   not a config change — see Open questions.

## Pitfalls found and fixed

Worth a footnote in the methodology section — each was a real, silent source
of wrong numbers, not a hypothetical.

- **Single-source default.** Early eval scripts measured only the first
  source in pool order (in practice, always ArXiv), not a sample across all
  configured sources. Fixed by explicit per-source grouping everywhere.
- **`n_docs` silently capped below what was asked.** A chained eval's own
  `--n-docs` flag controls how many documents to *chain*; `eval_n_docs_per_source`
  separately caps how many even enter the holdout pool, upstream, silently.
  Asking for 15 delivered 5 with no error. Fixed by raising the pool cap to
  match what was requested, and surfacing `Holdout.shortfalls` (existing,
  previously discarded) so a source that genuinely can't supply enough says
  so.
- **Chaining across sources instead of within one.** An early multi-document
  chained eval round-robinned documents across *different* sources into one
  continuous carry session — a shape `carry_scope=SOURCE` never produces in
  production, since each source's carrier is kept strictly separate. Fixed to
  chain multiple documents of one source per session.
- **Eval-time hyperparameter swaps are confounded by what the checkpoint was
  trained under.** Swapping e.g. `carried_decay` at eval time against a
  checkpoint trained under a different value tests a state-magnitude
  distribution the model's weights never saw. A negative result there
  doesn't falsify the hyperparameter, only that the checkpoint doesn't
  generalize to it — any change meant to affect the mechanism's own
  dynamics needs a matching training run, not just a different eval flag.

## Open questions

- Does `state/W0` climb past its bounded band given much longer training, or
  is the equilibrium stable regardless of how mature `w_target`/`new` get?
  Nothing so far distinguishes "not enough steps" from "structurally can't."
- Does the gate's weight-driven suppression (finding 4) generalize across
  sources? Not yet checked per-source.
- Was the 342→2230 growth (finding 6) actually `hebbian`? Rerun pinned and
  logged.
- **How should `hebbian` actually be bounded, if `carried_decay` needs to go
  above `0.9`?** Finding 7 confirms the current default (`clip_tau=2,000,000`)
  isn't safe at `carried_decay=0.98`, and the one tighter value tried
  (`2,000`) was reverted rather than adopted, on the grounds that it was an
  unvalidated guess derived from `delta_chunk`'s (no-longer-relevant)
  operating range rather than anything characterizing `hebbian` itself. A
  properly derived answer — e.g. characterizing `hebbian`'s actual per-step
  growth rate empirically before picking a ceiling — is still open. Until
  then, treat `carried_decay > 0.9` under `hebbian` as untested and possibly
  unsafe.
- `truncate_every`/wide-credit-assignment training (finding 8) was a
  `delta`/`delta_chunk`-specific idea — moot now that `hebbian` (which
  doesn't consult `truncate_every` at all) is the sole research path. Left
  here as a record of why that direction was considered, not as a live
  option.
