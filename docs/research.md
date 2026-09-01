# Research reference

[mechanism.md](mechanism.md) is the mechanism as it exists. This page is the
research built on top of it: the proposal's components mapped onto the code,
what each required check maps to in the eval scripts, the config knobs that
actually matter for the write-up, and what running it has actually found.
Reflects the latest proposal revision — two components, two research
questions, dropping an earlier draft's third, meta-learned-init component and
third research question. **Note:** the repo's `proposal.md` file itself still
holds that earlier draft as of this writing; this page follows the revision
given directly, not the checked-in file.

**Current research path: `hebbian` only.** `delta`/`delta_chunk` were
explored in depth and the code stays in the repo, working and tested, but
the active line of research no longer uses them — nothing here should be
read as a recommendation to run them. `TTTConfig`'s hebbian-relevant defaults
(`clip_tau=2,000,000`, `carried_decay=0.9`, `output_gate_bias_init=-2.0`) are
pinned to match the last git commit exactly. `carried_decay` above `0.9` is
untested and likely unsafe at the current `clip_tau` — `hebbian` has no
error-correction of its own, so nothing bounds its growth except these two
config values.

## The two components

| Component | Answers | Lives in |
|---|---|---|
| Within-document adaptation | baseline, no persistence | `chunk_size`, `eta`, one item |
| Cross-document persistence | Q1, and Q2 broken out per source | the carry, `carry_scope`, `carried_decay` |

Meta-learned per-source initialisation (an earlier draft's third component)
is **not part of the current proposal at all** — not de-emphasized, not a
secondary check, simply out of scope. The `everlasting` strategy and the
`Δseed` metric exist in code as leftover infrastructure from that earlier
draft; nothing below depends on them, and the source-mismatch check further
down uses only a plain accumulated carry (e.g. from `repeat_carry_eval_v1`),
not a trained per-source seed.

## Research questions → instrument

| Q | Question | Instrument |
|---|---|---|
| Q1 | How much does carry contribute to language model improvement? | `Δwithin`, `Δbetween`, aggregated with a confidence interval — see Methodology |
| Q2 | What kind of domains perform best with session-persistent TTT? | per-source `Δwithin` / `Δbetween`, ranked, against a domain-level correlate |

Formulas: [mechanism.md#the-result](mechanism.md#the-result).

## Methodology — what each required check maps to

The proposal's evaluation section describes several checks beyond "measure
perplexity once." Each one maps to existing code; none of them are new
mechanism work, only eval runs and analysis.

| Check | Why it's required | Maps to |
|---|---|---|
| Reset-baseline comparison | Isolates the gain attributable to persistence from within-document adaptation alone | `single_doc_eval_v1`'s `carry` vs `carry_off` (or `run_chained`'s per-slice `Δbetween`) |
| Repeat-and-track | Does perplexity change shape over a session — rising then levelling off, constant from the first repeat, or something else? Don't presuppose which | `repeat_carry_eval_v1` / `compounding_pilot_v1` — track `Δppl` per repeat, not just a final number |
| **Stability check** | A carry that grows without bound would make any perplexity gain misleading, not real — required *before* trusting the gain, not optional polish | `state_ratio` across the full repeat sequence — report it as a figure alongside the perplexity trend |
| **Source-mismatch swap test** | Confirms a gain is source-specific rather than incidental to any nonzero perturbation | `force_source` in `session_eval.session_perplexity` — evaluate a source-A carry against a source-B document. Any accumulated carry works; no trained seed is needed or used |
| `Δlora`, within the combined model | How much of the trained LoRA+TTT model's own improvement is LoRA's, holding that model's weights fixed | `lora_only` regime (`evolve=False`) — a decomposition of one trained checkpoint, not a comparison between two |
| **Architectural ablation, matched budget** | The comparison that actually answers "does adding TTT help" — a *separate* model with no TTT layers patched in at all, trained at the same LoRA rank/lr and the same step budget, then compared directly. Not the same question `Δlora` answers | Two independent training runs (`update_rule` irrelevant to the no-TTT run since it has no TTT layers), same `lora_r`/`lr_lora`/step count, final perplexity compared directly |
| **Rank-equivalence sweep** | A calibrated size for the combined model's total improvement: find the pure-LoRA (no TTT) rank whose own total improvement matches what LoRA+TTT achieves. Already known to be below the matched rank (that one wins outright) — the sweep finds where they cross. A number, not an explanation | Two or three more no-TTT training runs at lower `lora_r`, same step budget/data, compared to the LoRA+TTT model's total improvement |
| Domain correlate (Q2) | A ranking alone is a table; a correlation against something already measured is a stronger Q2 answer | Per-source `Δwithin`/`Δbetween` vs. that source's own `fresh`-regime perplexity — report as a correlation, not a causal claim |
| RULER probes | Supplementary held-out signal, not a benchmark match | `ruler_eval_v1` |

None of the above changes the mechanism or its config — it's the analysis
that turns "we ran it and got some numbers" into a result that answers Q1 or
Q2 defensibly. Statistical rigor matters here: aggregate `Δwithin`/`Δbetween`
across every held-out document with a bootstrap 95% CI, not a single
document's numbers — a small effect that's statistically distinguishable
from zero and a small effect that isn't are different, both honest,
findings, and only the CI tells you which one you have.

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
the rule this investigation isn't using — see Open questions.

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
| `carried_decay` | `0.9` | applies every item, not just cross-session. Steady state `1/(1-decay) = 10×`. **Untested and likely unsafe above this value at the current `clip_tau`** |
| `clip_tau` | `2,000,000` | matches the last git commit exactly, by decision — never retuned for `hebbian`'s own growth rate |
| `truncate_every` | `5` | `delta`/`delta_chunk`-only, not consulted by `hebbian` at all |
| `eval_n_docs_per_source` | `5` | caps the holdout pool *before* any downstream eval script's own doc-count flag sees it — see Pitfalls |

## Where this stands right now

Plainly, as observed, with no explanation attached — the causal "why" is not
established and doesn't belong here until it is:

- LoRA alone outperforms LoRA+TTT for general model improvement — including
  at matched LoRA rank with no TTT layers patched in at all.
- The carry does not grow over time under repeated exposure to the same
  document — it does not show an ability to accumulate information across a
  session.
- RULER shows no meaningful improvement, and sometimes a slight regression.

For context, not as an excuse: In-Place TTT (Feng et al., 2026), the
architecture this work extends, reports only small RULER movement itself
(sometimes none), and does not compare against a LoRA baseline at all. The
gap this research actually fills is the cross-document carry itself —
whether persisting the fast weight past a document boundary does anything —
which no prior work, including In-Place TTT, has implemented or measured.
That the carry does not demonstrate a growing, learning-from-context
property is itself a direct, reportable answer to Q1, not a shortfall in the
study.

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

- **How should `hebbian` actually be bounded, if `carried_decay` needs to go
  above `0.9`?** The current default (`clip_tau=2,000,000`) was never
  derived from `hebbian`'s own growth rate — it's simply the value that's
  been in the config. A properly derived ceiling — characterizing
  `hebbian`'s actual per-step growth empirically before picking one — is
  still open. Until then, treat `carried_decay > 0.9` as untested.
- Does the carry's flat, non-growing behavior under repeats hold across many
  documents and sources, or is it specific to the ones tested so far? Needs
  the statistical treatment in Methodology, not a single-document read.
