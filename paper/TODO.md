# Paper TODO: negative-result paper, 2 weeks, part-time, one training run

Claim (B): persistent hebbian/delta_chunk fast weights on top of a LoRA-fine-tuned LM give no measurable cross-document benefit, and the within-document gain is small next to LoRA. Scope: one base model, one corpus family, 600-step training budget, one seed per setting.

Evidence in hand (step 600, 106 held-out docs, 6 sources): `results/step600_hebbian_vs_deltachunk.md`.
Hard constraints: ~2 weeks to submit, part-time, budget for **one** LoRA-only training run (600 steps). Everything else is evaluation only.

## Rules for this plan

- No new training beyond the single LoRA-only run.
- Every experiment must serve the claim above or the positive control that makes the null credible; otherwise it goes to Future work.
- Write as you go. Results sections are drafted the day each result lands.

## Week 1: evidence

**Day 1-2: launch the one training run (do this first, it is the only long pole)**
- [ ] Otherwise train **LoRA-only, no TTT layers, tuned (higher) lr, 600 steps**, same data/seed settings as the TTT runs. Record the exact flags in the manifest.
- [ ] Evaluate it with `paper_eval_v1 --suites regimes` (no TTT layers, so only that suite runs).
- [ ] Paper wording: this is a tuned baseline, not a matched ablation. State the lr gap explicitly.

**Day 2-5: eval-only experiments, run in parallel while training goes** (all use `paper_eval_v1` and existing checkpoints)
- [ ] Later checkpoints: hebbian `ttt-v1.1/step_800` (or 847) and delta_chunk `step_1000`. Shows whether the 600-step null is undertraining. Do not evaluate every checkpoint; pick 2 per rule.
- [ ] **Positive control**: repeated / near-duplicate document. If carry cannot help here, the mechanism stores nothing; if it can, the null elsewhere is meaningful. This is the single most important credibility experiment.
- [ ] Bigger eval: raise `n_docs` (RedPajamaBook only fills 6 of 20; drop or merge sources that cannot fill the quota) and `n_chain` for longer sessions.
- [ ] Carry sensitivity at eval time (`carried_decay` <= 0.9, `clip_tau`): one small figure. Do not go above `carried_decay=0.9` for hebbian except to show the known blow-up.
- [ ] Controls: carry zeroed, shuffled-source carry. Should reproduce the null.
- [ ] Retrieval-in-context baseline (prepend retrieved prior docs) if it fits in a day; otherwise cite and move to limitations.

**Day 6-7: analysis**
- [ ] Re-render all bundles; one master table pooling both rules + LoRA-only baseline.
- [ ] Pick the single pre-declared primary comparison (`cross_doc`, pooled) and state it before looking at the new runs; label everything else secondary and note multiple comparisons.
- [ ] Flag "significant but negligible" results (e.g. hebbian `same_vs_other_source` +0.09%) honestly.

## Week 2: writing and submission

**Day 8-10: draft**
- [ ] Intro: question, why persistent fast-weight state is plausible, what we found.
- [ ] Method: the two update rules, carry, LoRA setup (reuse `docs/mechanism.md`; do not re-derive).
- [ ] Setup: data, held-out protocol, metrics (Δwithin, Δbetween, Δlora, cross_doc, swap), bootstrap.
- [ ] Results: gap decomposition, repeat/stability, swap, positive control, later checkpoints, tuned-LoRA baseline.
- [ ] Analysis: why the null (LoRA absorbs signal, bounded vs unbounded state, little cross-doc structure in these sources).
- [ ] Related work from `Literature_Review/` (trim to what the claim needs).

**Day 11-12: figures and polish**
- [ ] Final figures from `raw/` (regime_gaps, repeat_curve, stability, swap_matrix, positive control). Consistent fonts, colourblind-safe palette.
- [ ] Limitations section: one base model, one seed per setting (bootstrap covers eval noise, not training noise), LoRA baseline not lr-matched, delta_chunk off the main research path, small eval set.
- [ ] Reproducibility: commit hash and flags per bundle, one clean commit for the reported numbers, `README` in `paper/` listing which bundle produced which table.

**Day 13: review**
- [ ] One read-through for claim discipline: every sentence says "in this setup", nothing says "TTT does not work".
- [ ] One external reader if anyone is available.

**Day 14: submit**
- [ ] Format to venue template, check length and scope, upload.

## Cut list (do not attempt in 2 weeks)

- Full lr x method grid, rank/parameter-matched LoRA, extra training seeds.
- Retraining hebbian with new hyperparameters or longer than 600 steps.
- RULER, downstream-task evaluation, comparison against reimplemented Titans/other published TTT systems (cite them instead).
- Ablations over `eta`, `chunk_size`, layer placement.

Put these in a "Future work" paragraph; they also pre-empt the obvious reviewer asks.

## Open decisions

- [ ] Include delta_chunk? Recommended yes, as the error-correcting contrast: the null replicating across both rules is what makes the claim stronger than one failed config.
- [ ] Venue: with a 2-week deadline, confirm the target accepts this length/format and that "Q2" refers to the venue you can actually submit to by the date.
