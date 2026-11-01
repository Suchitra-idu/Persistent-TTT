# PLAN

Research plan for the session-persistent TTT project, from the current thesis to a TMLR submission.
Revised 2026-11-01 (first written 2026-10-27). Source of truth for every number is `results/`.
`plan/TMLR Paper writing guide.md` explains how the paper must be written.

---

## 1. Scope (fixed; do not grow it)

**Question.** Does a TTT fast weight that is trained to carry between slices of one document transfer
to the next document when its reset is removed, and when does it help or hurt? And does training
the carry across document boundaries change that (one single-seed probe)?

**Claim level.** Stream or domain adaptation measured by perplexity (S-level in Beyond Perplexity).
No memory, recall, personalisation or continual-learning claims, and no motivation that implies
them.

**Contributions (four).**
1. The naive carry (In-Place TTT with the reset removed) has no measurable cross-document effect;
   its clip is saturated, but so is the working carry's, so the clip alone does not explain it.
2. A carry path trained within documents gives a reliable between-slice gain (2 seeds); in a
   single-seed ablation its two changes appear to work together.
3. Across documents, the gain is ordered by relatedness; the paper reports how much of it survives
   a generic-carry control, that the carry harms after other-source text or a long chain, and, in a
   single-seed probe trained on sessions that cross document boundaries (1.5), whether that
   training removes the harm while keeping the gain.
4. Against attention on the same prime, which fits in the carry's own memory budget, the carry is
   small; it does not accumulate with re-reading.

**Out of scope, stated once in the paper.** A full study of cross-document training (only the
one-seed mixed-session probe 1.5); training past
step 600; comparison claims against dynamic evaluation (development-set result in an appendix
only); the delta rule beyond the naive-carry result (appendix); per-domain analysis (Q2, thesis
only); other models or sizes; downstream or recall benchmarks; compressed carries; long primes.

The main models are trained only within documents (`carry_window=2`), so their cross-document use
is a transfer test; the mixed-session probe (1.5) is the one model trained across boundaries. Both
sentences go in the abstract.

---

## 2. Working rules

1. **Treat the thesis as the submission.** Every claim meets TMLR's two criteria. Length is free
   in the thesis; clarity, evidence and honesty are not.
2. **The thesis is a working draft; the final paper is the author's.** Claude may edit
   `paper/main.tex` freely, and writes and tests code, runs experiments, and checks every number
   against `results/` and every citation against `Notes/`. The final TMLR paper is written by
   the author, using the thesis as material, so that every sentence is one the author can defend.
3. No em dashes, en dashes or colons in thesis prose. Figures as PDF. Every number traces to
   `results/`. Cite only works with a note in `Notes/`.
4. The LLM-use footnote states what was actually done (guide §9). If a sentence of it is not
   true, change the sentence, not the record.
5. Authorship follows contribution. Prof. Yapa is a co-author only if he contributes to the work
   or the paper, not to change how the submission looks.

---

## 3. Where the project stands

Numbers are Δppl on the development set, paired bootstrap 95 percent intervals, Qwen3-0.6B,
Hebbian, step 600 (FINDINGS 4b to 4f).

- Carry design: between slices +0.097 (s42), +0.092 (s43); vs matched LoRA +0.234, +0.214.
- Relatedness (Wikipedia, 200 cases): same article +0.103, related +0.085, random +0.051, other
  source −0.067 (s42; s43 replicates in sign and significance). Related minus random +0.034
  [+0.024, +0.044]. Plain Hebbian is flat.
- Cross-document in the boundary suite (chains of 20 SlimPajama documents) drifts with training:
  +0.010 at 200, −0.003 at 400, −0.020 at 600.
- Carry recovers about 4 percent of the gain of the previous slice in attention. The carried state
  is 44.0M values, about the bf16 attention cache of 1,536 tokens, so the 1,024-token prime fits
  in an equal-memory cache.
- No accumulation with re-reading. The carry path's clip is still saturated (state ratio median
  about 8k s42, 15k s43).

**Reviewer objections, in order of danger, and where each is answered.**

| # | Objection | Answer |
|---|---|---|
| 1 | Random-article gain is a generic "this is Wikipedia" prior | 1.1 generic carry; headline contrasts are against it |
| 2 | Every design choice was made on the evaluation documents | 1.2 fresh set, preregistered |
| 3 | No in-context baseline on the relatedness cases | 1.3 (also the equal-memory baseline) |
| 4 | Boundary suite (harm) and relatedness suite (help) disagree | 1.4 chain length, on Wikipedia |
| 5 | "Related" is title mention, confounded with word overlap | Defined as title-linked, word overlap included; no topic-vs-overlap claim |
| 6 | Training stopped at 600 of a 1,050-step schedule; cross-document effect still drifting | Limitation, with the drift reported |
| 7 | The carry is never trained across documents | 1.5 mixed-session probe (one seed); scope sentence (§1) |
| 8 | Two seeds; ablation single-seed at step 200 | Limitation |
| 9 | Dynamic-evaluation baseline on a different model | No comparison claim; appendix only |
| 10 | Clip-as-cause and novelty overclaims | Fixed in the thesis 2026-11-01 |
| 11 | Wikipedia eval articles may be in training data | 1.2 title and hash check |

---

## 4. The plan, by tier

### Tier 0. Free

| # | Item |
|---|---|
| 0.1 | Final TMLR paper written by the author (unmodified style file, at most 12 main pages so reviews take 2 weeks, anonymised) |
| 0.2 | Defence sheet `plan/DEFENCE_QA.md`, about 30 hard questions, answers traced to `results/`, studied by the author |
| 0.3 | Ask Prof. Yapa to read critically, and to co-author if he wants to contribute (rule 2.5) |
| 0.4 | Claims-to-evidence table; every configuration run, failures included; provenance of −6; true LLM-use footnote; code and bundles in an anonymous mirror for review |
| 0.5 | Confirm in W&B that all runs used the same LR schedule total (FINDINGS open item 1). If not, the "vs matched LoRA" numbers are confounded; fix before anything else |

### Tier 1. Required (about $30: one training run, then one evaluation pass)

| # | Item | Cost |
|---|---|---|
| 1.1 | Generic carry: mean of carried states over 200 random fresh Wikipedia articles disjoint from the test cases, rescaled to the median norm of single carried states, used for every target | ~$3 |
| 1.2 | Fresh test set. Relatedness on Wikipedia shards 2 and 3 (development used shards 0 and 1; the builder needs a shard offset, since `load_articles` reads `range(n_shards)`), each related prime used for one target only, titles checked against SlimPajama training documents. Regimes suite only on a new SlimPajama holdout seed, excluding the old 106 documents (the holdout pool is disjoint from training by construction, `split_holdout`). Fresh boundary and repeat suites are not run; their development results stay secondary. Preregistration committed and pushed before running | ~$6 |
| 1.3 | In-context baseline: prime in attention, same scored tokens. This is also the equal-memory baseline (1,024 tokens < 1,536) | ~$3 |
| 1.4 | Chain length on the fresh Wikipedia cases: 1, 2, 5, 10 random articles carried before the target | ~$2 |
| 1.5 | Mixed-session probe `mixed-carry-600`. Same data, flags, seed 42 and 1,050-step schedule as `hebbian-carry-600`, trained from scratch to step 600 (not continued, so extra training is not confounded with the change); only the session strategy differs. A new strategy `mixed`, a variant of `minilasting`: sessions chain 5 sliced documents, and each next document is from the same source with probability 0.5, otherwise from a random other source. The state is carried across every boundary; with `carry_window=2` the gradient crosses a boundary when it falls inside a 2-item window (about half of them). Checked in code 2026-11-01: `minilasting` and `hybrid` give the same step count, and memory matches the current window. Evaluated with the same fresh suites as the main models (relatedness with generic carry, chain length, regimes), paired on the same cases | ~$13 train, ~$3 eval |

**Analysis, fixed in the preregistration (no forking paths).**
- Metric: Δppl, as in every existing table; percent of reference perplexity in parentheses for
  headline numbers.
- Intervals: paired percentile bootstrap over documents, as now, with 10,000 resamples. Valid
  because each related prime is used once. Each seed reported separately.
- Headline contrasts (four): same minus generic, related minus generic, related minus random, carry
  minus in-context on related. Each gets a 95 percent interval and a Bonferroni-adjusted
  (98.75 percent) interval. No p-values.
- Decision rule for the main claim, applied identically whatever the outcome. "Document-specific
  transfer" appears in the title and abstract only if the adjusted interval of related minus
  generic excludes zero in both seeds. Otherwise the claim is "domain-level adaptation", and only
  the relatedness gap is reported as document-specific.
- Chain rule: if harm grows with chain length, the boundary vs relatedness conflict is explained
  as accumulated interference; otherwise it is stated as open.
- Mixed-session probe (1.5), seed 42 only, paired against `hebbian-carry-600` seed 42 on the same
  fresh cases. Four contrasts, mixed minus within-document model: other source, chain of 10,
  related, and between slices on fresh SlimPajama; 95 percent intervals. The sentence "training
  across documents removes the harm" is written only if the other-source and chain-10 contrasts
  are positive with intervals excluding zero and the related contrast's interval does not lie
  wholly below zero. Every result is labelled "in a single-seed probe". If the probe learns to
  close the carry everywhere (related and between-slice gains near zero), report that.

**Cut, stated as limitations where they bear on a claim.** Shared-token split, cluster bootstrap,
fresh boundary and repeat suites, training to step 1050, a third seed, a
second-seed ablation, rerunning dynamic evaluation, cloze probe, dose-response, low-rank carry,
λ sweep, mechanism logging, long primes, a second seed or other mixing ratios for the mixed
probe, a source-only or unrelated-only training run, delta-rule carry, larger model, DOI bundle
archive.

### Timeline (submission target early January 2027)

| When | What |
|---|---|
| Week 1 (to 8 Nov) | 0.5 LR check; commit and tag `pre-fresh-set`; preregistration; `mixed` strategy, then launch 1.5 training; rest of Tier 1 code while it trains |
| Week 2 | Fresh pass on all three checkpoints; analysis |
| Weeks 3 to 4 | Thesis update with fresh results; re-audit; defence sheet; supervisor reading |
| Weeks 5 to 8 | Final paper written by the author; pre-submission checks |
| Early January 2027 | Submit; arXiv preprint |
| About March to April 2027 | Decision (9-week target; reported medians about 13 weeks) |

---

## 5. TODO

Owner is the author unless marked (Claude).

### Done
- [x] Literature review rebuilt from notes; thesis audited; audit fixes applied
- [x] Guide and plan written; `Notes/` entries for Beyond Perplexity, TTT-NTP, continual facts, delayed supervision, AutoCompressors, Generative Adapter, Cartridges, MemoryLLM, Larimar, Infini-attention, LaCT
- [x] 2026-11-01 `Notes/` entries for Zhao et al. 2024 and Hardt and Sun 2024 (from arXiv HTML)
- [x] 2026-11-01 Thesis claim fixes: scope and claim level in the abstract; user motivation removed; novelty narrowed with three new citations; memory and "where attention cannot reach" claims removed; Ch. 6 retitled, clip-as-cause wording removed; development-set, generic-prior, equal-memory and step-600 drift limitations added
- [x] 2026-11-01 Thesis condensed (22k to 17k words): four contributions; literature review 6.7k to 2.8k words; Q2 and the capacity section condensed; dynamic evaluation moved to an appendix with no comparison claim

### A. Set-up (week 1)
- [ ] 1. Check LR schedule totals in W&B [0.5]
- [ ] 2. Commit all code, thesis and results; tag `pre-fresh-set`
- [ ] 3. Ask Prof. Yapa (reader, co-author if contributing) [0.3]
- [ ] 4. `plan/PREREGISTRATION.md` from §4 analysis rules; commit and push before any fresh run [1.2]

### B. Code (Claude, week 1)
- [ ] 5. `mixed` session strategy with unit tests (source mix, no document reused, step count equal to `hybrid`, gradient reaches the previous document when the boundary is inside a window); local smoke run; cost estimate [1.5]
- [ ] 6. Wikipedia shard offset; one target per related prime; title overlap check; new SlimPajama holdout seed excluding the old 106 documents [1.2]
- [ ] 7. Generic-carry arm, norm-matched [1.1]
- [ ] 8. In-context arm [1.3]
- [ ] 9. Chain-length arm in the relatedness suite [1.4]
- [ ] 10. 10,000 bootstrap resamples and Bonferroni-adjusted intervals in `paper_analysis` [§4]
- [ ] 11. Unit tests with fakes, comment-budget test, local smoke run, cost per suite for approval

### C. Runs (week 2)
- [ ] 12. Approve cost; train `mixed-carry-600` (seed 42, `save_every=50`, detached watcher that stops it after step 600) [1.5]
- [ ] 13. Approve cost; run the fresh pass on both seeds' step-600 checkpoints and `mixed-carry-600` from the tagged commit [1.1 to 1.5]
- [ ] 14. Analyse exactly as preregistered; regenerate `results/paper/`; update `FINDINGS.md`

### D. Thesis (weeks 3 to 4, Claude drafts, author reviews)
- [ ] 15. Fresh results as headline, development set in an appendix; claims-to-evidence table; configuration list; provenance of −6; update the limitations to what the fresh set shows
- [ ] 16. Re-audit (every number against `results/`, every citation against `Notes/`)
- [ ] 17. Defence sheet [0.2]; supervisor reading

### E. Paper and submission (weeks 5 to 8)
- [ ] 18. Release: clean repo, reproduce-from-bundles README, anonymous mirror
- [ ] 19. Write the paper from the guide [0.1]
- [ ] 20. Guide §11 checklist; arXiv novelty search on the day; submit; preprint with the preprint option
- [ ] 21. Respond to reviews within 2 weeks of the third review
