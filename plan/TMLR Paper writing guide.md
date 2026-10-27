# TMLR Paper Writing Guide

How to write the final paper so it passes TMLR's desk screen, its AI soundness review, three
reviewers, the Action Editor, and a possible author interview. Every rule has a source tag
(Section 15). **(judgement)** marks my own advice rather than a sourced fact.

Honest limit. No guide guarantees acceptance. This guide covers every rejection route TMLR
documents; the remaining risk is reviewers finding the evidence too weak, which only the Tier 1
experiments in `plan/PLAN.md` reduce. Before submitting, read 5 to 10 recent TMLR reviews on related
topics in a browser (I could not access them automatically).

---

## 1. How TMLR decides

**Two criteria, nothing else.**
1. Are the claims supported by accurate, convincing and clear evidence? The most important one [T1, T2].
2. Would some of TMLR's audience be interested, and does the paper communicate the findings clearly [B1]?

**What this means.**
- Novelty and state of the art are not required [T1]. Modest contributions are accepted [T2].
- If claims exceed evidence, you may add experiments **or reduce the claims** [T1].
- Rejected: bold unsupported statements, unclear writing, wrong novelty claims [T2].

**Who reads it, in order.** Editors-in-Chief screen (90 percent of desk rejects happen here) [B4];
Action Editor, who may desk reject within a week [T6]; an AI review that checks only claims against
evidence, advisory, which authors can answer [B2, B4]; at least three reviewers, who each answer the
two criteria Yes or No and list requested changes marked "critical" or not [T5]; then the AE decides
accept, accept with minor revision, or reject [T2].

**Numbers.**

| Fact | Value | Source |
|---|---|---|
| Desk rejection | about 53 percent now (6 percent in 2023) | [B4] |
| Review time | 2 weeks if main text is at most 12 pages, else 4 | [T4, T5] |
| Decision | about 9 weeks after submission | [T4] |
| Your response | within 2 weeks of the third review | [T4] |
| Supplementary | up to 100 MB, anonymised | [T3] |
| Quota | about 2 solo papers per author per year; desk rejects count | [B3] |

---

## 2. How papers die, and the fix

| How papers die | Source | Fix |
|---|---|---|
| Claims or evidence unclear from the writing | [B1, B4] | Sections 4, 5 |
| Evidence does not meet the claims | [T1, B4] | Sections 3, 6, 7 |
| Not interesting to the audience | [B4, T6] | Say what readers learn (4.2) |
| Fully or heavily AI-written | [B1, B4] | Section 10 |
| Authors cannot explain their paper (9 of 10 interviewed papers desk rejected) | [B5] | Section 10 |
| Not anonymised or modified style file | [T6] | Section 11 |
| Wrong novelty claim or overlap with other work | [T2, T6] | Section 9 |
| Lack of technical depth | [T6] | Sections 6, 7 |

---

## 3. Before writing, fix the claims

1. **Claims-to-evidence table first.** One row per claim: the claim, the experiment, effect size with
   interval, n, seeds, scope. Nothing goes in the abstract or introduction that is not a row. Put the
   table in the paper. [R5] **(judgement)**
2. **State the claim level** [R9]. Ours is stream or domain adaptation measured by perplexity, plus
   "bridge" evidence only if the cloze probe is run. Never claim the carry "remembers", "learns facts"
   or "helps a model keep improving on a user". Using perplexity to support a memory story is the
   named failure "evidence migration" [R9].
3. **Separate explanation from speculation** [R1]. Any mechanism story without an experiment is
   labelled "a plausible reading, not tested here".
4. **Red-team the narrative** [R8]. For us, these must each be ruled out or stated
   **(judgement)**:
   - a generic domain prior instead of document-specific transfer;
   - word overlap instead of relatedness;
   - extra parameters;
   - in-context learning;
   - selection bias from developing on the test set;
   - seed luck;
   - a handicapped dynamic-evaluation baseline;
   - the boundary harm vs. random-article help conflict;
   - undertraining (600 of 1050 steps).
5. **Mark development vs. fresh results** [R8]. Commit the analysis plan before running the fresh test
   set.

---

## 4. Structure

### 4.1 Format
- Unmodified TMLR style file; a modified one is a clear desk reject [T3, T6].
- At most about 12 pages of main text [T4]. Appendix after the references [T3].
- Reviewers may skip appendices [T3], so everything a main claim needs goes in the main text [R5].

### 4.2 Outline for our paper (about 12 pages)

| Section | Pages | Content |
|---|---|---|
| Title | | Plain and specific, no hype |
| Abstract | 0.3 | Question, setting, main finding with numbers, key control, scope, one limitation |
| 1 Introduction | 1.5 | In-Place TTT resets at every document and nobody tests carrying; our question; 3 to 4 numbered contributions pointing to sections; the claim-level sentence; headline figure |
| 2 Method | 1.5 | The TTT layer; the naive carry; the carry design (one equation, one diagram) |
| 3 Evaluation | 1.0 | Models, paired regimes, fresh vs. development data, statistics |
| 4 Results | 5.5 | Naive carry gives nothing (short); designed carry within a document; relatedness on fresh data with controls; baselines (in context, dynamic evaluation, equal-memory cache); what is carried; no accumulation, chains and decay |
| 5 Related work | 1.0 | Positioning table plus the 2026 papers |
| 6 Limitations | 0.5 | Section 6.4 |
| 7 Conclusion | 0.25 | What was learned, for whom |
| Appendices | | Clip diagnosis; failed attempts; ablation; domains; development-set results; every configuration run; hyperparameters; compute; reproduction; LLM use |

**Section rules.**
- **Abstract:** say what was found, why some researchers care, and what readers learn [B1, R8].
- **Introduction:** one to three claims around one theme [R8]. State the scope here, not only in the
  limitations.
- **Related work:** check every "no prior work" sentence on arXiv on the day you submit. Write "to our
  knowledge" [T2] **(judgement)**.
- **Limitations:** a paper without honest limitations is "substantially weaker" [R8]. Include
  assumptions and few runs or datasets [R5].

---

## 5. Write clearly

TMLR wants readers to understand the setting, setup, findings and takeaways [B1]. It does not want
elegant prose [B1]. It calls AI writing "convoluted" and full of "unnecessary jargon" [B1].
**(judgement, consistent with [B1, R1])**:

1. One idea per paragraph, stated in its first sentence.
2. Define every term once (carry, slice, regime, between-slice gain, state ratio). Never rename it.
   One model name everywhere, including figures; no code names like `delta_chunk`.
3. Measured words only: "lowers perplexity by", "transfers", "interferes". Avoid "memory",
   "understands", "learns" [R1].
4. Only equations that say something the text cannot (about 3 for us) [R1].
5. No filler such as "importantly", "novel", "comprehensive", or "robust" without a number.
6. Every figure and table is cited in the text, and its message is stated in the text.

---

## 6. Evidence and statistics

### 6.1 Reporting
- Point estimate, 95 percent interval, n and win rate for every result. Say exactly what the interval
  is [R2, R5].
- Effect sizes over significance stars [R2, R10]. Do not use p < .05 as a threshold [R8].
- **For us:** the paired percentile bootstrap over documents (1000 resamples). Report headline effects
  as log-perplexity percent, with absolute perplexity second **(judgement)**.

### 6.2 Seeds and comparisons
- Seeds alone can produce different results [R6]. Report each seed and the mean. With 2 seeds, write
  "replicates in sign and significance", never "robust".
- Keep the paired design (same documents in every arm) [R4].
- Apply a Holm correction across the headline contrasts **(judgement)**.
- List every configuration run, failures included. The editors found p-hacking in an interviewed
  paper [B5].

### 6.3 Never claim more than the evidence

| Do not write | Write |
|---|---|
| "strongest model" | "largest lead over the matched baseline (0.234 [0.188, 0.293] vs. 0.203 [0.165, 0.249])" |
| "both changes are needed" | "in a single-seed ablation at step 200, the two changes appear to work together" |
| "contributes nothing" | "no measurable effect; intervals rule out effects above about 0.01" |
| "the carry remembers" | "a carry built from a related article lowers perplexity by ..." |
| "useful where attention cannot reach" | only if experiment 1.4 shows it |

### 6.4 Limitations our paper must state
- **Scale and training:** one 0.6B model; training stopped at 600 of 1050 steps; 2 or 3 seeds; the
  ablation is single-seed at step 200; only the Hebbian rule was redesigned.
- **Measurement:** perplexity only (and the claim level that follows); relatedness by title mention;
  Wikipedia only; possible overlap of the Wikipedia dump with training data.
- **Design and baselines:** the carry is trained only within documents; dynamic evaluation is sparse;
  any part of the boundary vs. relatedness conflict that remains unexplained.

### 6.5 Test-set hygiene
- Never tune on the fresh test set. Dynamic evaluation stays tuned on disjoint cases [R7].

---

## 7. Baselines and controls

Use the strongest baselines possible [R8], with no unexplained source of gains [R1], and disclose
each baseline's cost and memory [R9].

| Baseline or control | Answers |
|---|---|
| Reset per slice and per document | Is anything carried? |
| Matched LoRA-only model | How much is extra capacity? |
| Related article in context | How does the carry compare with keeping the text? |
| Equal-memory attention cache | Is the carry better per byte? |
| Dynamic evaluation, same model, disjoint tuning | Is a gradient-based carry better? |
| Generic carry and shuffled prime | Document-specific transfer, or a domain prior? |
| Naive carry on the same cases | Is the effect due to the design? |
| Ablation | Which change matters? |

---

## 8. Figures and tables

- A figure plus its caption must make sense on its own [R8].
- Put one headline figure in the introduction: the relatedness gradient on fresh data with controls,
  baselines and intervals **(judgement)**.
- Every plot shows intervals and says what they are [R5].
- Same names and colours everywhere, no titles inside figures, readable ticks, PDF format
  **(judgement)**.

---

## 9. Novelty and overlap

- Theses and preprints are fine [T4]. Never submit the same work elsewhere at the same time [T2].
- As of 27 October 2026, no paper carries a fast weight across independently bounded documents
  [P1, P2, P3, R9]. Search arXiv again on the submission day.
- Cite and distinguish these, one sentence each [P1, P2, P3, P4, R9]:
  - TTT-NTP;
  - continual fact learning in weights;
  - delayed supervision for test-time memory;
  - In-Place TTT;
  - Beyond Perplexity.

---

## 10. LLM use and the author interview

**Rules.**
- LLMs may assist. Authors are fully responsible, and LLMs cannot be authors [T2].
- Disclose LLM use in a footnote on page 1 [T4].
- TMLR expects the ideas, claims and results to be human-sourced [T4].
- An entirely AI-generated paper is "unlikely to meet" the criteria [B1].

**What to do.**
1. Write the final paper yourself, and use AI only for checking **(judgement)**.
2. Use an honest footnote, for example: "We used a large language model as an assistant for code,
   literature organisation and proofreading. All research questions, designs, analyses and claims
   are the authors' own, and we verified every result and citation." Edit it until it is exactly
   true.
3. Have a co-author. All three interviewed authors who failed basic questions were solo authors
   [B5].

**Interview (about 30 minutes)** [B5]. Expect basic questions on the setting, notation and results,
then detailed ones on technical choices. You must be able to explain from memory:
- the TTT read and write equations;
- why the Hebbian state grows;
- the clip and the state ratio;
- each regime and gain;
- the zero-initialised scale and the −6 gate bias;
- training through the carry and the gradient-checkpoint fix;
- how relatedness cases are built;
- the bootstrap;
- every number in the abstract;
- every limitation.

Study `plan/DEFENCE_QA.md`.

---

## 11. Submission mechanics

1. **Style file:** unmodified TMLR style file in anonymised mode [T3, T6]. Choose a standard PDF
   submission, not Beyond PDF [T6].
2. **Anonymisation** [T3]:
   - no names, acknowledgements, named GitHub links or identifying self-citations;
   - anonymous code mirror;
   - anonymised supplementary material.
3. **arXiv:** posting is allowed, using the preprint option, but do not link it from the submission
   [T3, T4].
4. **OpenReview profiles:** complete for all authors, with conflicts (3 years) [T2, T3].
5. **Quota:** check it for each author [B3].
6. **Authors:** the author list is final at submission [T2].
7. **Action Editors:** suggest AEs in test-time training, memory or efficient sequence models
   **(judgement)**.
8. **Ethics:** add a short paragraph noting that a state carried across users could leak
   information, so deployment needs per-user isolation [T7] **(judgement)**.

---

## 12. During review and after

**During review.**
- Read the AI review first and answer it point by point [B2].
- Answer every requested change. Fix each "critical" one with evidence or a reduced claim [T1, T5].
- Upload a revised PDF after all 3 reviews are in [T2], and respond within 2 weeks [T4]. Post a
  comment saying what changed [T4].
- Write the responses yourself [B5]. Concede what is true and show where you fixed it
  **(judgement)**.

**After the decision.**
- **Minor revision:** make the changes in the camera-ready version [T4].
- **Camera-ready:** de-anonymise, add the OpenReview link, upload the code [T2, T4, T6].
- **Reject:** resubmit as a new submission linking the old one and listing the changes [T2].
- **J2C certification:** about 10 percent of accepted papers get it. It allows presentation at
  NeurIPS, ICML or ICLR [B6].

---

## 13. Pre-submission checklist

**Claims**
- [ ] Every abstract and introduction claim is in the claims table
- [ ] Claim level stated, [R9] cited
- [ ] No phrasing from the table in 6.3
- [ ] Every item in the red-team list (3.4) ruled out or stated
- [ ] Headline results from the fresh set; analysis plan committed first
- [ ] Every number traces to `results/`

**Statistics and baselines**
- [ ] Estimate, interval (type stated), n, win rate everywhere
- [ ] Seeds shown separately and averaged; Holm correction on headline contrasts
- [ ] All Section 7 controls present with their costs

**Writing**
- [ ] One name per concept, including in figures
- [ ] Setting and setup clear from the main text alone
- [ ] At most about 12 main pages; limitations cover 6.4
- [ ] I wrote it and can defend every sentence

**Integrity**
- [ ] LLM footnote on page 1
- [ ] Co-author has read the paper
- [ ] Every configuration listed; compute reported, including failed runs
- [ ] Defence Q&A studied

**Mechanics**
- [ ] Unmodified style file, anonymised, no identifying links
- [ ] Supplementary anonymised, under 100 MB
- [ ] Profiles and conflicts complete; quota checked
- [ ] arXiv novelty search on the submission day; not under review elsewhere

---

## 14. Claim bank (update after Tier 1)

| Safe now | Only if Tier 1 or 2 supports it | Never |
|---|---|---|
| The naive carry has no measurable cross-document effect | The gradient holds on fresh data | Stores facts or remembers documents |
| The designed carry lowers perplexity between slices by about 0.1 (about 1 percent) through 600 steps, 2 seeds | Transfer beyond a domain prior | Helps where attention cannot reach (unless 1.4) |
| Same article > related > random same-source > other source (harm), 2 seeds | Beats an equal-memory cache | Best model, or beats dynamic evaluation overall |
| A long same-source chain causes small harm | A low-rank carry keeps the gain | Enables continual learning or personalisation |
| No accumulation with re-reading | Content, not style, carries the gain | Generalises to other models or domains |
| Recovers about 4 percent of the gain of the previous text in attention | Bridge-level storage (cloze) | Anything about a delta-rule design |

---

## 15. Sources

**TMLR**
- [T1] Acceptance criteria. https://jmlr.org/tmlr/acceptance-criteria.html
- [T2] Editorial policies. https://jmlr.org/tmlr/editorial-policies.html
- [T3] Author guide. https://jmlr.org/tmlr/author-guide.html
- [T4] FAQ. https://jmlr.org/tmlr/faq.html
- [T5] Reviewer guide. https://jmlr.org/tmlr/reviewer-guide.html
- [T6] Action editor guide. https://jmlr.org/tmlr/ae-guide.html
- [T7] Ethics guidelines. https://jmlr.org/tmlr/ethics.html
- [B1] Increasing Emphasis on Clear Writing, 28 Aug 2026. https://blog.tmlr.org/2026/increasing-emphasis-on-clear-writing-in-acceptance-criteria/
- [B2] AI Reviews for Assessing Soundness, 4 Jul 2026. https://blog.tmlr.org/2026/ai-reviews-at-tmlr-for-assessing-soundness/
- [B3] Annual Author Submission Quotas, 17 Jun 2026. https://blog.tmlr.org/2026/annual-author-submission-quotas-for-tmlr/
- [B4] Adapting Our Processes, 15 Oct 2026. https://blog.tmlr.org/2026/adapting-our-processes-to-the-changing-world-of-scientific-publication/
- [B5] Asking Authors About Their Own Papers, 16 Sep 2026. https://blog.tmlr.org/2026/asking-authors-about-their-own-papers/
- [B6] Journal-to-Conference Track, 21 Oct 2025. https://blog.tmlr.org/2025/tmlr-joins-neurips-icml-iclr-journal-to-conference-track/

**Reporting guidance**
- [R1] Lipton and Steinhardt, Troubling Trends in ML Scholarship, 2018. https://arxiv.org/abs/1807.03341
- [R2] Agarwal et al., Statistical Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264
- [R4] Bouthillier et al., Accounting for Variance, MLSys 2021. https://arxiv.org/abs/2103.03098
- [R5] NeurIPS Paper Checklist. https://neurips.cc/public/guides/PaperChecklist
- [R6] Henderson et al., Deep RL that Matters, AAAI 2018. https://arxiv.org/abs/1709.06560
- [R7] Kapoor and Narayanan, Leakage and the Reproducibility Crisis, 2023. https://arxiv.org/abs/2207.07048
- [R8] Nanda, Highly Opinionated Advice on How to Write ML Papers, 2025. https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers
- [R9] Song et al., Beyond Perplexity, arXiv 2607.00368, 2026. https://arxiv.org/abs/2607.00368
- [R10] Meister and Cotterell, Language Model Evaluation Beyond Perplexity, ACL 2021. https://arxiv.org/abs/2106.00085

**Related work**
- [P1] Ouyang et al., Test-Time Training with Next-Token Prediction, Findings of EMNLP 2026. https://arxiv.org/abs/2606.21803
- [P2] O'Neill et al., Can a Language Model Learn Facts Continually in Its Weights?, 2026. https://arxiv.org/abs/2607.11020
- [P3] Kim and Kothari, Delayed Supervision for Test-Time Language Models, 2026. https://arxiv.org/abs/2609.32312
- [P4] Feng et al., In-Place Test-Time Training, 2026. https://arxiv.org/abs/2604.06169
