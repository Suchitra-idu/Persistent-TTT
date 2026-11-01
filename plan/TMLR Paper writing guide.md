# TMLR Paper Writing Guide

How to write the final paper so it survives TMLR's desk screen, the AI soundness review, three
reviewers and the Action Editor. Every rule has a source tag (Section 12). **(judgement)** marks
advice rather than a sourced fact. Facts checked against the sources on 2026-11-01.

No guide guarantees acceptance. The remaining risk is reviewers finding the evidence too weak,
which only the Tier 1 experiments in `plan/PLAN.md` reduce. Before submitting, read 5 to 10 recent
TMLR reviews on related topics on OpenReview.

---

## 1. How TMLR decides

**Two criteria.**
1. Are the claims supported by accurate and convincing evidence [T1]?
2. Would some individuals in TMLR's audience be interested, and are the findings communicated
   clearly [T1, B1]?

Novelty and state of the art are not required [T1]. If claims exceed evidence, add experiments
**or reduce the claims** [T1].

**Who reads it, in order.** Editors-in-Chief screen first and make about 90 percent of desk
rejections, mostly when claims or evidence are unclear from the writing, or when the paper looks
fully AI-written [B4]. The Action Editor checks within a week for anonymisation, an unmodified
style file, scope, technical depth and overlap with published work [T6]. An advisory AI review
checks claims against evidence, and authors may answer it once [B2]. At least three reviewers
answer the two criteria and mark requested changes as critical or not [T5]. The AE decides
accept, accept with minor revision, or reject; there is no major revision [T6].

| Fact | Value | Source |
|---|---|---|
| Desk rejection | about 53 percent now (about 6 percent in 2023) | [B4] |
| Length | no limit; over 12 main pages, reviewers get 4 weeks instead of 2 | [T4] |
| Decision | target about 9 weeks; reported medians about 13 weeks (2023, 2025) | [T4, S1, S2] |
| Your response | within 2 weeks of the third review | [T4] |
| Supplementary | up to 100 MB, anonymised | [T3] |
| Quota | about 2 solo-authored papers per author per year; desk rejects count | [B3] |

---

## 2. Before writing, fix the claims

1. **Claims-to-evidence table first.** One row per claim: claim, experiment, effect with interval,
   n, seeds, scope. Nothing goes in the abstract or introduction that is not a row. Put the table
   in the paper [R5] **(judgement)**.
2. **State the claim level** [R9]. Ours is S-level: stream or domain adaptation measured by
   perplexity. Never write that the carry "remembers", "stores" or "learns facts". Using
   perplexity to support a memory story is what [R9] calls "evidence migration".
3. **State the scope in the abstract.** The main carry is trained only within documents, so
   cross-document use is a transfer test; one single-seed probe is trained across boundaries.
4. **Separate explanation from speculation** [R1]. A mechanism story without an experiment is
   labelled "a plausible reading, not tested here". Clip saturation is an observation in both the
   naive and the trained carry, not the cause of the naive carry's failure.
5. **Red-team list.** Each is ruled out by an experiment or stated **(judgement)**:
   - a generic domain prior instead of document-specific transfer (generic carry);
   - word overlap instead of relatedness (stated: "related" means title-linked, overlap included);
   - extra parameters (matched LoRA);
   - the prime simply being usable in attention (in-context baseline);
   - selection bias from developing on the test set (fresh preregistered set);
   - seed luck (each seed shown);
   - the boundary harm vs. relatedness help conflict (chain length);
   - undertraining, with the cross-document effect still drifting at step 600 (stated);
   - Wikipedia evaluation text present in training (title and hash check);
   - the carry never trained across documents (mixed-session probe, one seed; scope).
6. **Preregister.** Commit and push the analysis plan before running the fresh set; the decision
   rules in `plan/PLAN.md` §4 apply whatever the outcome [R7] **(judgement)**.

---

## 3. Structure

- Unmodified TMLR style file; a modified one is a clear desk reject [T6].
- At most 12 main pages, so reviews take 2 weeks [T4]. Appendix after the references [T3].
- Reviewers may skip appendices [T3], so everything a main claim needs is in the main text.

| Section | Pages | Content |
|---|---|---|
| Title | | Plain and specific. States carry, documents, transfer or interference |
| Abstract | 0.3 | Question, scope sentence, main finding with numbers, key control, claim level |
| 1 Introduction | 1.25 | TTT layers reset at every document; we remove the reset; question; four numbered contributions pointing to sections; claim-level sentence; headline figure |
| 2 Method | 1.25 | The TTT layer; the naive carry; the carry path (one equation, one diagram) |
| 3 Evaluation | 1.0 | Model, suites, fresh vs. development data, statistics |
| 4 Results | 5.0 | Naive carry gives nothing; carry within a document; relatedness on fresh data against the generic carry and the in-context baseline; chain length; no accumulation; mixed-session probe (short) |
| 5 Related work | 1.0 | Positioning table (Section 8) |
| 6 Limitations | 0.5 | Section 5.3 |
| 7 Conclusion | 0.25 | What was learned, for whom |
| Appendices | | Ablation; failed attempts; delta rule and capacity attribution; dynamic evaluation (development set, no claim); development-set results; every configuration run; hyperparameters; compute; reproduction |

- **Introduction:** one theme, the claims of Section 2 only [R8].
- **Related work:** every "no prior work" sentence is checked on arXiv on the submission day and
  written as "to our knowledge" **(judgement)**.
- **Limitations:** honest limitations strengthen a paper [R8]. Include assumptions, few runs and
  few datasets [R5].

---

## 4. Write clearly

TMLR wants readers to understand the setting, setup, findings and takeaways; it does not want
elegant prose; it calls AI writing "convoluted" with "unnecessary jargon" [B1].
**(judgement, consistent with [B1, R1])**:

1. One idea per paragraph, stated in its first sentence.
2. Define each term once (carry, slice, suite, between-slice gain, generic carry) and never rename
   it. One model name everywhere, including figures; no code names like `delta_chunk`.
3. Measured words only: "lowers perplexity by", "transfers", "interferes". Not "memory",
   "understands", "learns" [R1].
4. Only equations that say something the text cannot (about 3).
5. No "importantly", "novel", "comprehensive", or "robust" without a number.
6. Every figure and table is cited in the text, with its message stated there. A figure plus its
   caption makes sense on its own [R8]; every plot shows intervals and says what they are [R5].

---

## 5. Evidence and statistics

### 5.1 Reporting
- Point estimate, interval (type stated), n and win rate for every result [R2, R5].
- Report effects and intervals, not p-values **(judgement)**.
- **Metric:** Δppl, as in every existing table; percent of the reference perplexity in parentheses
  for headline numbers.
- **Intervals:** paired percentile bootstrap over documents, 10,000 resamples, valid because each
  related prime is used once. The four headline contrasts also get Bonferroni-adjusted intervals.
- Show each seed [R6]. With 2 seeds, write "replicates in sign and in interval", never "robust".
- Keep the paired design (same documents in every arm) [R4].
- List every configuration run, failures included **(judgement)**; the EiCs found a p-hacking
  workflow while interviewing one author [B5].
- Never tune on the fresh test set [R7].

### 5.2 Never claim more than the evidence

| Do not write | Write |
|---|---|
| "strongest model", "largest lead" | "leads the matched baseline by estimate [interval]" |
| "both changes are needed" | "in a single-seed ablation at step 200, the two changes appear to work together" |
| "contributes nothing" | "no measurable effect; the interval rules out effects above ..." |
| "the carry remembers" | "a carry built from a related article lowers perplexity by ..." |
| "the naive carry fails because the clip saturates" | "the naive carry has no measurable effect; its clip is saturated, as is the trained carry's" |
| "useful where attention cannot reach" | nothing; not tested |
| "transfers more than dynamic evaluation" | nothing; the baseline is in an appendix with no claim |

### 5.3 Limitations the paper must state
- **Scale:** one 0.6B model; 2 seeds; training stopped at step 600 of 1050 with the
  cross-document effect still changing; ablation single-seed at step 200; Hebbian only.
- **Scope:** main carry trained only within documents; training across documents tested by one
  single-seed probe with one mixing ratio; no recall or downstream evaluation, so S-level only.
- **Measurement:** perplexity only; relatedness by title mention, which includes word overlap; Wikipedia only for relatedness;
  possible residual overlap with pretraining data.
- **Anything** of the boundary vs. relatedness conflict that remains unexplained.

---

## 6. Baselines and controls

Use the strongest baselines available, with no unexplained source of gains [R1], and give each
one's cost and memory [R9].

| Baseline or control | Answers |
|---|---|
| Reset per document | Is anything carried? |
| Matched LoRA-only model | How much is extra capacity? |
| Prime in context (fits in the carry's memory budget) | How does the carry compare with keeping the text? |
| Generic carry, norm-matched | Document-specific transfer, or a domain prior? |
| Naive carry on the same cases | Is the effect due to the carry path? |
| Ablation | Which change matters? |

---

## 7. Headline figure

One figure in the introduction: the relatedness gradient on fresh data with the generic carry and
the in-context baseline, both seeds, intervals. Same names and colours everywhere, no titles inside
figures, readable ticks, PDF **(judgement)**.

---

## 8. Novelty and overlap

- Theses and preprints do not block submission [T4]. Never submit the same work elsewhere at the
  same time [T2].
- **Do not claim** "no paper carries state across documents". Recurrent and state-space models
  pass state across packed documents unless it is reset [P5]; related vs. unrelated earlier
  documents in attention are studied in pre-training [P6]; TTT on related text is known to help
  [P7].
- **Claim, to our knowledge:** no work gives a TTT fast weight a trained carry path and measures
  its transfer across document boundaries by relatedness. Search arXiv again on the submission day.
- One sentence each distinguishing: In-Place TTT [P4]; TTT-NTP [P1]; continual facts in weights
  [P2]; delayed supervision [P3]; state passing [P5]; sequence composition [P6]; TTT on nearest
  neighbours [P7]; dynamic evaluation; Beyond Perplexity [R9].

---

## 9. LLM use and defending the paper

**Rules.** LLMs may assist; authors are fully responsible; LLMs cannot be authors [T2]. Disclose
LLM use in a footnote on the first page [T4]. TMLR expects ideas, claims and results to be
human-sourced [T4]. An entirely AI-generated paper is unlikely to meet the criteria [B1].

**What to do.**
1. Write the paper yourself, using the thesis as material and AI for code and checking
   **(judgement)**.
2. A footnote that is true. Starting point, to be edited until every word is accurate:
   "We used a large language model (name, version) to write and test experiment code, run and
   monitor experiments, organise literature notes, draft and edit a thesis version of this work,
   suggest experiments and framings, and check the manuscript against the result files. The
   authors chose the research question, the experiments and the claims, wrote this paper, and
   verified every number and citation."
3. Co-authors are people who contributed. Do not add one to avoid scrutiny of solo papers
   **(judgement, consistent with [B4])**.

**Be able to explain from memory** **(judgement)**: the TTT read and write equations; why the
Hebbian state grows; the clip and the state ratio; each suite and gain; the zero-initialised scale
and the −6 gate bias and where −6 came from; training through the carry and the
gradient-checkpoint fix; how relatedness cases and the generic carry are built; the paired
bootstrap; every number in the abstract; every limitation. In a one-off trial in August 2026 the
EiCs interviewed authors of papers already slated for desk rejection; the three who could not
answer basic questions were all solo authors [B5]. Study `plan/DEFENCE_QA.md`.

---

## 10. Submission and review

1. Unmodified TMLR style file, anonymised mode, standard PDF submission [T3, T6].
2. Anonymisation: no names, acknowledgements, named GitHub links or identifying self-citations;
   anonymous code mirror; anonymised supplementary [T3].
3. arXiv is allowed, with the style file's preprint option; do not link it from the submission
   [T3, T4].
4. OpenReview profiles complete for all authors, with conflicts [T2, T3]. Quota checked [B3]. The
   author list is final at submission [T2].
5. A broader impact statement only if there is a real risk [T7] **(judgement)**.
6. During review: answer the AI review point by point [B2]; answer every requested change and fix
   each critical one with evidence or a reduced claim [T1, T5]; upload a revised PDF after all
   three reviews and respond within 2 weeks [T2, T4]. Write the responses yourself
   **(judgement)**.
7. Minor revision: the changes go in the camera-ready version, which is final [T4]. Reject:
   resubmit as a new submission linking the old one and listing the changes [T2].

---

## 11. Pre-submission checklist

**Claims**
- [ ] Every abstract and introduction claim is in the claims table
- [ ] Claim level and scope sentence in the abstract; [R9] cited
- [ ] No phrasing from the left column of 5.2
- [ ] Every red-team item (2.5) ruled out or stated
- [ ] Headline results from the fresh set, analysed exactly as preregistered
- [ ] Every number traces to `results/`

**Statistics and baselines**
- [ ] Estimate, interval (type stated), n, win rate everywhere
- [ ] Seeds shown separately; adjusted intervals on the four headline contrasts
- [ ] All Section 6 controls present with their costs

**Writing**
- [ ] One name per concept, including in figures
- [ ] Setting and setup clear from the main text alone
- [ ] At most 12 main pages; limitations cover 5.3
- [ ] I wrote it and can defend every sentence

**Integrity and mechanics**
- [ ] LLM footnote on page 1, every word true
- [ ] Every configuration listed; compute reported, including failed runs
- [ ] Unmodified style file, anonymised, no identifying links; supplementary under 100 MB
- [ ] Profiles and conflicts complete; quota checked
- [ ] arXiv novelty search on the submission day; not under review elsewhere

### Claim bank (update after Tier 1)

| Safe now (development set) | Only if the fresh set supports it | Never |
|---|---|---|
| The naive carry has no measurable cross-document effect | Related minus generic carry > 0 ("document-specific") | Stores facts, remembers, or recalls documents |
| The carry path lowers perplexity between slices by about 1 percent, 2 seeds | Related > random holds on fresh data | Useful where attention cannot reach |
| Same > related > random same-source > other source (harm), 2 seeds | Harm grows with chain length | Continual learning or personalisation |
| A long same-source chain causes small harm | Training across documents removes the harm (single-seed probe, preregistered rule) | Beats dynamic evaluation; best model; topic rather than word overlap; the carry learns when to apply itself (from one seed) |
| No accumulation with re-reading | | Generalises to other models, domains or longer training |
| Recovers about 4 percent of the in-context gain | | The clip causes the naive carry's failure |

---

## 12. Sources

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
- [S1] Kamath, TMLR 2023 numbers (median 91.4 days for conference-length papers). https://x.com/thegautamkamath/status/1780695733928358007
- [S2] phdflow, TMLR explained (quotes the 2025 annual report, median 91 days; third party). https://phdflow.ai/guides/tmlr-explained

**Reporting guidance**
- [R1] Lipton and Steinhardt, Troubling Trends in ML Scholarship, 2018. https://arxiv.org/abs/1807.03341
- [R2] Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264
- [R4] Bouthillier et al., Accounting for Variance in ML Benchmarks, MLSys 2021. https://arxiv.org/abs/2103.03098
- [R5] NeurIPS Paper Checklist. https://neurips.cc/public/guides/PaperChecklist
- [R6] Henderson et al., Deep RL that Matters, AAAI 2018. https://arxiv.org/abs/1709.06560
- [R7] Kapoor and Narayanan, Leakage and the Reproducibility Crisis, 2023. https://arxiv.org/abs/2207.07048
- [R8] Nanda, Highly Opinionated Advice on How to Write ML Papers, 2025. https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers
- [R9] Song et al., Beyond Perplexity, arXiv 2607.00368, 2026. https://arxiv.org/abs/2607.00368

**Related work**
- [P1] Ouyang et al., Test-Time Training with Next-Token Prediction, Findings of EMNLP 2026. https://arxiv.org/abs/2606.21803
- [P2] O'Neill et al., Can a Language Model Learn Facts Continually in Its Weights?, 2026. https://arxiv.org/abs/2607.11020
- [P3] Kim and Kothari, Delayed Supervision for Test-Time Language Models, 2026. https://arxiv.org/abs/2609.32312
- [P4] Feng et al., In-Place Test-Time Training, 2026. https://arxiv.org/abs/2604.06169
- [P5] Gu and Dao, Mamba, COLM 2024 (state reset or passing at packed-document boundaries). https://arxiv.org/abs/2312.00752
- [P6] Zhao et al., Analysing the Impact of Sequence Composition on Language Model Pre-Training, ACL 2024. https://aclanthology.org/2024.acl-long.427/
- [P7] Hardt and Sun, Test-Time Training on Nearest Neighbors for Large Language Models, ICLR 2024. https://arxiv.org/abs/2305.18466
