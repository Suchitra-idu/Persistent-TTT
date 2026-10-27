# PLAN

Research plan for the session-persistent TTT project, from the current thesis to a TMLR submission.
Written 2026-10-27. Source of truth for every number is `results/`. The companion document
`plan/TMLR Paper writing guide.md` explains how the final paper must be written and why.

---

## 1. Working rules for the thesis phase

We work on the **thesis first** and write the **final TMLR paper last**.

1. **Imagine TMLR accepts thesis-size papers.** Treat the thesis as if it were the submission.
   Every claim must meet TMLR's two criteria (claims supported by accurate, convincing and clear
   evidence; findings of interest to some of TMLR's audience and communicated clearly). Length is
   not a constraint for the thesis, but clarity, evidence and honesty are judged exactly as TMLR
   would judge them.
2. **Whoever edits the thesis writes as the author, M A S I Malimbada, not as an AI.** Decisions,
   reasoning and voice are the author's. Write the way a careful undergraduate researcher who did
   this work would write it: plain, specific, first person plural ("we"), no generic filler, no
   inflated vocabulary, no hedging boilerplate. If something cannot be stated as the author would
   defend it in a 30-minute interview with a TMLR editor, it does not go in.
3. Existing rules still apply. No em dashes, en dashes or colons in thesis prose. Figures as PDF.
   Every number traces to a file in `results/`. Cite only works that have a note in `Notes/`
   (write the note first for any new paper).
4. LLM assistance is allowed by TMLR but must be disclosed in the final paper (footnote on the
   first page). The author must understand and be able to defend every sentence.

---

## 2. Where the project stands

- Thesis `paper/main.tex` rebuilt with a full literature review (70 references), audited, and
  corrected (claims, citations, mechanics). Old version is in git commit 992d756.
- Main result. A trained, gated fast-weight carry (Hebbian, Qwen3-0.6B) transfers across documents
  according to relatedness: same article +0.103, related +0.085, random same-source +0.051, other
  source −0.067 (seed 42; seed 43 replicates). Carried through a long chain of same-source documents
  it harms (−0.020, −0.031). It does not accumulate with re-reading. It recovers about 4 percent of
  the gain of the previous text in attention. The carry path's clip is still saturated.
- Weak points a reviewer will find, in order of danger:
  1. A generic "this is Wikipedia" carry might explain most of the random-article gain.
  2. Every design choice was made while looking at the same evaluation documents (selection bias).
  3. No in-context baseline on the relatedness cases; "where attention cannot reach" is untested.
  4. Boundary suite (harm) and relatedness suite (help) disagree on Wikipedia.
  5. Dynamic evaluation runs on a different model than the carry.
  6. "Related" is defined by title mention, so it is confounded with word overlap.
  7. Two seeds; ablation only at step 200 with one seed; training stopped at 600 of 1050 steps.
- TMLR context (details and sources in the guide). Desk rejection rose from 6 percent in 2023 to
  about 53 percent now; the editors desk reject when claims or evidence are unclear from the writing,
  when evidence does not meet the claims, or when the topic is out of scope. Every submission now
  gets an AI soundness review. The Editor-in-Chief has interviewed authors to test whether they
  understand their own papers. LLM use must be disclosed. Solo-authored submissions are under
  scrutiny and limited by quota.

---

## 3. The plan, by tier (ranked by value for money)

### Tier 0. Free, highest value (protects against desk rejection and the author interview)

| # | Item | Why |
|---|---|---|
| 0.1 | Final paper rewritten as a TMLR paper in the author's own words (TMLR style file, at most about 12 main pages, anonymised, appendices for the rest) | Clear writing is now part of the acceptance criteria and the main desk-reject ground |
| 0.2 | Defence question sheet (the 40 hardest questions, answers traced to `results/`), studied by the author | The EiC interviews authors; reviewers ask "why −6?" |
| 0.3 | Prof. Yapa as co-author and critical reader | Removes the solo-author signal, adds an experienced reader |
| 0.4 | State the claim level explicitly (stream or domain adaptation, plus bridge evidence only where measured), cite "Beyond Perplexity"; delete "keep improving on a user" style motivation | Prevents "evidence migration" from perplexity to memory claims |
| 0.5 | Claims-to-evidence table; list of every configuration run including failures; where −6 came from; LLM-use footnote; code and bundle release (anonymous link for review) | Turns the forking-paths and reproducibility objections into stated facts |
| 0.6 | Add missing related work (Beyond Perplexity; TTT-NTP; continual fact learning in weights; delayed supervision for test-time memory; AutoCompressors; Generative Adapter; Cartridges; MemoryLLM; Larimar; Infini-attention; LaCT), each with a note in `Notes/` first | Incorrect or missing novelty positioning is an explicit rejection ground |

### Tier 1. Cheap evaluation-only experiments, run together in one fresh-data pass (about $25 to $30)

| # | Item | Cost | Why |
|---|---|---|---|
| 1.1 | Generic carry (average of carried states over many random Wikipedia articles, used for every target) and shuffled related prime (same tokens, random order) | ~$3 | Tests the strongest alternative explanation of the main claim |
| 1.2 | Fresh untouched test set (Wikipedia shards 2 and 3; new SlimPajama holdout seed), analysis plan committed before running; key suites for both seeds and baselines; current results become the development set | ~$8 | Removes the selection-bias objection; becomes the headline |
| 1.3 | In-context baseline on the relatedness cases (prime in attention, same scored tokens) | ~$3 | Required control; states the memory comparison honestly |
| 1.4 | Long earlier text (8k to 16k tokens) against an equal-memory attention cache of its last tokens; carry stored in bf16 and at low rank (top 32, 64, 128 directions) | ~$4 | Supports "where attention cannot reach" or tells us to drop it; low rank could make the memory argument strong |
| 1.5 | Chain length (1 block, 1 article, chains of 2, 5, 10) and decay λ sweep (0.5, 0.9, 0.99) | ~$3 | Resolves the boundary versus relatedness contradiction |
| 1.6 | Dynamic evaluation on the carry model with its carry off, plus Krause's decay towards trained weights | ~$4 | Removes the different-model confound |
| 1.7 | Per-token gain split (tokens shared with the prime versus not) and mechanism logging (carry gate, carry scale, state cosine similarity) | ~$2 | Content versus style; the first look at mechanism |

**Decision gate after 1.1.** If the generic carry reproduces the random-article gain, reframe
honestly (same-source gain is a domain prior; only the related minus random gap is
document-specific). If it does not, the main claim is much stronger.

### Tier 2. Moderate cost, real but smaller value

| # | Item | Cost |
|---|---|---|
| 2.1 | Continuous relatedness dose-response (TF-IDF similarity bands within Wikipedia); only if 1.7 is ambiguous | ~$5 |
| 2.2 | Third training seed, key evaluations only, to report mean and spread | ~$15 |
| 2.3 | Fact cloze probe (answer log-probability against distractors, carry versus reset versus in context versus dynamic evaluation) to state the claim level from a measurement | ~$4 |
| 2.4 | Continue seed 42 to step 1050 (cross-document harm grows with training) | ~$8 to $10 |

### Tier 3. Skip and state as limitations

Training the carry across documents; redesigning the delta rule; a larger model; full-weight
dynamic evaluation; downstream benchmarks such as RULER.

### Expected acceptance chance (my estimate, not a measurement)

| State | Estimate |
|---|---|
| Current thesis submitted as is | 15 to 25 percent (mostly desk-reject risk) |
| Tier 0 only | 40 to 50 percent |
| Tier 0 and Tier 1, gate 1.1 favourable | 70 to 78 percent |
| Tier 0 and Tier 1, gate 1.1 shows a domain prior, honest reframe | 55 to 65 percent |
| Tier 2 on top | plus 3 to 5 points |

### Timeline (target: decision within 6 months, by end of April 2027)

| When | What |
|---|---|
| Weeks 1 to 2 (to mid November) | Supervisor ask; commit; notes for new papers; pre-registration; Tier 1 code |
| Weeks 2 to 3 | Tier 1 runs; gate 1.1; Tier 2 decisions |
| Weeks 3 to 6 | Thesis update and audit; defence sheet; supervisor review |
| Weeks 6 to 8 | Final TMLR paper written by the author from the guide |
| About mid December 2026 | Submit to TMLR; arXiv preprint |
| Late February to March 2027 | Decision (TMLR target is about 9 weeks; allow 13) |

---

## 4. TODO (in the order it should be done; tier in brackets)

Legend. `[ ]` open, `[x]` done. Owner is me (the author) unless marked (Claude) for code and checks.

### Done
- [x] Rebuild the literature review from all 69 notes [T0]
- [x] Audit the thesis (citations, numbers, consistency, mechanics) [T0]
- [x] Apply all text-only audit fixes to the thesis [T0]
- [x] Research TMLR and write `plan/TMLR Paper writing guide.md` [T0]
- [x] Write this plan [T0]

### Phase A. Set-up (week 1)
- [ ] 1. Email Prof. Yapa asking him to be co-author and critical reader [T0.3]
- [ ] 2. Commit all current code, thesis and results; tag `pre-fresh-set` so later runs record a clean commit [T0.5]
- [x] 3. Write `Notes/` entries for the new related papers (done 2026-10-27; PDFs in `New Papers/`, thesis updated) (Beyond Perplexity 2607.00368; TTT-NTP 2606.21803; continual fact learning in weights 2607.11020; delayed supervision 2609.32312; AutoCompressors; Generative Adapter; Cartridges; MemoryLLM; Larimar; Infini-attention; LaCT) [T0.6]
- [ ] 4. Write `plan/PREREGISTRATION.md` (suites, arms, contrasts, Holm correction across headline contrasts, decision rules for gate 1.1) and commit it before any fresh-set run [T1.2]

### Phase B. Tier 1 code (Claude, week 1 to 2)
- [ ] 5. Fresh test set builders: Wikipedia shards 2 and 3 for relatedness cases; new SlimPajama holdout seed for regimes, boundary and repeat [T1.2]
- [ ] 6. Generic-carry arm and shuffled-prime arm in the relatedness suite [T1.1]
- [ ] 7. In-context baseline arm (prime in attention, same scored tokens) [T1.3]
- [ ] 8. Per-token loss saving and shared-token gain split; mechanism logging of carry gate, carry scale and state cosine similarity [T1.7]
- [ ] 9. Chain-length and λ sweep suite [T1.5]
- [ ] 10. Long-prime suite against an equal-memory attention cache; bf16 and low-rank carry options [T1.4]
- [ ] 11. Dynamic evaluation on the carry model with carry off; Krause decay variant [T1.6]
- [ ] 12. Unit tests with fakes, comment-budget test, local smoke run; cost estimate per suite for approval

### Phase C. Runs and analysis (week 2 to 3)
- [ ] 13. Approve the budget; run all Tier 1 suites on Modal for seeds 42 and 43 plus baselines, sharded, from the clean commit [T1]
- [ ] 14. Analyse; regenerate tables and figures into `results/paper/`; update `FINDINGS.md` [T1]
- [ ] 15. Decision gate 1.1: keep the claim or reframe as domain prior plus document-specific gap [T1.1]
- [ ] 16. Decide and run any Tier 2 items the budget allows, in this order: 2.3 cloze, 2.1 dose-response, 2.2 third seed, 2.4 to step 1050 [T2]

### Phase D. Thesis (week 3 to 6)
- [ ] 17. Update the thesis: fresh-set results as the headline, development set in an appendix, new baselines and controls, explicit claim level, new related work, claims-to-evidence table, full configuration list, provenance of −6 [T0.4, T0.5, T0.6]
- [ ] 18. Re-audit the thesis (every number against `results/`, every citation against `Notes/`, no stale claims) [T0]
- [ ] 19. Write and study the defence question sheet `plan/DEFENCE_QA.md` [T0.2]
- [ ] 20. Supervisor reads the thesis; revise from his comments [T0.3]

### Phase E. Final paper and submission (week 6 to 8)
- [ ] 21. Prepare the release: clean repository, reproduce-from-bundles README, bundle archive with a DOI, anonymous mirror for review [T0.5]
- [ ] 22. Write the final TMLR paper myself, following `plan/TMLR Paper writing guide.md` [T0.1]
- [ ] 23. Pre-submission checks from the guide: final arXiv novelty search, LLM-use footnote, anonymisation, unmodified style file, quota check, OpenReview profiles complete [T0]
- [ ] 24. Submit to TMLR; post the arXiv preprint with the style file's preprint option [T0]
- [ ] 25. Answer reviews and post revisions within 2 weeks of the third review [T0]

---

## 5. Sources for the TMLR facts above

Full source list with quotes is in `plan/TMLR Paper writing guide.md`. Key ones:
- TMLR acceptance criteria, https://jmlr.org/tmlr/acceptance-criteria.html
- TMLR editorial policies, https://jmlr.org/tmlr/editorial-policies.html
- TMLR FAQ, https://jmlr.org/tmlr/faq.html
- Adapting Our Processes (15 Oct 2026), https://blog.tmlr.org/2026/adapting-our-processes-to-the-changing-world-of-scientific-publication/
- Asking Authors About Their Own Papers (16 Sep 2026), https://blog.tmlr.org/2026/asking-authors-about-their-own-papers/
- Increasing Emphasis on Clear Writing (28 Aug 2026), https://blog.tmlr.org/2026/increasing-emphasis-on-clear-writing-in-acceptance-criteria/
- Beyond Perplexity (arXiv 2607.00368), https://arxiv.org/abs/2607.00368
