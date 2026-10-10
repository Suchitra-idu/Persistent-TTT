Here's the full plan. It uses only the checkpoints you already have and the LoRA-only run in progress. Nothing in it assumes new TTT training.

## Phase 0: Housekeeping (no GPU)
1. **Commit the eval-mode fix.**
2. **Find the delta_chunk checkpoint.** `results/ttt-v1.1-clean-deltachunk/` exists locally, but that name doesn't show up at the top level of the volume.
3. **Write down the flags for every run you'll cite:** `ttt-v1.1`, delta_chunk, `lora-only-matched`, and `lora-only-highlr`. The paper_eval manifests already record them; build one table from those.
4. **Freeze the code version used for the paper's evals** (a git tag), so every number traces back to one commit.

## Phase 1: Evals on existing checkpoints (GPU, but cheap)
5. **Matched LoRA-only control:** run paper_eval on `lora-only-matched` at 600, using the regimes suite only (it has no TTT layers). This is your headline comparison: TTT + LoRA vs. LoRA alone, same LR, same step.
6. **Sanity check:** on the LoRA-only checkpoint, Δwithin and Δbetween should come out at 0.000. Report it in the paper as a check that the eval pipeline is sound.
7. **Training curve:** run paper_eval on `ttt-v1.1` at 100, 200, 400, 600 and 800, and on `lora-only-matched` at the same steps. Plot each gain against step: LoRA, within-document TTT, cross-document carry. This answers "it's just undertrained" without new training.
8. **Bigger eval set:** the current manifest has 20 docs per source, except Book with only 6. Raise it to 30 or more per source, and find out why Book falls short (probably document length filtering). Rerun the step-600 evals at that size.
9. **Delta_chunk at 600** on the same, larger eval set, so the two update rules are compared on identical docs.
10. **Optionally, `lora-only-highlr` at 300 and 600** as a "LoRA at a normal LR" reference row.

## Phase 2: Analysis (no GPU, from the result bundles)
11. **Gain breakdown table:** fresh → LoRA → within-document TTT → cross-document carry, with 95% CIs, per source and pooled. This is the paper's main table.
12. **TTT vs. control:** total perplexity of `ttt-v1.1` vs. `lora-only-matched`, at each step, with CIs.
13. **Why persistence fails.** Use the state ratio over a session and across documents. It roughly measures how large the carried state is relative to the weights it modifies. Decide which case applies:
    - The carry is negligible, so it's effectively a no-op.
    - The carry is large but holds nothing transferable.
    - The carry is overwritten or decays within a few chunks.

    This section is what makes the result count as more than "it didn't work".
14. **Repeat and swap as evidence for 13:**
    - Replaying the same document 9 times does no better than once (`repeat_9_vs_1` = 0).
    - A carry from the same source vs. another source makes almost no difference.

    Together these say the carry stores nothing reusable.
15. **Where in a document TTT helps:** the within-document gain against token position or slice index. Is there a warm-up length before it helps at all?
16. **Per-source pattern:** does the TTT gain follow the LoRA gain (Book and C4 high, Github low), or differ from it? Domain dependence is one of your stated objectives (O3).
17. **Hebbian vs. delta_chunk:** delta_chunk gets more within-document gain (+0.058 vs. +0.034) and a significant cross-slice gain (+0.015). Report it honestly: the better update rule helps within a document and still gives nothing across documents.

## Phase 3: Training, only if money appears (ranked)
18. **A second seed** of `ttt-v1.1` and `lora-only-matched`. Reviewers often ask for this.
19. **A second model size** (1.7B), at a short schedule.

If you skip both, list them as limitations. That's acceptable for Q2 if stated plainly.

## Phase 4: Writing
20. **Framing:** "Where does the gain in test-time training for language models come from? An attribution study of session-persistent fast weights." Present the null result for persistence as a finding, not as the method failing.
21. **Contributions to claim:**
    - The method: session-persistent In-Place TTT with per-source carries.
    - An evaluation protocol that separates LoRA, within-document and cross-document effects.
    - The finding: the gain is almost all LoRA, TTT adds a small within-document gain, and persistence adds nothing.
    - The diagnosis of why.
22. **Related work:**
    - TTT-Linear and TTT-MLP (Sun et al. 2024), In-Place TTT (Feng et al. 2026), Titans.
    - Fast weights and linear attention (Schlag et al.), continual and online learning for language models.
    - Negative-results papers, for precedent.
23. **Threats-to-validity section:**
    - Short training (600–847 steps) and a single model size (0.6B).
    - One seed, if item 18 is skipped, and RedPajama sources only.
    - The LoRA LR was deliberately lowered in the TTT runs.
    - The dropout bug: state that it affected only the in-loop eval, not paper_eval.
24. **Figures:**
    - Gains vs. step (from item 7).
    - The breakdown as a stacked bar per source.
    - State ratio over a session.
    - The repeat curve.
25. **Reproducibility appendix:** all flags, the git commit, the result bundles, and a code link.

## Phase 5: Submission
26. **Venues:** pick 2–3 Q2 journals with your supervisor. Check Scimago vs. JCR quartiles and which one your institution counts. Prefer journals that accept empirical or negative-result papers.
27. **Fallback:** the NeurIPS ICBINB workshop or the ACL Insights from Negative Results workshop. Check whether a workshop paper would block a later journal submission.

---

**Critical path:** items 1, 5, 6, 7, 8, 11, 12, 13 and 14. That's about 12–15 paper_eval runs and no new training. Everything else makes the paper stronger, but those nine items are what make it a credible Q2 submission.

I can write this into PLAN.md whenever you're ready, or prepare the paper_eval commands for Phase 1.