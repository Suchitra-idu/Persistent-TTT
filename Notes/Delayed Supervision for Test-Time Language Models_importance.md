# Delayed Supervision for Test-Time Language Models

**Citation:** Jinha Kim, Taksh Kothari (MIT CSAIL). arXiv:2609.32312v1 [cs.AI], 26 Sep 2026. Preprint.

## Important parts for my research
- Observes that next-token training "does not explicitly require a fact to remain accessible after many subsequent memory updates" in test-time memories (LaCT nonlinear fast weights, DeltaNet delta rule, RWKV-7).
- Adds a **delayed QA loss** during post-training, L = αL_NTP + βL_QA.
  - A question about the simulator state is asked "only after a long interval of unrelated events": at least 4,096 tokens and 16 intervening writes. DeltaNet gaps run 4,105 to 20,354 tokens (median 10,099).
  - Questions run on **disposable branches**, so answers never enter the stream.
  - The probes separate **retention** (a fact stays valid) from **revision** (answer with the latest value).
- "The native test-time update rule is unchanged: delayed supervision trains the slow parameters that shape writing, retention, and reading" (§1). Gradients from the delayed answer reach earlier writes: "the answer loss can train how evidence is encoded and survives subsequent updates" (§3.2).
- Data: TextWorld trajectories, 512 worlds, 128 outer updates.
- **Results** (Table 1, delayed QA vs. event-only NTP):

  | Model | BABILong | RULER single-needle |
  |---|---|---|
  | LaCT-760M | 31.08 vs. 25.60 | 88.85 vs. 87.39 |
  | DeltaNet-1.3B | 22.48 vs. 21.16, paired interval [−0.16, 2.84] | 83.52 vs. 80.24, [1.70, 4.97] |
  | RWKV-7 (separate panel) | +4.60 | +7.00 |

  - DeltaNet regresses at 1K and 2K. On the complementary panel, gains are "selective transfer rather than a general increase".
- Honest scoping: "The present comparisons do not identify delay as the sole cause of improvement". One training seed, and the intervals capture example variation, not seed variation.
- All persistence is within one continuous event stream, not across documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / Discussion) Kim and Kothari (2026) argue that next-token training gives a weak signal for keeping information through many later memory updates. They add a delayed loss whose gradient reaches earlier writes, the same principle as our training through the carry, applied within one stream rather than across documents.
- (Discussion) Their retention vs. revision split is a useful frame for our carry: we measure transfer, not retention of specific facts.
- (Methodology) Precedent for honestly scoping a single-seed result whose cause is not isolated.

## Verification
- Read the full main text (§1–5.3, PDF pages 1–7) via pdftotext. All quoted sentences, delay thresholds, Table 1 scores, the DeltaNet bootstrap intervals and the per-length remarks were read directly. The related work, discussion and appendix after §5.3 were not read.
