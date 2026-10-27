# Continual Learning with Deep Generative Replay

**Citation:** Hanul Shin, Jung Kwon Lee, Jaehong Kim, Jiwon Kim — NIPS 2017 (31st Conference on Neural Information Processing Systems), arXiv:1705.08690v3.

## Important parts for my research
- Defines catastrophic forgetting per McCloskey & Cohen (1989): performance on old tasks "abruptly degrades" when a network is trained on a new task — the canonical citation for the phenomenon our flat/non-compounding carry result is adjacent to.
- Deep Generative Replay (DGR): a "scholar" model = generator (GAN) + solver. Instead of storing real past data, the generator synthesizes pseudo-inputs from past task distributions; a solver's response to a generated input becomes the pseudo-target. Loss for solver i: L = r·E_{(x,y)~Di}[L(S(x;θi),y)] + (1−r)·E_{x'~Gi-1}[L(S(x';θi), S(x';θi-1))] (Eq. 1) — training loss mixes real current-task data and replayed synthetic old-task data at ratio r.
- Key empirical result: on MNIST permutation tasks (independent, non-overlapping input distributions), a naively sequentially-trained solver "None" collapses to near-chance on old tasks, while generative replay (GR) tracks the "exact replay" (ER) upper bound almost exactly. Sequential scholar chain (Solver1→Solver5) loses only ~0.25% accuracy total (98.81%→98.56%) after 5 tasks.
- Random-noise replay ("Noise" condition — pseudo-inputs are Gaussian noise paired with old-solver responses) fails to prevent forgetting — meaningful reconstruction of the *input* distribution, not just target distribution, is necessary for retention (Sec 4.1, 4.3 disjoint-class experiment).
- Compares against EWC (elastic weight consolidation — protects important weights via Fisher-Information-weighted regularization) and LwF (Learning without Forgetting — regularizes on the network's own pre-finetuning response to new-task inputs). Both are alternatives to replay; the paper frames them as complementary, not mutually exclusive, contributions to retention.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Canonical definition/citation for catastrophic forgetting (McCloskey & Cohen, 1989, as summarized here) when introducing the concept that motivates our carry-reset baseline.
- (Discussion — explains flat/non-compounding carry) The Noise-replay negative result — that replay/retention mechanisms fail when the recreated signal doesn't capture the real underlying distribution — is a useful analogy for why a carried hebbian fast weight might plateau rather than compound: if the "memory trace" a fast weight accumulates does not faithfully reconstruct informative structure (as opposed to being a generic accumulation), stability can appear at a low, flat level instead of growing forgetting or growing benefit.
- (Related Work) EWC/LwF as classical parameter-protection alternatives to episodic/generative replay for continual learning — useful contrast when situating session-persistent TTT (a form of implicit, decay-free replay via weight carry) against the standard continual-learning toolkit.

## Verification
- All numbers (98.81%→98.56% across 5 scholars, Eq. 1 loss formulation, Noise vs GR vs ER vs None conditions) checked directly against Table 1, Eq. 1, and Sections 3.1, 4.1, 4.3 of the PDF text.
- This paper does not address TTT, fast weights, or LLMs at all — it is included purely for the catastrophic-forgetting/replay framing; no claims here should be read as language-model-specific.
