# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Citation:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai — Google Research, arXiv:2305.13245v3, 23 Dec 2023.

## Important parts for my research
- Grouped-query attention (GQA): divides H query heads into G groups, each group sharing one key/value head; GQA-1 = multi-query attention (MQA), GQA-H = standard multi-head attention (MHA). GQA-8 is chosen as the favorable middle ground in their experiments.
- Uptraining recipe: convert an existing MHA checkpoint to MQA/GQA by mean-pooling the K/V projection matrices of each group (works better than selecting one head or random re-init, Fig. 4), then continue pretraining for a small fraction α (5%) of original pretraining compute to let the model adapt.
- Results (T5-XXL, summarization/QA/translation avg): MHA-XXL 47.2, uptrained MQA-XXL 46.6 at 6.3× lower inference latency (1.51s→0.24s/sample), uptrained GQA-8-XXL 47.1 (near-MHA quality) at 0.28s/sample — GQA closes most of the MQA quality gap at nearly MQA speed.
- MQA trained from scratch is unstable (frequent pretraining loss spikes, divergence on fine-tuning with long inputs); uptrained MQA is more stable but still high-variance; uptrained GQA is stable (Appendix A) — a concrete instability finding tied to reducing KV-head capacity too aggressively.
- Motivation is purely inference-time KV-cache memory bandwidth reduction (autoregressive decoding bottleneck), independent of TTT/fast-weights/continual learning.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology / architecture description) GQA-8 or GQA specifics if our experiments use a Qwen3 model variant with grouped-query attention — needed only to accurately describe the base architecture's attention mechanism, not for any TTT-related claim.
- (Related Work, minor) Uptraining-from-checkpoint technique (5% compute to adapt an existing checkpoint to a new attention structure) as a general precedent for cheap architectural adaptation via continued pretraining, if relevant to describing how a base Qwen3 checkpoint's attention differs from full multi-head.

## Verification
- Table 1 numbers (47.2/46.6/47.1 average, 1.51s/0.24s/0.28s latency) and the uptraining/mean-pooling procedure checked directly against Table 1, Figure 3, Figure 4, and Section 2.1 of the PDF.

## Note on relevance
This paper is an architecture-efficiency paper (attention head grouping for KV-cache reduction) with no direct connection to TTT, fast weights, hebbian/delta rules, carry, catastrophic forgetting, or dynamic evaluation. Its only relevance to this research is if the base model used (Qwen3) employs GQA, in which case a citation may be needed purely for architectural accuracy in a Methods section. No mechanism, number, or finding here bears on the research questions (Q1/Q2) or the null-result framing.
