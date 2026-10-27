# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Citation:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen, NeurIPS 2023 (arXiv:2306.14048v3).

## Important parts for my research
- Not a fast-weight/TTT paper — this is a KV-cache eviction policy for inference efficiency. Relevance here is as a contrasting approach to the same underlying problem (making long-context/multi-document inference tractable) via activation caching rather than parametric fast weights.
- Core mechanism: attention score matrices are >95% sparse even in densely-trained LLMs (measured on OPT family via Wiki-Text-103); a small set of tokens ("Heavy Hitters", H2) accumulate most of the attention mass and follow a power-law distribution. Removing H2 tokens causes severe accuracy collapse, confirming their functional importance.
- H2O eviction policy: dynamically retains a budget-k KV cache split between H2 tokens (by cumulative attention score, computed greedily from local/past statistics only — no future lookahead needed) and the most recent tokens. Formalized as a "dynamic submodular" maximization problem with a near-optimality guarantee (Theorem 4.4, informal: greedy top-k selection achieves (1-α)(1-1/e) of optimal minus β).
- With only 20% KV cache budget, H2O matches full-cache accuracy across OPT/LLaMA/GPT-NeoX (6.7B–66B) on COPA, OpenBookQA, PiQA, Winogrande, MathQA, RTE, XSUM, CNN/DailyMail — i.e., 5–10x memory reduction with no accuracy loss on these tasks.
- Throughput: up to 29x over DeepSpeed Zero-Inference, 29x over HuggingFace Accelerate, 3x over FlexGen (OPT-6.7B/30B, T4/A100 GPUs); latency reduced up to 1.9x at matched batch size vs FlexGen.
- Only-H2 or only-recent-tokens ablations both underperform full KV by 2.85–22.75%; combining both is necessary — recency alone (the "Local" strategy) collapses badly at high eviction rates (e.g., LLaMA-13B/XSUM, LLaMA-7B/CNN-DailyMail at 60% budget).
- Streaming result: H2O handles inputs up to 4 million tokens with lower perplexity than StreamingLLM's attention-sink method at matched cache size (tested on PG-19).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Efficient long-context inference has largely pursued activation-level compression — evicting or approximating which past key/value pairs to retain (Zhang et al. 2023's H2O finds >95% attention sparsity and a power-law 'heavy hitter' structure) — a fundamentally different strategy from compressing context into an updated parametric fast weight, which is the approach this work and In-Place TTT take." (Related Work)
- "H2O demonstrates that a small, dynamically-identified subset of tokens dominates attention allocation and that discarding the rest costs little accuracy, illustrating that most of a document's per-token information is not equally load-bearing for downstream prediction — a consideration relevant to interpreting why a compressed fast-weight carry might plateau rather than keep absorbing new documents." (Discussion)

## Verification
- Verified directly from PDF text: >95% attention sparsity claim (Sec 3.1), power-law heavy-hitter distribution and submodularity framing (Sec 3.2, Def 4.1, Thm 4.4), throughput multipliers (29x/29x/3x) and 1.9x latency figure (Abstract, Sec 5.2, Table 3/5), 20% budget matching full-cache accuracy across listed tasks (Table 1/2, Sec 5.1), 2.85–22.75% degradation range for single-component ablations (Sec 5.3 Q4), 4-million-token streaming claim and PG-19 comparison (Sec 5.3 Q1, Fig. 5).
- Could not independently verify exact numeric values in Tables 3–5 beyond what's shown in the extracted pages (e.g., precise per-cell throughput numbers were read directly and are accurate to the table as rendered).
