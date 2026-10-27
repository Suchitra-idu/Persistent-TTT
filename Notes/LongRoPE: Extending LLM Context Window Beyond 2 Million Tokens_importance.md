# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**Citation:** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang — Microsoft Research, arXiv:2402.13753v1 [cs.CL], 21 Feb 2024.

## Important parts for my research
- Extends LLaMA2-7B / Mistral-7B context from 4k to 2048k tokens purely via non-uniform RoPE rescaling — no architecture change, no fast-weight/memory mechanism. Two non-uniformities exploited: (1) different RoPE dimensions need different interpolation strength; (2) initial token positions (first n̂) should get less/no interpolation (echoes attention-sink findings).
- Evolutionary search over per-dimension rescale factors λ_i and threshold n̂ (Table 4 search space), with a monotonicity constraint (λ_i non-decreasing in dimension i, motivated by NTK theory) to prune the search space.
- Progressive extension: search+fine-tune to 256k, then a second non-fine-tuned search extends 8× further to 2048k without added training — "search generalizes across window sizes" is the key trick enabling million-token windows cheaply (1k fine-tuning steps total).
- Passkey retrieval ≥90% accuracy from 4k–2048k; standard 4k benchmarks (ARC, HellaSwag, MMLU, TruthfulQA) preserved after a final "short-context recovery" re-search.
- No claim of any parametric/weight update at inference time — this is a pure positional-embedding technique, entirely orthogonal to weight-carrying TTT approaches.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) "Positional-interpolation methods such as LongRoPE (Ding et al., 2024) extend effective context length without touching model weights, in contrast to our approach which carries information via weight updates (fast weights) rather than position encoding — the two are complementary, not competing."
- (Related Work) Non-uniform treatment of "recent"/boundary tokens (their n̂ initial-token exemption) parallels our within-document vs. cross-document boundary distinction — both literatures find that boundary positions/tokens need special handling.

## Verification
- All numbers (2048k window, 1k fine-tuning steps, ≥90% passkey accuracy, PG19/Proof-pile perplexities in Tables 1/5/6) directly read from text and tables, not inferred.
- No fast-weight, TTT, or carry mechanism appears anywhere in this paper — confirmed by full read; it is purely a RoPE-embedding search method.
