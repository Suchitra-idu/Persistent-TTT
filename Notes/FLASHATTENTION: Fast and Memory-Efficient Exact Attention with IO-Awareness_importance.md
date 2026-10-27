# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Citation:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, Stanford University / University at Buffalo, arXiv:2205.14135v2 [cs.LG], 23 Jun 2022.

## Important parts for my research
- Not about fast weights, TTT, or carry state — this is a systems/efficiency paper for exact softmax attention. Low direct relevance; useful only as infrastructure/methodology citation.
- Core idea: standard attention materializes the full N×N score matrix `S=QK^T` and `P=softmax(S)` in HBM (slow GPU memory), which is the actual bottleneck, not FLOPs. FlashAttention uses **tiling** (block-wise softmax via running max/sum statistics, "algebraic aggregation") and **recomputation** in the backward pass to avoid ever materializing the N×N matrix.
- IO complexity: standard attention requires Θ(Nd + N²) HBM accesses; FlashAttention requires Θ(N²d²M⁻¹) where M is SRAM size — proven asymptotically optimal (matching lower bound) across SRAM sizes.
- Memory: FlashAttention uses O(N) additional memory (linear in sequence length) vs O(N²) for standard attention — up to 20× more memory-efficient than exact-attention baselines.
- Reported speedups: 15% faster BERT-large training vs MLPerf 1.1 record; up to 3× faster GPT-2 training vs HuggingFace (1.8× vs Megatron-LM); 2.4× speedup on Long-Range Arena (seq len 1K–4K).
- Enables longer context cheaply: GPT-2 trained with 4K context via FlashAttention is still 30% faster than Megatron-LM's GPT-2 at 1K context, while achieving 0.7 better perplexity — i.e., longer context at lower wall-clock cost, not just parity.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "We use FlashAttention (Dao et al., 2022) for the transformer backbone's attention layers to make long-context / multi-document training and evaluation runs computationally tractable at the sequence lengths this study requires." (Methodology / Implementation)
- "Longer effective context enabled by IO-aware exact attention has been shown to directly improve language modeling perplexity (0.7 PPL improvement on GPT-2 with 4× context), independent of any fast-weight or recurrent-state mechanism — a confound worth controlling for when attributing perplexity gains to a persistent carry rather than to context length alone." (Methodology — potential confound to rule out)

## Verification
- All numbers (15% BERT speedup, 3×/1.8× GPT-2 speedup, 2.4× LRA speedup, 0.7 PPL gain at 4× context, 20× memory efficiency, IO complexity Θ(N²d²M⁻¹) vs Θ(Nd+N²)) were read directly from the abstract, Table 1, Table 2, Table 4, and Theorem 2 in the PDF text.
- Did not read remaining pages (backward-pass appendix, extended benchmarks) — not needed given this paper's low relevance to the proposal's core mechanism; nothing beyond page 10 was used in this note.
