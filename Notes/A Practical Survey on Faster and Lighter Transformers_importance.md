# A Practical Survey on Faster and Lighter Transformers

**Citation:** Quentin Fournier, Gaétan Marceau Caron, Daniel Aloise, 2023, arXiv:2103.14636v2 (ACM Computing Surveys format)

## Important parts for my research
- Taxonomy of efficiency methods: general (gradient checkpointing, reversible layers, parameter sharing, pruning, distillation, mixed-precision/quantization, micro-batching, MoE) vs. specialized "x-formers" (sparse attention, factorized/low-rank attention, architectural change).
- Transformer-XL (Dai et al. 2019): segment-based recurrence — FIFO memory caches hidden states of the previous window; attention reads this cache but gradients do not flow into it. Achieves 4x the relative effective context length (RECL) of vanilla Transformer at same budget, but is bounded to the FIFO window range and cannot capture dependencies beyond it — a clean architectural analogy for "state carried across chunks but not indefinitely," useful contrast to session-persistent TTT (fast weight carries indefinitely, no FIFO eviction).
- Compressive Transformer (Rae et al. 2020) extends Transformer-XL: when the FIFO memory is full, oldest representations are compressed (pooling, 1D/dilated conv, or attention-based) into a secondary compressed memory instead of being discarded, trained with an auxiliary reconstruction loss.
- Linear/kernelized attention (Katharopoulos et al. 2020 "Transformers are RNNs"): rewrites softmax attention as a kernel feature map φ, giving Attention_i = φ(Q_i)(Σφ(K_j)V_jᵀ) / φ(Q_i)(Σφ(K_j)) — an explicitly recurrent/linear form where Σφ(K_j)V_jᵀ is an additive state analogous to a fast-weight matrix updated by outer products, directly relevant to hebbian-style associative memory framing. Performer (Choromanski et al. 2021) approximates the softmax kernel via random Fourier features (orthogonal Gaussian projections) instead of elu(x)+1.
- Long-Range Arena benchmark table (Tay et al. 2021, reproduced): BigBird 55.01% avg score (best), vanilla Transformer 54.39%, Linformer/Performer/Linear Transformer much faster (up to ~8x steps/sec at 4K length) but slightly lower average scores (~50-51%) — a reusable evidence point that efficiency variants trade a few points of quality for large throughput gains.

## What cite
- (Related Work) Transformer-XL/Compressive Transformer as the standard "memory across chunks within one continuous stream" baseline against which session-persistent cross-document carry should be distinguished — these carry hidden-state cache, not model weights, and are gradient-free/frozen at read time, bounded by FIFO eviction.
- (Related Work / Methodology) Katharopoulos et al.'s linear-attention-as-RNN reformulation as the standard equation for framing fast-weight-style associative state (Σφ(K)V^T) — useful for explaining hebbian outer-product updates by analogy.
- (Discussion) Long-Range Arena scores as a citable number for efficiency/quality trade-off framing if discussing computational cost of TTT compared to standard efficient-attention alternatives.

## Verification
- All specifics (Transformer-XL RECL 4x figure, Compressive Transformer memory-full compression mechanism, linear attention equation, LRA table values) directly read from PDF pages 26 and 24-25 and Table 3 (p.28). No inferred numbers.
