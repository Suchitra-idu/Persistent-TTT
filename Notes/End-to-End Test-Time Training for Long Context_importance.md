# End-to-End Test-Time Training for Long Context

**Citation:** Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun — arXiv:2512.23675v2 [cs.LG], 31 Dec 2025 (Astera Institute, NVIDIA, Stanford, UC Berkeley, UC San Diego).

## Important parts for my research
- TTT-E2E: fast weights (inner loop, W) updated via gradient descent on next-token prediction loss at test time; initialization W0 (outer loop, θ) meta-learned end-to-end so training loss matches test loss — resolves the TTT-naive/dynamic-eval mismatch where W0 is optimized for out-of-the-box loss, not post-TTT loss (Eq 1–6).
- Mini-batch TTT: gradient step every b=1K tokens, sliding-window attention k=8K (must have k≥b so local context bridges each batch before a TTT step). Only updates MLP layers, only last 1/4 of transformer blocks, and adds a second static MLP per updated block as "safe storage" to prevent forgetting pretrained knowledge — three implementation details load-bearing per ablations.
- **Critical for our gap framing**: fast weights RESET at every document boundary — the paper explicitly discards all training documents shorter than 8K "to avoid resetting the updated MLPs across document boundaries when different documents are packed into the same training sequence, since resetting slows down training" (footnote 4). Confirms no cross-document persistence is attempted or evaluated anywhere in this paper.
- TTT-KVB connection: when the fast-weight model g is linear and gradient is w.r.t. W_{t-1}, TTT-KVB reduces to DeltaNet; w.r.t. fixed W0 it reduces to linear attention; with a non-parametric kernel estimator it recovers self-attention (footnote 1, Sec 2.4.1) — a clean unifying equation-in-words for hebbian-vs-delta positioning.
- Number-of-layers-updated ablation: updating only 1 or 3 layers fails to scale with context length like full attention; updating 6 or 12 does, with 12 no better than 6 — larger fast-weight "state" is what enables context scaling, not more compute per se.
- On NIAH (RULER pass-key/UUID tasks), full attention dramatically outperforms all TTT/RNN methods including TTT-E2E — supports interpreting TTT as compression (lossy, loses irrelevant detail) rather than lossless retrieval memory.
- Ablation: TTT-E2E improves loss by 0.018 over full attention even when window k=8K equals full-attention context (no architecture difference), showing the gain is from learning-to-learn, not architecture.
- Scaling: TTT-E2E tracks full-attention's compute-scaling trend (regime boundary ~760M params / ~48B tokens) while Mamba2/Gated DeltaNet do not; 2.7× faster prefill latency than full attention at 128K context, but 3.4× slower training latency at 8K (1.2× faster at 128K) — training cost is a stated limitation.
- Pretraining: DCLM (filtered Common Crawl, 3.8T tokens, docs <8K discarded); long-context fine-tuning/eval: Books (not SlimPajama-6B, though SlimPajama and FineWebEdu are tested anecdotally for tokenizer/data-quality sensitivity).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Research Gap) "we discard all documents shorter than 8K to avoid resetting the updated MLPs across document boundaries... since resetting slows down training" — direct textual evidence that In-Place-TTT-style fast weights reset at document boundaries and cross-document persistence is unaddressed.
- (Related Work) TTT-E2E's meta-learned (E2E) initialization vs. dynamic evaluation (Krause 2018/2019, non-E2E) — useful contrast for framing our carry mechanism as building on non-meta-learned dynamic eval rather than E2E meta-learning.
- (Methodology) KV-Binding unifying equation: linear model → DeltaNet under W_{t-1} gradient, → linear attention under fixed-W0 gradient (footnote 1, Sec 2.4.1) — for explaining hebbian (fixed-init, no error correction) vs. delta-rule terminology precisely.
- (Discussion) NIAH: full attention "dramatically outperforms" TTT-E2E on pass-key/UUID retrieval — supports framing compression-based fast weights as fundamentally non-lossless, relevant to explaining why RULER shows no improvement in our own results.
- (Discussion) Two-MLP "safe storage" design and last-1/4-layers restriction as explicit anti-forgetting engineering choices — parallel to our need to explain stability/forgetting behavior in a persistent-carry setting.

## Verification
- All numbers (2.7×, 3.4×, 1.2×, 0.018 loss delta, 760M/48B regime boundary, k=8K/b=1K, last 1/4 layers, 88M vs 18M hidden state) checked directly against Figures 1/4/5/8, Table 1, and body text on pages 1–17.
- Footnote 4 (document-boundary reset) read directly, not inferred; this is the single most load-bearing fact for our Related Work section and was verified against the exact PDF wording.
