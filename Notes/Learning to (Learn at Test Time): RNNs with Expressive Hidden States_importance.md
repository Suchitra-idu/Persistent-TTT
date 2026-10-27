# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

**Citation:** Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin (Stanford, UC San Diego, UC Berkeley, Meta AI). arXiv:2407.04620v4 [cs.LG], last revised 31 Aug 2025.

## Important parts for my research
- Core formulation: hidden state is itself model weights W_t; output rule z_t = f(x_t; W_t); update rule W_t = W_{t-1} - η∇ℓ(W_{t-1}; x_t) — a gradient step per token is TTT.
- Self-supervised loss is multi-view reconstruction: corrupted "training view" θ_K x_t, "label view" θ_V x_t, loss ℓ(W;x_t)=‖f(θ_K x_t;W) − θ_V x_t‖², output uses a separate "test view" θ_Q x_t — θ_K,θ_Q,θ_V are outer-loop learned, analogous to attention's K/Q/V.
- Two instantiations: TTT-Linear (f linear) and TTT-MLP (2-layer MLP, hidden dim 4× input, GELU); f always wraps LN + residual: f(x)=x+LN(f_res(x)) — critical for stability.
- Mini-batch TTT (chunk size b=16) is the single biggest ablation win (Table 1: 15.23→12.35 ppl), far more than learnable W_0 or learnable η; dual form gives >5× wall-clock speedup on TPUs by avoiding materializing per-token gradients.
- Theorem 1: TTT-Linear + batch GD + η=1/2 + W_0=0 is exactly equivalent to linear attention. Theorem 2: TTT with Nadaraya-Watson (nonparametric) learner is exactly equivalent to self-attention.
- DeltaNet is stated as equivalent to TTT-Linear with inner-loop batch size 1, without LN/residual — direct link between our hebbian/delta-rule design space and this framework's ablations.
- Learnable η(x)=η_base·σ(θ_lr·x) acts as a per-token gate on the gradient step — a mechanism for input-dependent update magnitude, relevant to controlling carry strength.
- Empirical: at 8k context, TTT-Linear/MLP beat Mamba; Mamba's per-token-index perplexity plateaus after 16k while TTT and Transformer keep improving to 32k — motivates why fixed-size RNN states saturate but TTT's learned compression does not (as much).
- W_0 (θ_init) is shared/learned across ALL sequences but hidden state is reset to W_0 at the start of every sequence — the paper's TTT is single-continuous-sequence only; no cross-document/session carry is discussed or tested.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Introduction/Related Work) Foundational TTT-layer framing: hidden state as a model, update as an SGD step, unifying RNNs/attention/TTT under one abstraction (Sun et al. 2025).
- (Methodology) Multi-view reconstruction loss and K/Q/V-style outer-loop parameterization as the template our self-supervised task follows.
- (Related Work — contrast) Theorem 1 equivalence of TTT-Linear+batch-GD to linear attention, and the paper's own statement that DeltaNet = TTT-Linear with batch size 1 minus LN/residual — useful for positioning hebbian (pure outer-product, no correction) vs. delta-rule variants in our design space.
- (Research Gap) This paper's TTT resets W_0 at every new test sequence; it never proposes or evaluates carrying fast weights across independently-bounded documents — supports our gap claim.
- (Discussion — mini-batch/chunk size effect) Chunk size b materially changes perplexity (Fig. 7); relevant if we discuss how chunking/frequency of updates could interact with flat carry.

## Verification
- Equations, Theorem 1/2 statements, Table 1 ablation numbers (15.91→11.09), and the DeltaNet-equivalence claim were read directly from the PDF text (pp. 4–17).
- The claim that W_0 is shared but resets per-sequence (no cross-document persistence) is a direct reading of Section 2.7 ("Learnable W_0") plus the absence of any cross-sequence carry discussion in the paper — not explicitly stated as a "limitation" by the authors, so this is an inference from what is and isn't described.
