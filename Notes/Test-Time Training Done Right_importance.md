# Test-Time Training Done Right (LaCT)

**Citation:** Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan (MIT, Adobe Research). arXiv:2505.23884v1 [cs.LG], 29 May 2025.

## Important parts for my research
- **Large Chunk Test-Time Training (LaCT):**
  - Updates fast weights once per very large chunk, 2,048 to 1M tokens, instead of every 16–64 tokens.
  - Raises GPU utilisation "up to 70% on NVIDIA A100s" from "often below 5%".
  - Allows nonlinear fast-weight state of up to 40% of model size (a "state-to-parameter size ratio ≥ 40%", against 0.1%–5% before).
- **Update and apply:** W ← W − η∇_W L(f_W(k), v) and o = f_W(q), with a SwiGLU-MLP fast weight (§2.1, §3).
- **Stability is the central issue:** "Fast-weight updates in TTT repeatedly accumulate gradients, and thus suffer from magnitude explosion or decayed memory" (§3.2).
  - Their fix normalises the weights after every update, weight-update(W, g) = L2-Normalize(W − g), or uses a Muon update that normalises the gradient's spectral norm (Muon(g) ≈ UVᵀ).
  - "We do not use explicit weight-decay term as in previous methods."
  - Muon makes the learning rate "only reflect the relative importance of tokens within a chunk as Muon normalizes the absolute scale".
- **Language modelling:** chunk size is a hyperparameter (2,048 or 4,096), and sliding-window attention handles within-chunk causality. Results are "competitive ... compared to SoTA methods such as DeltaNet".
- **Other domains:** novel view synthesis with over 1M context, and a 14B autoregressive video diffusion model.
- Fast weights are per sequence; nothing is carried across sequences.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) LaCT shows that accumulated fast-weight updates "suffer from magnitude explosion or decayed memory" and fixes this by L2-normalising the fast weights after each update — Zhang et al. (2025). This is a designed bound that keeps only the direction, the same effect our read-time clip has once it saturates.
- (Related Work) Large-chunk TTT scales fast-weight state to 40% of model parameters, which is the regime where a carried state becomes as large as an attention cache.
- (Methodology) Precedent for treating a whole chunk of text as one fast-weight update, as In-Place TTT and our slices do.

## Verification
- Read the abstract, introduction, preliminaries and method (§1–3.3, PDF pages 1–6) and the language-modelling setup (§4.2) via pdftotext. The quoted sentences, Eq. 8–10, the 70%/5% utilisation and ≥40% state ratio figures were read directly. The language-model result tables (§5.2) and appendices were not read in detail.
