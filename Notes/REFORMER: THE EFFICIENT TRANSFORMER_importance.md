# Reformer: The Efficient Transformer

**Citation:** Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya, 2020, ICLR (Published as a conference paper at ICLR 2020; arXiv:2001.04451)

## Important parts for my research
- Two independent efficiency tricks, not related to fast weights: (1) LSH attention replaces O(L²) dot-product attention with O(L log L) via angular locality-sensitive hashing — queries only attend to keys in the same hash bucket; hash h(x)=argmax([xR; -xR]) for random rotation matrix R. Multi-round hashing (n_rounds hashes) trades compute for accuracy: 8 hashes ≈ full-attention accuracy on a synthetic duplication task (Table 2: LSH-8 reaches 94.8%/99.9%/99.9%/99.9% across train/eval settings vs full attention).
- (2) Reversible residual layers (from RevNets, Gomez et al. 2017): paired streams y1=x1+F(x2), y2=x2+G(y1), with F=attention, G=feed-forward; activations recomputed on backward pass instead of stored, removing the N-layer memory multiplier. Combined with chunked feed-forward computation, memory becomes independent of both depth and d_ff.
- Shared-QK attention (k_j = q_j/||q_j||) needed to make LSH bucketing meaningful; empirically shown not to hurt performance (enwik8/imagenet64 loss curves nearly identical to separate Q/K).
- Results: 12-layer Reformer reaches 1.19 bits/dim on enwik8 in 20K steps (1.05 with more tuning); processes 64K-token sequences on enwik8 and 12K on imagenet64 on a single accelerator; LSH attention speed stays flat with sequence length while full attention grows linearly (Figure 5).
- Purely an inference/training-efficiency architecture — no test-time parameter updates, no persistence across inputs, no memory carried between documents. All efficiency comes from approximating a fixed, static attention computation.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Reformer represents the efficient-long-context-via-approximation family (LSH attention, reversible layers) — orthogonal to and non-competing with TTT/carry approaches: it reduces the cost of processing a long context but does not persist any adapted state across documents or sessions, unlike In-Place TTT or our session-persistent carry.
- (Related Work) Useful contrast point when framing "how prior work achieves long-context efficiency without weight updates" — Reformer's O(L log L) attention is a pure architectural/compute answer to the same length problem that TTT/fast-weight methods answer via parameter adaptation.

## Verification
- All complexity claims (O(L²)→O(L log L)), the shared-QK result, reversible-layer equations (Eq. 7–9), and the enwik8/imagenet64 bits/dim numbers were read directly from the PDF (Tables 1–4, Figures 3–5, Section 2–3). No inferred figures.
