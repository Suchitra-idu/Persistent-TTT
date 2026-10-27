# Attention Is All You Need

**Citation:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, 2017, NeurIPS (31st Conference on Neural Information Processing Systems, NIPS 2017), also arXiv:1706.03762

## Important parts for my research
- Defines the position-wise feed-forward network (FFN) sub-layer exactly as: FFN(x) = max(0, xW1 + b1)W2 + b2 (Eq. 2), described as "two convolutions with kernel size 1," applied identically and independently at each sequence position, with d_model=512 input/output and d_ff=2048 inner dimensionality in the base model — this is the two-matrix (up-projection, down-projection) MLP structure that In-Place TTT's fast weights modify (the "down-projection" is W2 here), so this is the canonical definition of the module being adapted.
- Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V (Eq. 1); multi-head attention runs h=8 parallel heads with d_k=d_v=d_model/h=64, concatenated and projected by W^O (base model config in Table 3).
- Self-attention layer complexity is O(n²·d) per layer with O(1) sequential operations and O(1) maximum path length between any two positions (Table 1) — the standard complexity citation motivating why long-context/efficient-attention and non-attention (recurrent/fast-weight) alternatives are studied at all.
- Encoder/decoder each use residual connections around every sub-layer followed by LayerNorm: LayerNorm(x + Sublayer(x)) — standard pre-existing structure any adaptation (LoRA, TTT fast weight) must be inserted around or into.
- Sinusoidal positional encoding (not relevant to fast weights directly, but establishes the base architecture has no persistent state across the sequence beyond attention over the visible context window — motivating why nothing here "carries" past a context boundary without an added mechanism.)
- Base model trained for 100K steps (12h) on 8 P100 GPUs; achieves 27.3/38.1 BLEU on EN-DE/EN-FR (Table 2); big model 28.4/41.8 BLEU. These numbers are not otherwise relevant to TTT but confirm this is the base Transformer architecture being cited for background, not a TTT-adjacent contribution.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The Transformer architecture's feed-forward sub-layer is two affine transformations around a ReLU, FFN(x) = max(0, xW1+b1)W2+b2, applied identically at every sequence position (Vaswani et al. 2017); it is the down-projection matrix W2 of this module that In-Place TTT (Feng et al. 2026) and this work's fast-weight mechanism update at inference time" (Background/Methodology — grounds what layer is being modified).
- "Self-attention layers scale quadratically in sequence length, O(n²·d) per layer (Vaswani et al. 2017), the standard motivation cited across the efficient-attention and long-context literature this proposal builds on" (Background).
- "The base Transformer architecture has no mechanism for retaining information across independently-processed inputs beyond what is explicitly re-supplied in context (Vaswani et al. 2017); persistence across document/session boundaries is not a capability of the base architecture and must be added" (Introduction/Motivation — sets up the gap the proposal addresses).

## Verification
- Verified directly from the PDF: Eq. 1 (attention formula) and Eq. 2 (FFN formula) as printed in Sec. 3.2.1 and 3.3; d_model=512, d_ff=2048, h=8, d_k=d_v=64 from Sec. 3.1/3.2.2 text and Table 3 "base" row; Table 1's O(n²·d) self-attention complexity and O(1) sequential-ops/max-path-length entries; BLEU scores 27.3/38.1 (base) and 28.4/41.8 (big) from Table 2; 100K steps/12 hours training detail from Sec. 5.2.
- No claims in this note required inference beyond what is printed in the text/tables; all are read directly.
