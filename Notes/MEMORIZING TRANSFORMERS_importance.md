# Memorizing Transformers

**Citation:** Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy — Google, Published as a conference paper at ICLR 2022, arXiv:2203.08913v1 [cs.LG], 16 Mar 2022.

## Important parts for my research
- Mechanism: one attention layer near the top of the stack gets kNN-augmented attention into a large non-differentiable external memory of (key, value) pairs from past tokens; gate g=σ(b_g) combines memory-attention output V_m and local-context output V_c: V_a = V_m⊙g + V_c⊙(1-g) (Eq. 1–2). Keys/values are never backpropagated into — memory is filled from prior forward passes only, making it cheap to scale to 262K tokens.
- **Directly relevant to cross-document persistence gap:** "each subsequence in the batch comes from a different document, and thus requires a separate external memory, which is cleared at the start of each new document" (Sec 3.1, Batching). This is the same reset-at-document-boundary pattern as In-Place TTT — confirms that both retrieval-augmented and TTT/fast-weight long-context methods conventionally treat documents as memory-isolated units; nobody in this lineage carries state across independently-bounded documents either.
- Memory helps most on rare words: proper names, citations, references, function/variable names (Sec 4.6) — i.e., recall of *specific* prior tokens, not generalization. Case study on Isabelle: 8/10 sampled predictions of lemma names successfully retrieved the lemma's own definition body from memory.
- Distributional shift / staleness (Sec 3.2): because model parameters producing queries drift during training while stored keys/values don't, older memory entries become "stale"; normalizing keys/queries mitigates but doesn't eliminate this. Training instability appears with very large memories unless first pretrained small then fine-tuned to larger memory — relevant analogue for why accumulated fast-weight carry might degrade or plateau: staleness of a persisted update relative to a drifting readout.
- Perplexity improves monotonically with memory size up to 262K tokens (near-longest documents in arXiv); diminishing/no further gain expected past document length (Table 4, 5) — a clean numeric example of a compounding external representation, useful as *contrast* to our flat carry.
- Small Memorizing Transformer (8K memory) matches a vanilla Transformer with 5× more parameters (Fig. 1) — striking comparable-numbers claim for Related Work.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) "Prior long-context memory mechanisms, whether retrieval-based (Memorizing Transformers, Wu et al. 2022) or fast-weight-based (In-Place TTT), uniformly clear their memory/fast-weights at document boundaries; our carry ablation is, to our knowledge, the first to test what happens if this reset is removed."
- (Discussion — explains flat/non-compounding carry) Staleness argument: a persisted representation whose producing parameters keep drifting can decay in usefulness over repeats, a candidate explanation alongside "no error-correction" for flat carry.
- (Comparable numbers) 8K-memory model ≈ 5× larger vanilla Transformer in perplexity (Fig. 1); diminishing returns past ~262K memory ≈ document length (Table 5).

## Verification
- Gate equation, batching/clearing-per-document quote, staleness discussion, and the 8K-memory-vs-5×-params claim all directly read from Sections 3.1–3.2, 4.3–4.6 and Figure 1 captions. No inference.
