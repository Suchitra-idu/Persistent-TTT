# Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

**Citation:** Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal (Google). arXiv:2404.07143v2 [cs.CL], 9 Aug 2024. Preprint, under review at the time.

## Important parts for my research
- Each attention layer keeps a **compressive memory**, an associative matrix per head, alongside local causal attention within a segment.
- The memory stores old KV states "instead of discarding them like in the standard attention mechanism" and passes them to the next segment, "building in a recurrence in each attention layer" (§3).
- **Update rules** (Eq. 8–9):
  - "Linear": M_s ← M_{s−1} + σ(K)ᵀV, with a normaliser z. This is "an associative binding operator (Smolensky, 1990; Hebb, 2005)", i.e. Hebbian.
  - "Linear + Delta": subtracts the retrieved value first, so the memory is left "unmodified if the KV binding already exists".
- **Mixing:** memory output and local attention are combined by a learned per-head gate, sigmoid(β) (Eq. 10). After training, heads specialise: near 0 (local), near 1 (memory), or mixers near 0.5.
- **Results:**
  - Beats Transformer-XL and Memorizing Transformers on long-context language modelling with "114x" less memory than a 65K-token retrieval memory.
  - 1M-token passkey retrieval with a 1B model.
  - 500K-token book summarisation with an 8B model.
  - Linear and Linear+Delta perform similarly; perplexity is 2.21 vs. 2.20 with 100K training.
- The memory is carried across segments of one long input, not across separate documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Infini-attention carries a compressive associative memory across the segments of one long input, written by a Hebbian or delta rule and mixed with local attention through a learned gate — Munkhdalai et al. (2024). Our carry is a similar fixed-size associative state, but it lives in the MLP fast weight and crosses document boundaries.
- (Discussion) Its learned per-head memory gate, which specialises towards memory or local attention, is a parallel to our separate, gated carry path.

## Verification
- Read the abstract, introduction, background and method (§1–3.1, PDF pages 1–5) via pdftotext. Eq. 8–10, the quoted sentences, the 114× memory figure, the gating-score observation and the 2.21/2.20 perplexities were read directly. Full result tables were skimmed only.
