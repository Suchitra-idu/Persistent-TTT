# Recurrent Memory Transformer

**Citation:** Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev — 36th Conference on Neural Information Processing Systems (NeurIPS 2022) (arXiv:2207.06881v2, 8 Dec 2022).

## Important parts for my research
- RMT augments a Transformer's input/output sequence with special [mem] tokens — m read-memory tokens prepended, m write-memory tokens appended per segment — with no change to the backbone Transformer itself. Segment τ+1's read memory is initialized from segment τ's write-memory output: H^mem_{τ+1} := H^write_τ (segment-level recurrence via token passing, not internal state).
- Trained with BPTT; the number of previous segments to backpropagate through ("BPTT unroll," 0–4 in experiments) is a key hyperparameter — deeper unrolling substantially improves language-modeling perplexity, and is the mechanism that makes the recurrent memory tokens actually useful (not just a fixed-point compressor).
- On algorithmic Copy/Reverse tasks, RMT solves the task near-perfectly up to 9 segments (360 tokens total) with fixed memory size, while Transformer-XL degrades sharply past ~2–3 segments; RMT with memory size 60 tokens still copies correctly out to 720-token sequences.
- Language modeling: RMT matches Transformer-XL perplexity using far less memory (e.g., RMT memory 5–25 ≈ Tr-XL cache 50–75 tokens on WikiText-103 and enwik8) — RMT compresses more effectively per stored token. Combining Tr-XL cache (short-term) with RMT memory (long-term) gives the best PPL (23.99 vs. ~24.0–24.68 for either alone).
- Attention-map analysis shows interpretable write/read patterns: write-memory attention compresses the whole segment (diagonal pattern); for Reverse, the model learns to write tokens into memory in reversed order — i.e., the write operation is itself task-adapted, not a generic average.
- Real-task test: adding just 10 recurrent memory tokens to pretrained encoder LMs (RoBERTa-base, BERT, DeBERTa, T5) and fine-tuning for Hyperpartisan news classification achieves SOTA F1 (up to 98.11), beating models with 4× longer native context — memory augmentation transfers to already-pretrained models without pretraining from scratch.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) RMT as the clearest prior example of state passed *across segment/document boundaries* — but the carried state is a compact learned token-based representation trained end-to-end via BPTT-through-segments, not raw fast weights updated by an unsupervised/self-supervised rule; and RMT's segments are still slices of one continuous training corpus, not independently-bounded documents the way our carry crosses.
- (Discussion — explains flat/non-compounding carry) RMT's finding that deeper BPTT unrolling is *necessary* for the recurrent memory to be useful is a candidate explanation for our flat carry: our carry has no analogous supervised signal teaching the write step what to preserve for a future document — it accumulates via pure Hebbian outer-product updates with no gradient feedback from later performance, so nothing trains it to be predictively useful the way RMT's memory tokens are trained to be.
- (Methodology) The "memory can be retrofitted onto an already-pretrained model without retraining from scratch" result (Hyperpartisan experiment) supports methodological choices that add a persistence mechanism to a pretrained base model rather than training from scratch.

## Verification
- Segment-recurrence equations, BPTT-unroll ablations, and all quantitative results (Tables 1–5, Figs. 3–6) were read directly from the PDF text and tables (pages 3–9). Nothing here is inferred beyond direct reading.
