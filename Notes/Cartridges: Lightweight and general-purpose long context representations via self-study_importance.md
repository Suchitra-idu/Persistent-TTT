# Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study

**Citation:** Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Ré (Stanford, Caltech, University at Buffalo). arXiv:2506.06266v3 [cs.CL], 13 Jun 2025.

## Important parts for my research
- **Problem:** serving many queries about the same large corpus with in-context learning is costly, because the KV cache grows with input length.
- **Method:** "training a smaller KV cache offline on each corpus" by backpropagating into key and value vectors, "essentially equivalent to prefix tuning". The trained cache (the **Cartridge**) is loaded at inference.
- **Naive next-token training fails at generality:**
  - it memorizes the corpus "perfectly using 107× less memory than the KV-cache";
  - but the resulting Cartridges "degrade the LM's ability to respond to diverse questions beyond regurgitating the corpus".
- **Self-Study fix:**
  - generate synthetic conversations about the corpus;
  - train with a context-distillation objective toward the model with the corpus in context.
- **Results:**
  - Matches ICL quality "while using 38.6× less memory and enabling 26.4× higher throughput".
  - Extends effective context, from 128k to 484k tokens on MTOB.
  - Cartridges "can be composed at inference time without retraining".
  - A KV-cache parameterisation beats a LoRA parameterisation.
- The memory is built per corpus by **offline gradient training**, amortised over many queries. It is not built during the forward pass and does not cross independent corpora by itself.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Cartridges trade compute for memory in the opposite way from a carried fast weight: a compact KV cache is trained offline per corpus and reused for many queries, matching in-context quality with 38.6× less memory — Eyuboglu et al. (2025).
- (Discussion) Eyuboglu et al. (2025) find that next-token training on the corpus alone memorizes but does not generalize; generality needed context distillation. This is a caution that a state trained only by next-token loss may carry surface statistics rather than usable knowledge.
- (Related Work) Compressed contexts can be composed across documents without retraining, a further point of comparison for cross-document transfer.

## Verification
- Read the abstract, introduction and the start of the preliminaries (PDF pages 1–3) directly via pdftotext. The quoted sentences and numbers (107×, 38.6×, 26.4×, 128k→484k, composability, KV vs LoRA) were read directly from that text. The method details (§4), full results tables and appendices were not read.
