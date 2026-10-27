# Adapting Language Models to Compress Contexts (AutoCompressors)

**Citation:** Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen (Princeton University). arXiv:2305.14788v2 [cs.CL], 4 Nov 2023. Published at EMNLP 2023 (main conference), pp. 3829–3846, doi:10.18653/v1/2023.emnlp-main.232.

## Important parts for my research
- **Method:**
  - Builds on the Recurrent Memory Transformer.
  - Fine-tunes pretrained LMs (OPT-1.3B/2.7B, Llama-2-7B) to emit κ "summary vectors" at the end of each segment. These are passed as soft prompts to later segments.
  - **Summary accumulation** concatenates the summaries of all earlier segments.
  - **Randomized segmenting** varies the segment length during training.
- Trained with plain next-token loss over the whole document, so the summaries learn to "store useful information" for future segments.
- **BPTT with stop-gradients**: summary vectors are cached "and stop their gradients after 2 compression steps", assuming "it is sufficient to predict the tokens in the adjacent S_{i+1}". This "incurs no penalty" and reduces memory (§3.1). This is the same design choice as our carry window of two slices.
- **Language modelling** (Table 1, OPT-2.7B, 2,048-token target): perplexity falls from 6.28 to 5.93 in domain (−5.6%) with 3 compressed segments (6,144 tokens) using only 150 summary vectors. Extended full attention reaches 5.94 with 2,048 extra tokens.
- **Cross-document use** (§6.1, retrieval-augmented LM):
  - Summary vectors of retrieved passages are fused, "despite a mismatch in training since we draw independent summary vectors from separate documents".
  - The model was "only ever trained to accumulate 3 sets of summary vectors, and yet it benefits from fusing the summary vectors of up to 10 documents".
  - With 50-token passages, fused summaries outperform fused plain passages and REPLUG; with 512-token passages, REPLUG top-10 still wins over fused summaries.
- Storage cost: compressing 10B tokens takes 5 TB per domain in half precision.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) AutoCompressors compress earlier segments into summary vectors trained by next-token loss and reuse them across documents in retrieval — Chevalier et al. (2023). Their carried object is a set of soft-prompt activations, not a fast weight.
- (Methodology) Chevalier et al. (2023) also stop gradients after two compression steps without loss of quality, the same choice as our carry window of two slices.
- (Discussion) As with our carry, AutoCompressors are trained within documents yet still help when summaries from independent documents are combined. Training within one unit does not preclude transfer across units.

## Verification
- Read the abstract, introduction, related work, method (§3) and the language-modelling evaluation (§4.1–4.2, Tables 1–3) directly via pdftotext. The retrieval section (§6.1, Table 5) was read through the extracted text, and its quoted sentences and the ranking of methods were read directly (Table 5's numbers were garbled in extraction, so no gains are quoted). In-context learning (§5) and passage re-ranking (§6.2) were not read. The EMNLP 2023 venue was confirmed on the ACL Anthology (https://aclanthology.org/2023.emnlp-main.232).
