# Long Range Arena: A Benchmark for Efficient Transformers

**Citation:** Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler (Google Research, Google DeepMind). arXiv:2011.04006v1 [cs.LG], 8 Nov 2020 (Preprint).

## Important parts for my research
- Six synthetic/real-data tasks spanning 1K–16K tokens, purpose-built to probe long-range reasoning independent of pretraining: Long ListOps (2K, hierarchical 10-way classification), byte-level IMDb text classification (4K), byte-level ACL Anthology document retrieval/matching (4K per doc, two-tower, no cross-attention), CIFAR-10 pixel-sequence image classification (1024), Pathfinder (1024, binary path-connectivity), and Pathfinder-X (16K, same task at extreme length).
- Defines "required attention span" (mean query–attended-token distance weighted by attention) as a task-difficulty proxy computed from a trained vanilla Transformer; all LRA tasks score high, confirming they need genuinely long-range dependency use, not just local context.
- Striking failure mode: every one of 10 evaluated efficient-Transformer variants (plus vanilla Transformer) fails to learn Pathfinder-X (~50%, chance) despite succeeding at the identical task at 1024 length — pure sequence-length scaling, no architecture/task change, can cause total training failure rather than graceful degradation.
- No one-size-fits-all result: BigBird has the best average LRA score (55.01) but is not best on any individual task; kernel-based models (Performer, Linear Transformer) trade quality for large speed/memory wins (Performer 5.7× faster, Linformer ~10× less memory than vanilla Transformer at 4K).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology) The Pathfinder-X total-failure-at-length finding is a useful cautionary parallel for our "stability checked before trusting any gain" step — evidence that extreme-length/repeat-count regimes can produce qualitative training breakdowns rather than smooth trends, motivating our own stability checks before trusting carry effects.
- (Related Work / Methodology) "Required attention span" as a difficulty proxy and LRA's broader philosophy (isolate long-range dependency use, avoid conflating inductive bias with pretraining benefits) is a reusable framing for justifying our repeated-session, carry-isolating evaluation design.

## Verification
- Task descriptions, Table 1 results (Transformer 54.39, BigBird 55.01, all models FAIL on Path-X), and efficiency numbers (Table 2) were read directly from the PDF (pp. 3–8).
