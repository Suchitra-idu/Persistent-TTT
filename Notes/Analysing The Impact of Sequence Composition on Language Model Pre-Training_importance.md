# Analysing The Impact of Sequence Composition on Language Model Pre-Training

**Citation:** Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłoś, Yuxiang Wu, Pasquale Minervini. Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), ACL 2024, pages 7897–7912. arXiv:2402.13991.

## Important parts for my research
- **Setting:** 1.3B models, 2K and 8K context, pre-trained on 150B tokens of SlimPajama (the same seven sub-corpora our holdout draws from) (§3.1).
- **Three ways to pack documents into a training chunk (§2.1):**
  - MixChunk, documents sampled at random from the whole corpus (the usual baseline);
  - UniChunk, every chunk from a single source;
  - Bm25Chunk, documents retrieved with BM25 so that packed documents are related.
- **Unrelated earlier documents distract under causal masking.** Concatenating two random documents separated by [eos], "causal masking models can be distracted by unrelated documents in context", with distraction proportions of 45 to 52 percent up to the 256th token of the second document (§5.1).
- **Masking earlier documents out helps.** Intra-document causal masking gives the lowest average perplexity (8.410 at 2K, 8.079 at 8K, Table 1) at a 4.0 percent efficiency cost.
- **Related earlier documents help.** "Increasing the relatedness of documents in a sequence can improve the language modelling ability of models" (§3.2). Against MixChunk, Bm25Chunk lowers average perplexity by 0.62 (2K) and 0.78 (8K), and UniChunk (same source) by 0.34 and 0.21 (Table 1).
- The state here is the attention context during pre-training, not a fast weight, and the effect is on training, not on a test-time carry.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work, positioning) The attention-side counterpart of our relatedness gradient: in pre-training, related earlier documents help, same-source ones help less, and unrelated ones distract — Zhao et al. (2024). Our result is the same ordering for a carried fast weight at test time, so the ordering itself is not new; what is new is where the carried state lives.
- (Discussion) Their intra-document masking result is the analogue of resetting the fast weight at each document, which is what In-Place TTT does.

## Verification
- Read the arXiv abstract and the HTML version (§2.1, §3.1, §3.2, Table 1, §5.1) on 2026-11-01. Section numbers and figures above were read there. The PDF was not read, and §4 results were read only for the memorisation figure.
