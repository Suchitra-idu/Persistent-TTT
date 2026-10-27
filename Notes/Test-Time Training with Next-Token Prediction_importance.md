# Test-Time Training with Next-Token Prediction

**Citation:** Xuan Ouyang, Zefan Cai, Junjie Hu (University of Wisconsin–Madison). arXiv:2606.21803v2 [cs.CL], 30 Aug 2026. Accepted to Findings of EMNLP 2026 (per the arXiv comment).

## Important parts for my research
- Direct successor of In-Place TTT. Same fast-weight site, the MLP down-projection of selected SwiGLU blocks ("Following Feng et al. (2026)"), and the same chunk-parallel rank-one writes. It changes only the **write target**.
- **Target**: the same layer's next-position contextual hidden state through a learned linear projection, v = W_proj h_{ℓ,t+1}. The write is ΔW = (W_proj h_{ℓ,t+1}) z_tᵀ (Eq. 5–7). This replaces In-Place TTT's "small learned convolution over a local window of hidden states".
- The training-time chunk write "sums bare outer products and drops the residual term that would appear in a literal least-squares regression step", because that term "is not prefix-summable" (§3.4). It is a Hebbian-style accumulation for parallelism, like ours.
- At inference it instead solves a ridge regression in closed form over the prompt (Eq. 16–17).
- **No persistence across inputs**: "We apply W_down ← W_down + ηΔW^CF, decode the answer, and restore the original weight after each sample" (§3.4). Adaptation is per prompt only.
- **Results**:
  - RULER Full-13, averaged over 4k–32k: +3.9 Llama-3.1-8B, +3.0 Mistral-7B, +4.1 Qwen3-4B, **+2.9 Qwen3-0.6B** over the released backbone.
  - LongBench-v2: +5.6 and +3.7.
  - Commonsense and knowledge benchmarks are roughly unchanged.
- **Key matched comparison**: "Under matched data, compute, fast-weight placement, and chunk size, the prior in-place TTT method does not improve over the released backbone on RULER" (§1). Continual pretraining alone and In-Place TTT "frequently degrade long-context retrieval" (§4.3).
- Target ablation: next-position supervision beats Past-5, Next-5 and Bi-dir-5 convolutional targets "by at least five RULER points at every evaluated length".

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) The closest follow-up to In-Place TTT keeps the down-projection fast weight but changes its write target to the next contextual hidden state, and still restores the original weight after every sample — Ouyang et al. (2026). Its adaptation never crosses inputs.
- (Related Work / Discussion) Under matched continual pretraining, the original In-Place TTT target did not improve RULER over the backbone (Ouyang et al., 2026). This is a caution that small TTT gains depend on the write target, consistent with our naive carry's null cross-document result.
- (Methodology) Like our Hebbian rule, its training-time write drops the error term to stay chunk-parallel (Ouyang et al., 2026, §3.4).
- (Related Work) It includes Qwen3-0.6B, the model we use (+2.9 RULER).

## Verification
- Read the abstract, introduction, related work, method (§1–3.4) and the experimental setup and baselines (§4.1–4.2) directly from the PDF via pdftotext. The quoted sentences, equation forms and the four RULER deltas were read directly. The per-length RULER tables and LongBench breakdown were only skimmed; the "+11.7 at 16k on Qwen3-4B" figure appears in §4.3 text.
