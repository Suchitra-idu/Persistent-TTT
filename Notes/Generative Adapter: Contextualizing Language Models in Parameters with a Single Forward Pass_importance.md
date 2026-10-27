# Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass

**Citation:** Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Benjamin Van Durme, Luke Zettlemoyer, Jianfeng Gao, Hao Cheng (University of Washington, Microsoft). arXiv:2411.05877v1 [cs.LG], 8 Nov 2024.

## Important parts for my research
- Maps a context directly to a low-rank adapter added to a **frozen** LM, "using only forward passes", "inspired by fast weights (Ba et al., 2016; Schmidhuber, 1992)".
- **Generator:** for each adapted layer, W_Δ = (A₁A₂)(Σ_m h_m ⊗ h_m)(B₁B₂). This is an outer-product (Hebbian-like) accumulation of the base model's hidden states, projected by learned matrices (Eq. 1).
- **Streaming update:** S_t ← S_{t−1} + A₂H_tᵀH_tB₁, so the adapter accumulates over a stream of context chunks without storing them (Eq. 3–4). The accumulated adapter is used for later queries.
- **Stability:** "using the naive outer product for generating weights led to instability during training". The generated matrix sends inputs to near-zero or extremely large values "due to its skewed distribution of its singular values". The fix is **SVD normalization**, which resets all singular values to one (norm(M) = UVᵀ), keeping only the directions (§2.4). This parallels our finding that a read-time clip keeps only the direction of the carried state.
- **Training:** self-supervised on 1B SlimPajama tokens with reconstruction plus completion losses, then instruction tuning. Main experiments adapt only the attention output projection of Mistral-7B-Instruct or Llama2-7B-Chat.
- **Results:**
  - StreamingQA F1 of 31.5 vs. 19.5 for supervised fine-tuning, at contexts up to 32K.
  - MetaICL average 44.9 across 26 tasks.
  - **Multi-Session Conversation (MSC)**, which asks about a user's earlier sessions: closed-book F1 8.1, full-conversation prompting **66.0**, Generative Adapter **40.2**, at about a quarter of prompting's computation.
  - So prompting with the full history remains much more accurate.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Generative Adapter builds a context-dependent adapter in a single forward pass by accumulating outer products of hidden states over a stream of chunks, a learned fast weight carried across context chunks and sessions — Chen et al. (2024). Unlike our carry, it adapts attention output projections through a separately trained generator, and it is evaluated on question answering rather than perplexity.
- (Discussion) Chen et al. (2024) found naive outer-product weights unstable and normalised their singular values to one, keeping only directions. This is an independent case of a fast-weight memory that works with its magnitude discarded, like our saturated clip.
- (Discussion) On multi-session conversations, keeping the full history in context (F1 66.0) clearly beats the forward-pass adapter (40.2). This matches our finding that attention on earlier text is far stronger than a carried state.

## Verification
- Read the abstract, introduction and method (§1–2.4, PDF pages 1–5) and the experimental settings and document-QA setup (§3–4.1) directly via pdftotext. MSC Table 1 values (8.1, 66.0, 40.2) and the "4x" cost statement were read from the extracted text of §4.3. MetaICL details and §5 ablations beyond the first paragraph were not read. No venue is stated on the arXiv PDF.
