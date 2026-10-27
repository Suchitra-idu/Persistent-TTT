# Big Bird: Transformers for Longer Sequences

**Citation:** Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed, 2020, NeurIPS 2020 (Google Research), arXiv:2007.14062v2

## Important parts for my research
- BigBird's sparse attention = random attention (r random keys per query) + sliding window (width w) + global tokens (g tokens attending to/from everything), reducing complexity from quadratic to linear in sequence length while remaining a universal approximator of sequence functions and Turing complete (given O(1) global tokens).
- Two global-token variants: ITC (internal transformer construction — designate existing tokens as global) vs ETC (extended transformer construction — add extra global tokens, effectively extra storage locations "similar to memory," per the authors' own framing).
- Ablation (Table 1, @512 length): window-only attention gets MLM bitrate 58.3 vs BERT-base's 64.2; random-only gets 60.1; random+window combined gets 62.7 — random and local attention alone are each insufficient to match full attention, motivating the combination.
- Handles sequences up to 8x longer than prior full-attention models on the same hardware; achieves SOTA/near-SOTA on HotpotQA, Natural Questions, TriviaQA, WikiHop QA and long-document summarization (Arxiv, PubMed, BigPatent) and genomics tasks (promoter-region prediction F1 99.9, chromatin-profile prediction).
- Explicit no-free-lunch result: constructs a task (finding the furthest vector by inner product, related to the Orthogonal Vector Conjecture) that any sparse-attention mechanism with Õ(n) edges needs Ω̃(n^(1-o(1))) layers to solve, while full attention solves it in O(1) layers — sparsity has a provable cost on some tasks.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) BigBird as the representative "extend usable context length via sparse/global attention within one forward pass" approach — contrasts with session-persistent TTT's strategy of carrying fast weights across independently-bounded documents rather than extending the single attended context window.
- (Related Work) The no-free-lunch/OVC lower bound result as a citable formal argument that any linear-complexity context mechanism (attention-based or fast-weight-based) trades some expressivity for efficiency, useful context for discussing why persistence-based approaches might show flat/limited gains on some tasks.

## Verification
- All figures (MLM bitrate ablation, 8x context claim, promoter F1 99.9, no-free-lunch OVC-based lower bound) read directly from PDF pages 1, 4, 6, 9. Paper is tangential to the TTT/fast-weight/carry research core — no persistence or fast-weight mechanism is discussed; scope kept accordingly limited.
