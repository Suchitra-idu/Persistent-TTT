# MemoryLLM: Towards Self-Updatable Large Language Models

**Citation:** Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, Julian McAuley (UC San Diego, Amazon, UCLA). Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235. arXiv:2402.04624v2.

## Important parts for my research
- **Design:**
  - Adds a fixed-size **memory pool** of hidden vectors ("memory tokens") to every layer of Llama2-7B, about 1B memory parameters in total.
  - The pool is the "self-updatable parameters" θ; the LLM weights φ stay static.
  - During generation every token attends to all memory tokens.
- **Self-update without backpropagation:**
  - New text is processed together with the last K memory tokens.
  - The resulting K new memory tokens replace K randomly dropped old ones, θ′ = U(θ, x).
  - Over a stream x₁…xₙ, θₙ = U(…U(θ, x₁)…, xₙ) (Eq. 1–2).
- **Built-in forgetting:** dropping K of N tokens per update gives exponential forgetting at rate K/N, "inspired by the Ebbinghaus Forgetting Curve" (§3.1.3).
- **Stated requirements** (§2.1): efficiency ("potentially eliminating the need for back-propagation"), efficacy, knowledge retention with a gradual phase-out, integrity, non-redundancy.
- **Results:**
  - Evaluated on model-editing benchmarks, long-context QA and custom retention tests.
  - "Operational integrity without any sign of performance degradation even after nearly a million memory updates" (abstract).
- Beyond Perplexity (2026) codes MemoryLLM as B-level evidence.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) MemoryLLM keeps a fixed-size pool of memory tokens in every layer, updated by forward passes over a stream of texts, with designed exponential forgetting — Wang et al. (2024). It persists state across many inputs, but as attended activations rather than as a fast weight inside the MLP.
- (Discussion) Its controlled forgetting rate (K/N per update) is a designed alternative to our fixed EMA decay of λ = 0.9, which harmed after long chains of unrelated documents.

## Verification
- Read the abstract, introduction, problem statement and structure design (§1–3.1.3, PDF pages 1–3) via pdftotext. The quoted sentences, the 1B pool size and the K/N forgetting argument were read directly. The training procedure, results tables and the "nearly a million updates" experiment details were not read beyond the abstract and introduction. The ICML 2024 venue is printed on the PDF.
