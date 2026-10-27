# Larimar: Large Language Models with Episodic Memory Control

**Citation:** Payel Das, Subhajit Chaudhury, Elliot Nelson, Igor Melnyk, Sarathkrishna Swaminathan, Sihui Dai, Aurélie Lozano, Georgios Kollias, Vijil Chenthamarakshan, Jiří Navrátil, Soham Dan, Pin-Yu Chen (IBM AI Research, Princeton). Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, pp. 10109–10126. arXiv:2403.11901v4 [cs.LG], 21 Aug 2024.

## Important parts for my research
- Brain-inspired, after **complementary learning systems** theory: a hippocampus-like fast-learning **episodic memory** coupled to a neocortex-like slow-learning LLM decoder.
- **Memory write and read:**
  - Uses a Kanerva-Machine-style memory with "Generative Pseudo-inverse Memory".
  - "Reformulates these Bayesian updates as finding least-square solutions to linear systems" (§2–3), e.g. "Compute posterior memory M = W₀†Z_ξ".
  - The write is therefore error-correcting (a least-squares fit), unlike a Hebbian sum.
- **"Dynamic, one-shot updates of knowledge without the need for computationally expensive re-training or fine-tuning".**
- **Results:**
  - Fact-editing accuracy comparable to strong baselines, including sequential editing, with "speed-ups of 8-10x".
  - Also supports selective fact forgetting, information-leakage prevention, and input-context-length generalisation via recursive search (abstract, §1).
- The memory is explicit and external to the LLM weights, conditioning the decoder. It is aimed at fact editing, not at perplexity across documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Larimar adds an external episodic memory to an LLM, written in one shot by least-squares solutions rather than by gradient steps, for fast knowledge editing and forgetting — Das et al. (2024). Like the delta rule, its write corrects what is already stored, whereas our Hebbian carry only adds.
- (Discussion) Its explicit mechanisms for selective forgetting and leakage prevention show what a deployable carried memory would need, since our carry has none and a state carried across users could leak information.

## Verification
- Read the abstract, introduction and contributions (PDF pages 1–2) and searched the method text for the memory update (least-squares / pseudo-inverse lines read directly via pdftotext). The editing benchmarks, tables and appendix were not read. The arXiv PDF shows no venue; the ICML 2024 venue was confirmed on PMLR (https://proceedings.mlr.press/v235/das24a.html).
