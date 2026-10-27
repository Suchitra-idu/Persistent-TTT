# Can a Language Model Continually Learn Facts in its Weights?

**Citation:** Charles O'Neill, Max Kirkby, Jonathon Liu, Michael Psenka (Base Labs). arXiv:2607.11020v3 [cs.CL], 5 Oct 2026. NeurIPS 2026 Workshop on Continual Learning in the Era of Foundation Models and Embodied Agents (per the PDF footer).

## Important parts for my research
- Studies continual learning in the **iterative regime**: 20 invented facts are written one at a time, each into "a Qwen3 model already modified by previous writes". Each write trains a fresh rank-16 LoRA (or full fine-tuning) that is merged before the next.
- Conditions:
  - Training data: **repeated statement** (the same fact in two formats) vs. **varied examples** (24 restatements, QA, applications, contrasts).
  - Method: SFT vs. off-policy and on-policy distillation.
  - Models: Qwen3-4B and 8B.
- Evaluation is **behavioural**: held-out questions of five types (recall, paraphrase, application, composition, counterfactual), graded strictly and leniently.
  - Original-model floor 4%; fact-in-prompt ceiling 91% (lenient).
  - 95% bootstrap intervals over facts, paired across conditions.
- **Results**:
  - After 20 sequential writes, strict accuracy on facts 1–19 is **1%** with repeated statements and **46%** with varied examples.
  - Distillation gives 13% (off-policy), 27% (on-policy forward KL) and 31% (reverse KL).
  - Retention is set by how the **later** facts are trained: varying the later writes changes retention by **+37.6 points [28.9, 46.3]**; varying the original fact's own data changes it by −2.4 [−9.2, 4.1].
  - The O-LoRA orthogonality penalty does not help (46.7% vs. 44.4%).
- "Behavioral forgetting of an earlier fact does not erase its presence from the log-probabilities". For facts that failed every question, a median of 69% of the write-induced log-probability increase remained (§3.4, Fig. 4).
- The writes are gradient-based and merged; there is no fast weight, test-time memory or document boundary.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Sequentially writing facts into weights with merged LoRA updates retains only 1–46% of earlier facts after 20 writes, and retention depends mostly on how later writes are made — O'Neill et al. (2026). A carried state faces the same interference problem.
- (Discussion) O'Neill et al. (2026) find that log-probability evidence of a written fact can remain after behaviour has failed, so likelihood gains such as ours do not imply usable knowledge.
- (Discussion) Interference from later writes is the parallel to our finding that a carry built over a long chain of documents harms the next document.

## Verification
- Read the abstract, introduction, problem formulation and results (§1–3.4, PDF pages 1–7) via pdftotext. All numbers above (1%/46%, 13/27/31%, +37.6 [28.9, 46.3], −2.4 [−9.2, 4.1], 46.7 vs 44.4, 4% floor, 91% ceiling, 69% median retained signal) were read directly. The discussion, appendices and second fact set were not read.
