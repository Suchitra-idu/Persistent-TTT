# Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training

**Citation:** Xiangchen Song, Zhenhao Chen, Lingjing Kong, Shaoan Xie, Xinshuai Dong, Guangyi Chen, Kun Zhang (Carnegie Mellon University, MBZUAI). arXiv:2607.00368v1 [cs.CL], 1 Jul 2026. Preprint, no venue stated.

## Important parts for my research
- Names the failure mode **"evidence migration"**: "evidence that directly supports one evaluation regime is carried into a stronger deployment narrative without the behavioral tests required for that narrative" (§1). This is the exact risk for our paper if perplexity results are described as memory.
- Distinguishes **in-sequence adaptation** (update on a support chunk, evaluate on future tokens of the same stream) from **deployment-time behavioral learning** (sparse facts, preferences, corrections, procedures used later after the context is removed) (§1).
- **Three claim levels** (§4, Fig. 2):
  - **S-level**, stream or domain adaptation: updated on a recent stream, target-domain data, retrieved examples or task attempts, and evaluated by "nearby loss, same-stream performance, domain accuracy, long-context performance, or reward". Evidence: PPL/loss, future-token gain, long-context accuracy, throughput, reward.
  - **B-level**, bridge or internalization: "information can be stored, compressed, or transformed inside model state, but does not yet establish sparse delayed user-facing behavior". Evidence: answer NLL or ranking after support removal, needle recall, ablation tests.
  - **D-level**, deployment-time behavioral learning: recall, paraphrase, delay, locality, conflict, action use after context removal.
  - Each paper gets "the strongest claim level directly supported by its reported evaluations instead of the level suggested by its broadest motivating statement".
- **Audit** of 24 coded papers (§4, Table 2): In-Place TTT, TTT layers, LaCT, TTT-E2E and dynamic evaluation are coded **S-level**; MemoryLLM, PERK, Locas, SEAL, Absorber LLM are **B-level**; assistant-memory benchmarks (LoCoMo, LongMemEval, MemoryBench) are **D targets**.
- **Why perplexity is a narrower question** (§2):
  - dense same-stream evidence reflects "continuation fitting, topic priming, or memorization of nearby regularities, which is precisely what dynamic evaluation was designed to exploit";
  - teacher forcing is easier than behavioural extraction;
  - "A stream protocol can conflate immediate adaptation to the current support item with cumulative adaptation from all preceding chunks", so **reset-vs-stream comparisons** are needed.
- **Diagnostic** (§5, Table 3):
  - Setup: Qwen3-1.7B/4B/8B, one-step LoRA update on nonce access-code facts, 48 probes per model.
  - Losses improve: support ΔNLL −0.97 to −1.22; answer ΔNLL −0.46 to −0.71.
  - Greedy generated recall is **0.0%** for direct, paraphrased and delayed questions at all three scales.
  - A stronger "stress" update raises recall to 72.9% (direct and delayed) and 54.2% (paraphrased), but locality falls from 97.9% to 9.7%.
- **Protocol** (§6, Table 4):
  - Future-token loss only supports "Improves stream adaptation, target-domain adaptation, or task performance under the stated objective".
  - Matched baselines should include a no-update model, exact-context or long-context prompting, and retrieval or memory, matched on "update tokens, trainable parameters, optimization steps, latency, memory footprint, context length".
- Discussion (§7): "perplexity is already behavioral for language models" is "reasonable in dense-stream settings"; the claim only fails when used for sparse user memory.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Introduction / Scope) State our claim level explicitly: our evidence is S-level (stream and domain adaptation measured by perplexity), and we make no deployment-memory claim — Song et al. (2026).
- (Related Work) In-Place TTT and related TTT layers are coded as S-level evidence by Song et al. (2026); our carried state extends the same evidence type across documents.
- (Methodology) The reset-vs-stream comparison recommended by Song et al. (2026) is exactly our reset-per-document vs carried arm.
- (Methodology / Discussion) The no-update, in-context (exact-context) and memory baselines with matched budgets are the controls a memory-like claim needs.
- (Limitations) Perplexity gains can coexist with zero behavioural recall (Table 3), so our results do not show that the carry stores retrievable facts.

## Verification
- Read the full main text (pages 1–10, sections 1–8) from the PDF via pdftotext. All quotes and numbers above (claim-level definitions, Table 2 codings, Table 3 ΔNLL and 0.0% recall, the 72.9%/54.2%/97.9%→9.7% stress numbers, Table 4 rows) were read directly. Appendices A–F (search accounting, extra diagnostics) were not read.
