# Learning What to Remember: Test-Time Training via Context Distillation

**Citation:** Zixuan Wang, Xingyu Dang, Rui-Jie Zhu, Zixin Wen, Hengyu Fu, Wenhao Chai, Jason D. Lee (Princeton, UC Berkeley, UC Santa Cruz, Carnegie Mellon University). arXiv:2608.01672v1 [cs.CL], preprint dated 3 Aug 2026.

## Important parts for my research
- Proposes TTCD (Test-Time Context Distillation): a shared frozen backbone runs a long-window "teacher" and short-window "student" over the same chunk; the fast weight is updated to make the student's MLP output match the teacher's (stop-gradient) output, rather than reconstructing raw tokens.
- IP-TTCD is the in-place instantiation: it updates the MLP down-projection matrix W_down exactly as In-Place TTT does, but with the distillation loss instead of reconstruction — same fast-weight site as our proposed extension of In-Place TTT.
- Update rule (chunk i, size C): W_down^(i) = W_down^(i-1) − (η_i/|C_i|)[Y_i^(S) − stopgrad(Y_i^(T))][Z_i^(S)]ᵀ — this is an error-correcting (delta-rule-like) update using the teacher−student residual as the "error" signal, NOT pure Hebbian outer-product.
- Theoretical result (Proposition 1): under stated assumptions, the fast-weight update induced by an earlier in-context episode provably shifts the student's hidden state at a later related query toward the teacher's (in-context) solution — a formal argument for why fast weights can carry useful signal to later positions within a sequence.
- Practical add-ons: causal short convolutions on teacher/student MLP activations plus a learned projection before forming the write key/value (ΔW_down = β·R·Kᵀ), off-policy chunk parallelism (all chunk updates computed from the base W_down^(0) then accumulated causally, avoiding sequential dependency).
- Results (760M, RULER NIAH avg): IP-TTCD 65.79/46.92/41.75/21.96 at 4k/8k/16k/32k vs IP-TTT 59.83/33.88/23.92/9.29 — more than doubles IP-TTT at 32k, gap widens with context length. Sliding-window perplexity on Books (Fig. 2) is consistently lowest for IP-TTCD across 1k–32k at both 340M and 760M, beating DeltaNet, Gated DeltaNet, SWA, and IP-TTT.
- Continual pretraining: applying IP-TTCD to already-pretrained SmolLM2-360M/1.7B and LLaMA-3.1-8B (lightweight augmentation, no from-scratch retrain) improves RULER average accuracy over IP-TTT and plain continual-pretraining baselines at most context lengths, though IP-TTT edges out IP-TTCD at 64K for the 360M model specifically.
- All experiments are within continuous long-context chunking (teacher/student windows inside one sequence) — no cross-document/session persistence is proposed or tested.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) IP-TTCD directly extends In-Place TTT (Feng et al. 2026), reusing the MLP down-projection as the fast-weight site but replacing reconstruction with context distillation; explicitly frames In-Place TTT as "a fundamental step towards connecting temporary memory and long term memory" — good citation for motivating persistent fast weights.
- (Methodology — contrast) Their update is error-correcting/delta-style (teacher−student residual), directly contrastable with our hebbian-only (pure outer-product, no error correction) active update rule — a citable methodological fork.
- (Discussion — explains flat carry) Proposition 1's mechanism (residual-driven write) suggests carry strength depends on a nonzero teacher−student discrepancy; under hebbian-only updates without such an error signal, weaker/flatter carry growth is plausible — cite as a candidate explanation.
- (Research Gap) Confirms nobody in this line (including this Aug-2026 paper) tests persistence across independently-bounded documents/sessions — all evaluation is chunk-wise within one long sequence.

## Verification
- Update equations, Proposition 1 statement, and RULER/sliding-window numbers were read directly from the PDF (pp. 4–9). The 360M-vs-64K anomaly (IP-TTCD 9.79 < IP-TTT 10.35) was cross-checked against Table 2 and is a genuine reading, not an inference — flagged for nuance rather than omitted, since the abstract's "gains widen with context length" claim doesn't hold uniformly at every model scale/length combination.
