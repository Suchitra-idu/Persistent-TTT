# Training Compute-Optimal Large Language Models

**Citation:** Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al., 2022, DeepMind (arXiv:2203.15556, published version 2023 — "Chinchilla" paper)

## Important parts for my research
- Central finding: for a fixed FLOPs budget, model size (N) and training tokens (D) should scale in equal proportion (N_opt ∝ C^0.5, D_opt ∝ C^0.5 across three independent estimation methods), contradicting Kaplan et al. (2020) which recommended model size scale faster (a=0.73) than data (b=0.27).
- Chinchilla (70B params, 1.4T tokens) uses the same compute budget as Gopher (280B params, 300B tokens) — i.e., 4x smaller model, 4x more tokens — and uniformly outperforms Gopher, GPT-3 (175B), Jurassic-1 (178B), and MT-NLG (530B) on a large range of downstream tasks (MMLU 67.6% 5-shot vs Gopher's 60.0%, a 7.6% absolute improvement).
- Table 3 gives concrete compute-optimal token counts by parameter count relevant to the proposal's target sizes: a 1B-parameter model is optimally trained on ~20.2B tokens (1.21e20 FLOPs); a 10B-parameter model on ~205.1B tokens. No exact entries for 0.6B–4B, but this establishes the reference ratio (~20 tokens/parameter) for judging whether SlimPajama-6B (a ~6B-token corpus) is adequate or undersized relative to Qwen3 0.6B–4B targets.
- Parametric loss form: L̂(N,D) = E + A/N^α + B/D^β — an entropy floor term (E) plus separate terms for model-size-limited underperformance and finite-optimization-steps underperformance. Not directly about fast weights, but a useful template for describing loss decomposition if the paper wants to argue carry contributes a bounded, saturating term rather than unbounded improvement.
- Notes that current LLMs are "significantly undertrained" relative to their parameter count — a general scaling caution but not specific to TTT/persistence mechanisms.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Model size and training-data volume should be scaled in tandem for compute-optimal pretraining; the ratio implied by Hoffmann et al. (2022) suggests small-scale TTT experiments on models in the 0.6B–4B range should be checked against comparably scaled token budgets" (Methodology — justifying model/dataset scale choices).
- "Chinchilla's 4x-smaller, 4x-longer-trained model outperforming Gopher demonstrates that undertrained large models leave headroom that better data exposure — not just added parameters — can close, a caution against over-interpreting small perplexity gains as capability gains without accounting for training-token adequacy" (Discussion, if used to caveat magnitude of observed effects).

## Verification
- Directly verified from PDF text: N_opt ∝ C^0.50, D_opt ∝ C^0.50 (Table 2, Approach 1); Chinchilla 70B/1.4T tokens vs Gopher 280B/300B tokens (Table 1, Section 4); MMLU Chinchilla 67.6% vs Gopher 60.0% (Table 6); parametric loss equation L̂(N,D)=E+A/N^α+B/D^β (Eq. 2); Table 3 entries for 1B (20.2B tokens) and 10B (205.1B tokens) parameters read directly.
- No specific entry exists in the paper for 0.6B–4B parameter models — the "~20 tokens/parameter" comparison to SlimPajama-6B is my inference from interpolating Table 3, not a number stated directly in the paper. Flagged as inference, not a verified paper claim.
