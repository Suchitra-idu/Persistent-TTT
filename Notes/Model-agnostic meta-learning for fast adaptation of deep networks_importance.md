# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

**Citation:** Chelsea Finn, Pieter Abbeel, Sergey Levine — Proceedings of the 34th ICML, PMLR 70, 2017 (arXiv:1703.03400v3, 18 Jul 2017).

## Important parts for my research
- MAML meta-learns an initialization θ such that a small number of gradient steps on a new task's small dataset produces good generalization: inner update θ'_i = θ − α∇_θ L_{T_i}(f_θ) (one or more steps); outer/meta update θ ← θ − β∇_θ Σ_i L_{T_i}(f_{θ'_i}) — evaluated at post-adaptation parameters (Eq. 1, Algorithm 1).
- Framed as maximizing sensitivity of new-task loss to θ: MAML finds a point in parameter space where small, task-specific gradient steps produce large improvements (dynamical-systems view).
- Requires backprop through the gradient step (second derivatives / Hessian-vector products). A first-order approximation (drop second-order term) gives nearly identical MiniImageNet accuracy (48.07±1.75% vs 48.70±1.84% 1-shot) with ~33% less compute — most of the benefit comes from gradients evaluated at post-update θ', not from the second-order correction.
- Continues to improve with additional gradient steps at meta-test time without overfitting to the small K-shot set (sinusoid regression, Fig. 3) — contrasts with a pretrained-then-fine-tuned baseline, which does not extrapolate.
- Multi-task parameter averaging (500 separately trained models) underperforms output-space pretraining-on-all-tasks, which itself underperforms MAML (5-shot sinusoid MSE: multi-task 2.19–3.85 vs pretrain-on-all 2.19–2.41 vs MAML 0.35–0.67 across gradient-step counts) — parameter averaging destroys per-task solutions.
- Results: Omniglot 5-way 1-shot 98.7±0.4%, 5-shot 99.9±0.1%; MiniImageNet 5-way 1-shot 48.70±1.84%, 5-shot 63.11±0.92%. RL (half-cheetah/ant, TRPO): MAML adapts velocity/direction in 1–2 gradient steps; pretraining is sometimes worse than random initialization.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) MAML as the canonical gradient-based meta-learning approach to fast within-task adaptation — contrast against our fast-weight/carry approach: MAML meta-learns an initialization exploited by ordinary SGD steps at test time (no persistent parametric memory carried forward across independent tasks/documents), whereas our carry persists an actual fast-weight state across independently-bounded documents.
- (Methodology) MAML's "continues to improve with more gradient steps, no overfitting" diagnostic (Fig. 3) is a useful contrast case for our finding that carry does *not* grow over repeats — MAML's compounding improvement under repeated gradient steps is the positive control our null result lacks.
- (Discussion — framing null result) The first-order-approximation finding — most of MAML's benefit comes from gradient direction at the adapted point, not the second-order meta-gradient term — is a precedent for isolating which mechanism actually drives an adaptation method's gains, relevant to isolating carry's contribution via the carry-reset baseline.

## Verification
- Equations 1–4, Algorithms 1–3, and all numerical results (Table 1, Tables 2–5, Figs. 2–7) were read directly from the PDF. No inferred/uncited figures used.
