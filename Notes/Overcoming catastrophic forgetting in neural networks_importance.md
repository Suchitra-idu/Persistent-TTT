# Overcoming catastrophic forgetting in neural networks

**Citation:** Kirkpatrick, Pascanu, Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis, Clopath, Kumaran, Hadsell — arXiv:1612.00796v2, 25 Jan 2017 (DeepMind). [PDF header shows arXiv only; the paper was later published in PNAS 2017, not stated on this PDF's title page.]

## Important parts for my research
- Elastic Weight Consolidation (EWC): loss L(θ) = L_B(θ) + Σ_i (λ/2)·F_i·(θ_i − θ*_{A,i})², a quadratic penalty anchoring each parameter to its old (task-A) value, scaled by its importance F_i (Eq. 3).
- F is the diagonal of the Fisher information matrix, used as a Laplace approximation to the posterior over θ given past data: log p(θ|D) = log p(D_B|θ) + log p(θ|D_A) − log p(D_B) — i.e., the old task's posterior becomes the new task's prior. F is cheap (first-order derivatives only), guaranteed PSD, and ≈ second derivative of the loss surface near the optimum.
- Sum of two quadratic penalties is itself quadratic, so EWC composes across >2 sequential tasks without added terms.
- Permuted-MNIST: plain SGD catastrophically forgets task A upon training B; L2 (uniform) regularization protects A but blocks learning B (protects all weights equally); EWC learns B while retaining A, scaling to many more tasks than dropout-regularized SGD (~2 permutations).
- Perturbation validation (Fig. 3C): perturbing weights along the inverse-Fisher direction harms performance far less than uniform perturbation, confirming diagonal-Fisher importance estimates are meaningful — but perturbing in the Fisher's "nullspace" (nominally unimportant weights) hurts just as much, indicating the point-estimate diagonal Fisher underestimates parameter uncertainty. Authors flag this as the method's chief limitation.
- Atari/DQN continual-RL experiment: EWC penalty (Fisher multiplier 400) applied only after 20M frames per task; task inferred via an HMM-based generative "forget-me-not" module. EWC lifts total human-normalized score well above plain SGD (which collapses toward 0 after task switches) but stays below separately trained single-task DQNs.
- Runtime is linear in #parameters and #training examples, unlike prior curvature-approximation approaches (French & Chater 2002 recomputes curvature per sample; ELLA needs matrix inversion of size #params).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains carry-stability rationale) EWC's core diagnosis: catastrophic forgetting is unconstrained parameter drift with no importance-weighted anchor to the previously good solution; the quadratic penalty exists precisely to bound this drift.
- (Discussion — framing flat/non-compounding carry) Contrast our Hebbian-only, pure-outer-product fast-weight carry (no error correction, no per-weight importance weighting, no explicit anchor to a prior solution) against EWC's explicitly importance-weighted constraint — motivates why an unconstrained carry might either drift destructively or plateau/self-cancel rather than compound.
- (Related Work) EWC as the canonical solution to catastrophic forgetting via synaptic consolidation, contrasted with our carry-stability check as an empirical (not penalty-based) guard against the same failure mode.
- (Methodology) Fisher-based importance weighting and perturbation-sensitivity testing (Fig. 3C) as a template for validating whether a proposed "important"/protected direction in parameter space actually matters.

## Verification
- Equation 3, the Bayesian posterior derivation (Eq. 1–2), MNIST/Atari results, and the Fisher-overlap/perturbation findings were all read directly from the PDF text (pages 2–7). The PNAS venue is common knowledge but not stated on this PDF's title page, so citation above reports only what's shown (arXiv, DeepMind, 2017).
