# Meta-Learning in Neural Networks: A Survey

**Citation:** Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey — University of Edinburgh / Samsung AI Centre, arXiv:2004.05439v2 [cs.LG], 7 Nov 2020.

## Important parts for my research
- Formalizes meta-learning as bilevel optimization: outer loop learns meta-knowledge ω (e.g. initialization, optimizer, architecture), inner loop solves task-specific θ*(ω) given ω fixed (Eq. 5–6). TTT/fast-weight adaptation at inference time is a degenerate single-task instance of this inner loop, with the base pretrained weights playing the role of ω.
- Proposes a three-axis taxonomy: meta-representation ("what" is learned — initialization, optimizer, embedding, architecture, hyperparameters), meta-optimizer ("how" — gradient, RL, evolution), meta-objective ("why" — few/many-shot, fast adaptation vs asymptotic performance, online vs offline). Useful vocabulary for positioning session-persistent TTT: our ω is the base LM weights, our meta-representation is the fast-weight update itself, meta-objective is "fast per-document adaptation with a persistence-vs-reset choice."
- Distinguishes optimization-based (MAML-style, explicit inner-loop gradient steps) from feed-forward/black-box (amortized, single forward pass) from metric-based methods — TTT-Linear/MLP and our Hebbian carry are optimization-based in this taxonomy, since the "inner loop" is literal gradient/Hebbian weight updates on new data, but with none of it backpropagated through an outer meta-objective (unlike MAML).
- Continual Learning (CL) section (5.7): explicitly notes CL "seeks to accelerate learning of new tasks... without forgetting old tasks" and that meta-objectives for CL are an active but still nascent research area — situates our carry-across-documents evaluation as adjacent to continual learning but distinct (single model, no explicit task boundaries or replay).
- Meta-overfitting / meta-generalization (Sec 6) discussion: models can fail to generalize from meta-train to meta-test tasks, especially when the number of "tasks" is small — a cautionary framing applicable to our small number of held-out replay documents.
- "Task families" and narrow vs diverse task distributions (Sec 6) are flagged as a major open challenge — relevant to our domain-comparison research question (Q2), since meta-learning literature shows performance is highly sensitive to how diverse/narrow the task/document distribution is.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / framing) General bilevel meta-learning formalization (Hospedales et al. 2020) as the broader category TTT/fast-weight methods belong to, distinguishing our approach as optimization-based with no explicit outer meta-objective (unlike MAML-based approaches).
- (Discussion) Their meta-generalization / narrow-task-family caution supports treating a null/flat carry result as a plausible, well-precedented outcome rather than an implementation bug, given known difficulties of generalizing across diverse task/document distributions.

## Verification
- Taxonomy, bilevel formalization (Eq. 5–6), and continual-learning/meta-generalization sections read directly from Sections 2–3 and 5.7/6 of the PDF. No numeric results are claimed from this paper (survey, not empirical), so no numbers to fact-check.
