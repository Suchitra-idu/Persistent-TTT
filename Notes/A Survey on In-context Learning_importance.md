# A Survey on In-context Learning

**Citation:** Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, Baobao Chang, Xu Sun, Lei Li, Zhifang Sui, 2024, arXiv:2301.00234v6 (Peking University et al.)

## Important parts for my research
- Formal definition: ICL is training-free — the model M takes a demonstration context C (task instruction + k examples) and query x, and predicts ŷ = argmax_yj P(yj | C, x) with no backward-gradient parameter update (Eq. 1-2, following Brown et al. 2020). This is the key contrast point with TTT: ICL conditions on context in the forward pass only, whereas TTT explicitly updates weights via a self-supervised loss.
- "Gradient Descent View" (§5.2.2): Dai et al. (2023a) show a dual form between Transformer attention and gradient descent, arguing GPT-style ICL behaves like implicit fine-tuning; von Oswald et al. (2023) show linear attention with hand-constructed weights reproduces models learned by gradient descent. This is the standard citation for "attention implicitly performs a weight update," useful for framing fast-weight/TTT mechanisms as an explicit, mechanistic version of what ICL does implicitly.
- Reported finding: increasing the number of in-context demonstrations does not reliably improve performance and can be detrimental (cited under Long-context ICL, §7) — a directly analogous phenomenon to a flat/non-compounding carry, worth noting as a parallel "more context ≠ better" result in a different mechanism.
- Bayesian view: ICL as implicit Bayesian inference over a latent task concept (Xie et al. 2022) — an alternative theoretical lens distinct from the gradient-descent view.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Definitional distinction that ICL performs no parameter update (Eq. 1-2) — positions TTT/fast-weight carry as mechanistically different from and complementary to ICL.
- (Related Work / Discussion) Dai et al. (2023a) / von Oswald et al. (2023) dual-form-of-attention-as-gradient-descent results, as the established link between attention and implicit weight updates that motivates explicit fast-weight TTT.
- (Discussion — explains flat carry, by analogy) The finding that more ICL demonstrations do not monotonically help and can hurt, as a parallel non-monotonic-context phenomenon from a different literature.

## Verification
- Definition equations, gradient-descent-view claims, and the long-context ICL degradation finding were read directly from PDF pages 2 and 7-8. Paper is tangential to core research (no TTT/fast-weight/carry content); scope kept accordingly narrow.
