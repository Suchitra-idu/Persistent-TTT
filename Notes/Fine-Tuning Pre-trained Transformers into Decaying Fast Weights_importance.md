# Fine-Tuning Pre-trained Transformers into Decaying Fast Weights

**Citation:** Huanru Henry Mao (Jenni), arXiv:2210.04243v1 [cs.LG], 9 Oct 2022 (EMNLP-style formatting, cites as ACL workshop-style refs; venue not stated beyond arXiv).

## Important parts for my research
- Frames kernel-linear-attention recurrent state as a **fast weight**: `S_t ∈ R^{d×m}`, updated by outer product `v_t φ(k_t)^T`, matching Schmidhuber (1992) fast-weight framing — directly analogous to a Hebbian outer-product carry.
- **Additive rule (pure Hebbian, no forgetting)**: `S_t = S_t-1 + v_t φ(k_t)^T` — explicitly stated to be "unable to forget past memories," which "can overwhelm the fixed state capacity and lead to poor performance." This is the closest published analogue to the proposal's pure-Hebbian carry.
- **Gated rule** (Peng et al. 2021): scalar gate `g_t = σ(W_g x_t)` interpolates old/new state uniformly across all elements — paper calls this "problematic as it overwrites all state elements equally without fine-grained control."
- **Delta rule** (Schlag et al. 2021a, the same delta rule referenced in the proposal as "error-correcting"): `S_t = S_t-1 − g_t S_t-1 φ'(k_t)φ'(k_t)^T + g_t v_t φ'(k_t)^T` — removes the old value associated with the current key before writing the new one (true error-correction), requires sum-normalized features and a very small initial gate (`g_t ≈ 0.007`) or training diverges.
- **Their proposed decay rule**: `S_t = G_t ⊗ S_t-1 + v_t φ(k_t)^T` with a *learned low-rank matrix* gate `G_t = σ(W_z x_t+b_z)σ(W_f x_t+b_f)^T` (element-wise, per-state-cell decay, not a single scalar) — sigmoid bounds values for "stable linear recurrent dynamics," citing Miller & Hardt (2019).
- **Key empirical finding (Table 1, best configs, PPL on The Pile val, GPT-2 fine-tuning, state size m=4):** delta rule performs *worst* of all four rules tested (20.1 PPL); the pure additive/Hebbian rule (20.0) actually *outperforms* the delta rule under its best configuration — attributed to delta's "lack of stable recurrent dynamics… during the fine-tuning setting" (citing Chen 2017). Gated (17.1) and decay (17.1) both beat additive and delta.
- Scaling state capacity (Table 2, m=32, no normalization/feature maps): decay rule reaches 14.6 PPL vs GPT-2 dense-attention baseline of 14.5 — "recovers 99% of GPT-2's performance," clearly beating gated (16.4) at the same m.
- Delta rule is fragile: needs sum-normalization of features and near-zero gate init to avoid divergence; without these tricks "the model diverges." Additive/gated rules diverge under different ablation combinations too (marked `*` in Table 1), showing all these fast-weight update rules are sensitive to normalization/init choices.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A pure additive (Hebbian) outer-product fast-weight update is known to have unbounded/uncontrolled state growth and can be overwhelmed by fixed capacity, providing no mechanism to forget or reweight prior contributions (Mao, 2022)." (Discussion — explains flat/non-compounding carry)
- "Prior work fine-tuning attention into recurrent fast-weight states found the error-correcting delta rule (Schlag et al., 2021) to be *more* unstable and worse-performing than a plain additive/Hebbian update in a fine-tuning setting, contrary to the intuition that error-correction should always help (Mao, 2022)." (Related Work / Discussion — nuances the delta-vs-Hebbian comparison)
- "Gated and decay-based fast-weight update rules consistently outperform both plain additive and delta-rule updates for recurrent-state language modeling, suggesting that some form of controlled forgetting, not error-correction alone, is what stabilizes fast-weight carries over long sequences (Mao, 2022)." (Related Work — positions the proposal's hebbian-only, no-decay carry as a known-weaker configuration)

## Verification
- All PPL numbers (Table 1: Add 20.0/Delta 20.1/Gated 17.1/Decay 17.1 under φ✗ ablation; Table 2: Decay m=32 = 14.6 vs GPT-2 14.5) and equations (Eq. 3 additive, Eq. 5 gated, Eq. 6 delta, Eq. 7 decay) were read directly from the PDF text, not inferred.
- The claim that additive "outperforms" delta is a direct quote/paraphrase from Section 4.1.3 ("Surprisingly, under its best configuration, the simple additive rule outperforms the delta rule"), confirmed in text.
- Delta rule's fragility (sum-normalization + g_t≈0.007 init requirement) is from footnote 3 in Section 4, confirmed directly.
