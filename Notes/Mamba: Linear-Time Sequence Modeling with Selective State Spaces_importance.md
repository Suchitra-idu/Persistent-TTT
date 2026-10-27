# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Citation:** Albert Gu, Tri Dao — CMU / Princeton, arXiv:2312.00752v2 [cs.LG], 31 May 2024.

## Important parts for my research
- Core mechanism: makes SSM parameters (Δ, B, C) input-dependent ("selective"), breaking linear time-invariance (LTI) so the hidden state h_t can be written/erased content-adaptively rather than by fixed dynamics — direct architectural analogue to what a carried fast weight would need in order to be useful across boundaries.
- Theorem 1: with N=1, A=-1, B=1, selective SSM recurrence reduces to an RNN gate h_t=(1-g_t)h_{t-1}+g_t x_t, g_t=σ(Linear(x_t)) — Δ plays the role of a generalized gate; large Δ resets state and focuses on current input, small Δ persists state and ignores input. This is a clean formal vocabulary for "carry vs. reset" that maps onto our carried_decay / clip_tau knobs.
- **Boundary Resetting (Sec 3.5.2, directly relevant to explaining flat/non-compounding carry):** "Selective SSMs can reset their state at boundaries... these settings may occur artificially (e.g. packing documents together)... LTI models will bleed information between sequences" while selective models can filter it out. This is an explicit mechanistic account of state persistence vs. reset across document boundaries — the same problem our carry-reset ablation targets, but Mamba's selection mechanism can *choose* to reset while a Hebbian (non-selective, pure outer-product) fast-weight update has no such gating and thus cannot distinguish useful cross-document signal from noise.
- Ablations (Table 7): making Δ selective alone is the single largest driver of improvement (10.93→9.81 perplexity); adding selective B, C on top gives further but smaller gains (→8.71) — evidence that selectivity/gating, not mere presence of a recurrent update, is what makes an accumulating state useful. This is suggestive: an update rule without error-correction or selectivity (our Hebbian, no delta rule) plausibly explains flat carry.
- "Filtering Context": models without selection cannot ignore irrelevant history, so performance does not improve monotonically with context — cited explanation for why many sequence models don't benefit from more context, relevant to why naive carry may not compound.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains flat carry) Gu & Dao (2024)'s selection-mechanism argument: state updates lacking input-dependent selectivity/gating cannot filter relevant from irrelevant signal, offering a mechanistic account for why our pure-Hebbian (non-selective, non-error-correcting) carry stays flat rather than compounding.
- (Related Work) Boundary-resetting discussion directly parallels our document-boundary carry-vs-reset framing; Mamba treats it as an architectural choice (Δ→∞), we treat it as an empirical ablation on fast weights.
- (Methodology) Δ-as-gate formalization (Theorem 1) is a citable reference point for describing carry/decay hyperparameters in fast-weight terms.

## Verification
- Theorem 1 equation, boundary-resetting quote, and Table 7 ablation numbers checked directly against PDF text (pages 8–9, 16-17 of paper). No inference beyond direct reading.
