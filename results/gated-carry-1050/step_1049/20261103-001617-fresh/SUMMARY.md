# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                       source  n_docs    Δppl            95% CI         rel   win
lora                          ALL         73  +2.177  [+1.713, +2.761]  *  +15.49%  1.00
within                        ALL         73  +0.009  [+0.002, +0.016]  *   +0.07%  0.64
between_slices                ALL         73  +0.152  [+0.107, +0.207]  *   +1.28%  0.99
total                         ALL         73  +2.337  [+1.818, +2.994]  *  +16.63%  1.00
gain_self                     ALL        200  +0.135  [+0.120, +0.150]  *   +0.73%  0.92
gain_related                  ALL        200  +0.102  [+0.087, +0.117]  *   +0.55%  0.86
gain_random                   ALL        200  +0.021  [+0.006, +0.036]  *   +0.11%  0.59
gain_other_source             ALL        200  -0.109  [-0.128, -0.091]  *   -0.59%  0.18
self_vs_related               ALL        200  +0.033  [+0.022, +0.044]  *   +0.18%  0.68
related_vs_random             ALL        200  +0.081  [+0.066, +0.097]  *   +0.44%  0.78
random_vs_other_source        ALL        200  +0.130  [+0.111, +0.150]  *   +0.70%  0.85
gain_generic                  ALL        200  +0.065  [+0.050, +0.079]  *   +0.35%  0.74
gain_in_context               ALL        200  +0.311  [+0.066, +0.613]  *   +1.69%  0.45
gain_chain_1                  ALL        200  +0.013  [-0.002, +0.028]      +0.07%  0.58
gain_chain_2                  ALL        200  +0.035  [+0.020, +0.050]  *   +0.19%  0.67
gain_chain_5                  ALL        200  +0.051  [+0.037, +0.065]  *   +0.28%  0.72
gain_chain_10                 ALL        200  +0.055  [+0.040, +0.069]  *   +0.30%  0.74
chain_10_vs_chain_1           ALL        200  +0.042  [+0.030, +0.054]  *   +0.23%  0.70
self_vs_generic               ALL        200  +0.070  [+0.055, +0.085]  *   +0.38%  0.81
self_vs_generic_adjusted      ALL        200  +0.070  [+0.052, +0.089]  *   +0.38%  0.81
related_vs_generic            ALL        200  +0.037  [+0.024, +0.050]  *   +0.20%  0.69
related_vs_generic_adjusted   ALL        200  +0.037  [+0.021, +0.054]  *   +0.20%  0.69
related_vs_random             ALL        200  +0.081  [+0.065, +0.098]  *   +0.44%  0.78
related_vs_random_adjusted    ALL        200  +0.081  [+0.061, +0.102]  *   +0.44%  0.78
carry_vs_in_context           ALL        200  -0.209  [-0.518, +0.032]      -1.15%  0.59
carry_vs_in_context_adjusted  ALL        200  -0.209  [-0.611, +0.089]      -1.15%  0.59
```

## Domain correlates (Spearman vs fresh ppl)

- within: ρ = -0.10 over 5 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
