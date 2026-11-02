# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                       source  n_docs    Δppl            95% CI         rel   win
lora                          ALL         73  +2.129  [+1.674, +2.702]  *  +15.15%  1.00
within                        ALL         73  +0.007  [+0.003, +0.013]  *   +0.06%  0.66
between_slices                ALL         73  +0.118  [+0.082, +0.163]  *   +0.99%  1.00
total                         ALL         73  +2.254  [+1.751, +2.886]  *  +16.03%  1.00
gain_self                     ALL        200  +0.117  [+0.105, +0.130]  *   +0.63%  0.94
gain_related                  ALL        200  +0.086  [+0.075, +0.098]  *   +0.47%  0.86
gain_random                   ALL        200  +0.036  [+0.025, +0.047]  *   +0.20%  0.69
gain_other_source             ALL        200  -0.059  [-0.071, -0.049]  *   -0.32%  0.24
self_vs_related               ALL        200  +0.031  [+0.023, +0.039]  *   +0.17%  0.70
related_vs_random             ALL        200  +0.050  [+0.039, +0.061]  *   +0.27%  0.72
random_vs_other_source        ALL        200  +0.096  [+0.081, +0.111]  *   +0.52%  0.86
gain_generic                  ALL        200  +0.068  [+0.057, +0.080]  *   +0.37%  0.83
gain_in_context               ALL        200  +0.310  [+0.064, +0.612]  *   +1.67%  0.45
gain_chain_1                  ALL        200  +0.034  [+0.023, +0.045]  *   +0.18%  0.72
gain_chain_2                  ALL        200  +0.049  [+0.037, +0.060]  *   +0.26%  0.75
gain_chain_5                  ALL        200  +0.060  [+0.049, +0.071]  *   +0.32%  0.78
gain_chain_10                 ALL        200  +0.060  [+0.048, +0.071]  *   +0.32%  0.80
chain_10_vs_chain_1           ALL        200  +0.026  [+0.018, +0.034]  *   +0.14%  0.70
self_vs_generic               ALL        200  +0.049  [+0.038, +0.060]  *   +0.26%  0.76
self_vs_generic_adjusted      ALL        200  +0.049  [+0.035, +0.063]  *   +0.26%  0.76
related_vs_generic            ALL        200  +0.018  [+0.009, +0.026]  *   +0.10%  0.62
related_vs_generic_adjusted   ALL        200  +0.018  [+0.007, +0.029]  *   +0.10%  0.62
related_vs_random             ALL        200  +0.050  [+0.039, +0.061]  *   +0.27%  0.72
related_vs_random_adjusted    ALL        200  +0.050  [+0.036, +0.064]  *   +0.27%  0.72
carry_vs_in_context           ALL        200  -0.224  [-0.533, +0.017]      -1.23%  0.58
carry_vs_in_context_adjusted  ALL        200  -0.224  [-0.626, +0.076]      -1.23%  0.58
```

## Domain correlates (Spearman vs fresh ppl)

- within: ρ = +0.00 over 5 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
