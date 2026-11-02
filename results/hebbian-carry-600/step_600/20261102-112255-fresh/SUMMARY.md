# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                       source  n_docs    Δppl            95% CI         rel   win
lora                          ALL         73  +2.141  [+1.684, +2.716]  *  +15.23%  1.00
within                        ALL         73  +0.012  [+0.006, +0.020]  *   +0.10%  0.73
between_slices                ALL         73  +0.125  [+0.086, +0.173]  *   +1.05%  0.93
total                         ALL         73  +2.278  [+1.768, +2.918]  *  +16.20%  1.00
gain_self                     ALL        200  +0.105  [+0.091, +0.119]  *   +0.57%  0.89
gain_related                  ALL        200  +0.086  [+0.071, +0.100]  *   +0.46%  0.85
gain_random                   ALL        200  +0.055  [+0.043, +0.068]  *   +0.30%  0.74
gain_other_source             ALL        200  -0.084  [-0.102, -0.066]  *   -0.45%  0.21
self_vs_related               ALL        200  +0.019  [+0.013, +0.026]  *   +0.11%  0.66
related_vs_random             ALL        200  +0.031  [+0.020, +0.041]  *   +0.17%  0.68
random_vs_other_source        ALL        200  +0.139  [+0.118, +0.162]  *   +0.75%  0.88
gain_generic                  ALL        200  +0.071  [+0.056, +0.085]  *   +0.38%  0.80
gain_in_context               ALL        200  +0.321  [+0.076, +0.624]  *   +1.73%  0.46
gain_chain_1                  ALL        200  +0.054  [+0.041, +0.067]  *   +0.29%  0.76
gain_chain_2                  ALL        200  +0.064  [+0.050, +0.077]  *   +0.34%  0.79
gain_chain_5                  ALL        200  +0.067  [+0.053, +0.082]  *   +0.36%  0.80
gain_chain_10                 ALL        200  +0.071  [+0.058, +0.084]  *   +0.38%  0.81
chain_10_vs_chain_1           ALL        200  +0.017  [+0.009, +0.025]  *   +0.09%  0.64
self_vs_generic               ALL        200  +0.034  [+0.027, +0.042]  *   +0.19%  0.70
self_vs_generic_adjusted      ALL        200  +0.034  [+0.025, +0.045]  *   +0.19%  0.70
related_vs_generic            ALL        200  +0.015  [+0.008, +0.022]  *   +0.08%  0.61
related_vs_generic_adjusted   ALL        200  +0.015  [+0.006, +0.024]  *   +0.08%  0.61
related_vs_random             ALL        200  +0.031  [+0.019, +0.042]  *   +0.17%  0.68
related_vs_random_adjusted    ALL        200  +0.031  [+0.016, +0.045]  *   +0.17%  0.68
carry_vs_in_context           ALL        200  -0.235  [-0.545, +0.006]      -1.29%  0.57
carry_vs_in_context_adjusted  ALL        200  -0.235  [-0.641, +0.066]      -1.29%  0.57
```

## Domain correlates (Spearman vs fresh ppl)

- within: ρ = +0.70 over 5 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
