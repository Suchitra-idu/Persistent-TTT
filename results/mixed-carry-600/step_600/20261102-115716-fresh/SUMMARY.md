# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                       source  n_docs    Δppl            95% CI         rel   win
lora                          ALL         73  +2.153  [+1.695, +2.731]  *  +15.31%  1.00
within                        ALL         73  +0.003  [-0.001, +0.006]      +0.02%  0.55
between_slices                ALL         73  +0.073  [+0.050, +0.102]  *   +0.61%  0.95
total                         ALL         73  +2.228  [+1.735, +2.847]  *  +15.85%  1.00
gain_self                     ALL        200  +0.117  [+0.104, +0.129]  *   +0.63%  0.94
gain_related                  ALL        200  +0.112  [+0.100, +0.125]  *   +0.61%  0.91
gain_random                   ALL        200  +0.101  [+0.089, +0.115]  *   +0.55%  0.89
gain_other_source             ALL        200  +0.037  [+0.028, +0.046]  *   +0.20%  0.74
self_vs_related               ALL        200  +0.004  [-0.001, +0.009]      +0.02%  0.56
related_vs_random             ALL        200  +0.011  [+0.005, +0.017]  *   +0.06%  0.59
random_vs_other_source        ALL        200  +0.065  [+0.054, +0.075]  *   +0.35%  0.82
gain_generic                  ALL        200  +0.115  [+0.103, +0.127]  *   +0.62%  0.92
gain_in_context               ALL        200  +0.357  [+0.114, +0.660]  *   +1.93%  0.46
gain_chain_1                  ALL        200  +0.104  [+0.093, +0.116]  *   +0.56%  0.93
gain_chain_2                  ALL        200  +0.109  [+0.098, +0.122]  *   +0.59%  0.91
gain_chain_5                  ALL        200  +0.114  [+0.102, +0.127]  *   +0.62%  0.92
gain_chain_10                 ALL        200  +0.111  [+0.099, +0.124]  *   +0.60%  0.91
chain_10_vs_chain_1           ALL        200  +0.007  [+0.003, +0.012]  *   +0.04%  0.55
self_vs_generic               ALL        200  +0.002  [-0.003, +0.007]      +0.01%  0.53
self_vs_generic_adjusted      ALL        200  +0.002  [-0.005, +0.008]      +0.01%  0.53
related_vs_generic            ALL        200  -0.002  [-0.007, +0.003]      -0.01%  0.47
related_vs_generic_adjusted   ALL        200  -0.002  [-0.009, +0.004]      -0.01%  0.47
related_vs_random             ALL        200  +0.011  [+0.005, +0.017]  *   +0.06%  0.59
related_vs_random_adjusted    ALL        200  +0.011  [+0.003, +0.019]  *   +0.06%  0.59
carry_vs_in_context           ALL        200  -0.245  [-0.557, -0.002]  *   -1.35%  0.57
carry_vs_in_context_adjusted  ALL        200  -0.245  [-0.649, +0.059]      -1.35%  0.57
```

## Domain correlates (Spearman vs fresh ppl)

- within: ρ = +0.50 over 5 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
