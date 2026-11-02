# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                       source  n_docs    Δppl            95% CI         rel   win
lora                          ALL         73  +2.161  [+1.699, +2.743]  *  +15.37%  1.00
within                        ALL         73  +0.002  [-0.000, +0.005]      +0.02%  0.53
between_slices                ALL         73  +0.087  [+0.062, +0.119]  *   +0.73%  0.92
total                         ALL         73  +2.250  [+1.752, +2.877]  *  +16.01%  1.00
gain_self                     ALL        200  +0.111  [+0.099, +0.123]  *   +0.60%  0.95
gain_related                  ALL        200  +0.102  [+0.090, +0.115]  *   +0.55%  0.89
gain_random                   ALL        200  +0.083  [+0.072, +0.095]  *   +0.45%  0.84
gain_other_source             ALL        200  +0.004  [-0.005, +0.014]      +0.02%  0.53
self_vs_related               ALL        200  +0.009  [+0.003, +0.014]  *   +0.05%  0.60
related_vs_random             ALL        200  +0.019  [+0.012, +0.027]  *   +0.10%  0.64
random_vs_other_source        ALL        200  +0.079  [+0.067, +0.090]  *   +0.42%  0.86
gain_generic                  ALL        200  +0.103  [+0.091, +0.115]  *   +0.56%  0.91
gain_in_context               ALL        200  +0.331  [+0.087, +0.632]  *   +1.79%  0.45
gain_chain_1                  ALL        200  +0.084  [+0.074, +0.096]  *   +0.46%  0.86
gain_chain_2                  ALL        200  +0.095  [+0.084, +0.107]  *   +0.52%  0.89
gain_chain_5                  ALL        200  +0.099  [+0.088, +0.111]  *   +0.54%  0.88
gain_chain_10                 ALL        200  +0.099  [+0.088, +0.111]  *   +0.53%  0.91
chain_10_vs_chain_1           ALL        200  +0.014  [+0.008, +0.021]  *   +0.08%  0.66
self_vs_generic               ALL        200  +0.007  [+0.001, +0.013]  *   +0.04%  0.58
self_vs_generic_adjusted      ALL        200  +0.007  [-0.000, +0.015]      +0.04%  0.58
related_vs_generic            ALL        200  -0.001  [-0.007, +0.004]      -0.01%  0.46
related_vs_generic_adjusted   ALL        200  -0.001  [-0.008, +0.006]      -0.01%  0.46
related_vs_random             ALL        200  +0.019  [+0.012, +0.027]  *   +0.10%  0.64
related_vs_random_adjusted    ALL        200  +0.019  [+0.010, +0.029]  *   +0.10%  0.64
carry_vs_in_context           ALL        200  -0.229  [-0.538, +0.012]      -1.26%  0.56
carry_vs_in_context_adjusted  ALL        200  -0.229  [-0.631, +0.070]      -1.26%  0.56
```

## Domain correlates (Spearman vs fresh ppl)

- within: ρ = +0.30 over 5 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
