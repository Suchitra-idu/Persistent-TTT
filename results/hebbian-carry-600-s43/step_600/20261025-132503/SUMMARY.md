# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                 source  n_docs    Δppl            95% CI         rel   win
lora                    ALL        106  +1.967  [+1.593, +2.435]  *  +15.32%  1.00
within                  ALL        106  +0.014  [+0.007, +0.023]  *   +0.13%  0.70
between_slices          ALL        106  +0.092  [+0.069, +0.123]  *   +0.85%  0.97
total                   ALL        106  +2.073  [+1.685, +2.575]  *  +16.15%  1.00
cross_doc               ALL        100  -0.031  [-0.043, -0.021]  *   -0.29%  0.16
cross_slice             ALL        106  +0.092  [+0.071, +0.121]  *   +0.85%  0.97
carry_gain              ALL        106  +0.104  [+0.078, +0.136]  *   +0.97%  0.97
context_gain            ALL        106  +2.776  [+2.357, +3.210]  *  +25.75%  1.00
context_over_carry      ALL        106  +2.672  [+2.292, +3.139]  *  +25.02%  1.00
gain_self               ALL        200  +0.116  [+0.103, +0.130]  *   +0.65%  0.93
gain_related            ALL        200  +0.083  [+0.071, +0.095]  *   +0.46%  0.84
gain_random             ALL        200  +0.034  [+0.023, +0.044]  *   +0.19%  0.68
gain_other_source       ALL        200  -0.046  [-0.058, -0.035]  *   -0.26%  0.29
self_vs_related         ALL        200  +0.033  [+0.025, +0.042]  *   +0.19%  0.68
related_vs_random       ALL        200  +0.049  [+0.038, +0.060]  *   +0.27%  0.74
random_vs_other_source  ALL        200  +0.080  [+0.066, +0.094]  *   +0.45%  0.80
```

## Carry against full context

The carried state recovers 3.8% [3.2%, 4.3%] of the gain from placing the previous slice in attention.

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.37 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.03 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
