# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.701  [+1.373, +2.116]  *  +13.25%  1.00
within          ALL        106  +0.027  [+0.017, +0.039]  *   +0.24%  0.83
between_slices  ALL        106  +0.034  [+0.026, +0.045]  *   +0.31%  0.92
total           ALL        106  +1.763  [+1.429, +2.187]  *  +13.73%  1.00
cross_doc       ALL        100  +0.008  [+0.004, +0.012]  *   +0.07%  0.73
cross_slice     ALL        106  +0.034  [+0.026, +0.045]  *   +0.31%  0.92
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.09 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.09 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
