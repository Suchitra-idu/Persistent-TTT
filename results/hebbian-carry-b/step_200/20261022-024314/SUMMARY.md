# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.708  [+1.380, +2.121]  *  +13.30%  1.00
within          ALL        106  +0.018  [+0.011, +0.027]  *   +0.16%  0.74
between_slices  ALL        106  +0.099  [+0.074, +0.133]  *   +0.89%  0.92
total           ALL        106  +1.825  [+1.480, +2.265]  *  +14.21%  1.00
cross_doc       ALL         24  +0.002  [-0.018, +0.025]      +0.02%  0.58
cross_slice     ALL         30  +0.119  [+0.065, +0.202]  *   +0.82%  0.93
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.20 over 6 sources
- cross_slice: ρ = +0.20 over 6 sources
- within: ρ = +0.14 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
