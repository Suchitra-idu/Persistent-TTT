# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.764  [+1.425, +2.191]  *  +13.74%  1.00
within          ALL        106  +0.011  [+0.005, +0.017]  *   +0.10%  0.71
between_slices  ALL        106  +0.015  [+0.010, +0.022]  *   +0.14%  0.79
total           ALL        106  +1.789  [+1.455, +2.222]  *  +13.94%  1.00
cross_doc       ALL        100  +0.004  [+0.001, +0.007]  *   +0.04%  0.63
cross_slice     ALL        106  +0.015  [+0.010, +0.022]  *   +0.14%  0.79
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.60 over 6 sources
- cross_slice: ρ = +0.54 over 6 sources
- within: ρ = +0.31 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
