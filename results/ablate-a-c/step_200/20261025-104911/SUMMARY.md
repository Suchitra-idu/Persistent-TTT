# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.736  [+1.404, +2.155]  *  +13.52%  1.00
within          ALL        106  +0.023  [+0.012, +0.037]  *   +0.21%  0.74
between_slices  ALL        106  +0.023  [+0.013, +0.034]  *   +0.21%  0.85
total           ALL        106  +1.782  [+1.447, +2.213]  *  +13.88%  1.00
cross_doc       ALL        100  +0.011  [+0.006, +0.016]  *   +0.10%  0.68
cross_slice     ALL        106  +0.023  [+0.014, +0.035]  *   +0.21%  0.85
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.54 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.26 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
