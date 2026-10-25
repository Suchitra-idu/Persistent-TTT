# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.680  [+1.359, +2.086]  *  +13.09%  1.00
within          ALL        106  +0.058  [+0.043, +0.077]  *   +0.52%  0.91
between_slices  ALL        106  +0.035  [+0.022, +0.052]  *   +0.31%  0.82
total           ALL        106  +1.773  [+1.433, +2.203]  *  +13.81%  1.00
cross_doc       ALL        100  +0.014  [+0.007, +0.021]  *   +0.13%  0.72
cross_slice     ALL        106  +0.035  [+0.023, +0.050]  *   +0.31%  0.82
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.31 over 6 sources
- cross_slice: ρ = +0.31 over 6 sources
- within: ρ = +0.54 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
