# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.667  [+1.347, +2.074]  *  +12.99%  1.00
within          ALL        106  +0.053  [+0.040, +0.069]  *   +0.48%  0.93
between_slices  ALL        106  +0.104  [+0.081, +0.133]  *   +0.94%  0.97
total           ALL        106  +1.824  [+1.483, +2.261]  *  +14.21%  1.00
cross_doc       ALL        100  +0.008  [+0.003, +0.013]  *   +0.07%  0.58
cross_slice     ALL        106  +0.104  [+0.084, +0.133]  *   +0.94%  0.97
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.66 over 6 sources
- cross_slice: ρ = +0.14 over 6 sources
- within: ρ = +0.37 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
