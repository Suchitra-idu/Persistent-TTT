# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.707  [+1.380, +2.124]  *  +13.30%  1.00
within          ALL        106  +0.025  [+0.015, +0.035]  *   +0.22%  0.83
between_slices  ALL        106  +0.008  [+0.001, +0.015]  *   +0.07%  0.75
total           ALL        106  +1.740  [+1.409, +2.158]  *  +13.55%  1.00
cross_doc       ALL        100  +0.003  [+0.001, +0.006]  *   +0.03%  0.62
cross_slice     ALL        106  +0.008  [+0.002, +0.015]  *   +0.07%  0.75
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.49 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.66 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
