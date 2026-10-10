# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.353  [+1.090, +1.683]  *  +10.54%  1.00
within          ALL        106  +0.276  [+0.223, +0.342]  *   +2.40%  1.00
between_slices  ALL        106  +0.051  [+0.038, +0.067]  *   +0.46%  0.93
total           ALL        106  +1.680  [+1.360, +2.092]  *  +13.08%  1.00
cross_doc       ALL         24  -0.002  [-0.013, +0.008]      -0.01%  0.50
cross_slice     ALL         30  +0.062  [+0.034, +0.110]  *   +0.42%  0.97
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.37 over 6 sources
- cross_slice: ρ = +0.03 over 6 sources
- within: ρ = +0.94 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
