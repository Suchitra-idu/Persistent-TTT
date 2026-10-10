# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.670  [+1.349, +2.074]  *  +13.01%  1.00
within          ALL        106  +0.144  [+0.117, +0.179]  *   +1.29%  1.00
between_slices  ALL        106  +0.027  [+0.019, +0.037]  *   +0.25%  0.83
total           ALL        106  +1.842  [+1.489, +2.284]  *  +14.34%  1.00
cross_doc       ALL         24  +0.003  [-0.005, +0.012]      +0.02%  0.58
cross_slice     ALL         30  +0.027  [+0.013, +0.049]  *   +0.19%  0.77
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.66 over 6 sources
- cross_slice: ρ = +0.03 over 6 sources
- within: ρ = +0.83 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
