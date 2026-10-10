# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.852  [+1.499, +2.300]  *  +14.42%  1.00
within          ALL        106  +0.014  [+0.009, +0.020]  *   +0.13%  0.79
between_slices  ALL        106  -0.002  [-0.005, +0.001]      -0.02%  0.46
total           ALL        106  +1.864  [+1.517, +2.306]  *  +14.52%  1.00
cross_doc       ALL         24  -0.003  [-0.009, +0.002]      -0.02%  0.50
cross_slice     ALL         30  +0.004  [-0.002, +0.011]      +0.02%  0.60
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.49 over 6 sources
- cross_slice: ρ = +0.31 over 6 sources
- within: ρ = +0.26 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
