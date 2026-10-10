# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.909  [+1.542, +2.364]  *  +14.87%  1.00
within          ALL        106  +0.059  [+0.047, +0.073]  *   +0.54%  0.99
between_slices  ALL        106  +0.007  [+0.003, +0.012]  *   +0.07%  0.65
total           ALL        106  +1.975  [+1.601, +2.442]  *  +15.38%  1.00
cross_doc       ALL         24  +0.006  [-0.000, +0.012]      +0.04%  0.67
cross_slice     ALL         30  +0.002  [-0.004, +0.008]      +0.01%  0.60
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.20 over 6 sources
- cross_slice: ρ = +0.49 over 6 sources
- within: ρ = +0.66 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
