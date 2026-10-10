# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI        rel   win
lora            ALL        106  +0.475  [+0.389, +0.587]  *  +4.07%  1.00
within          ALL        106  +0.016  [+0.005, +0.028]  *  +0.14%  0.72
between_slices  ALL        106  +0.014  [+0.008, +0.021]  *  +0.12%  0.70
total           ALL        106  +0.504  [+0.407, +0.628]  *  +4.32%  1.00
cross_doc       ALL         24  +0.008  [-0.004, +0.023]     +0.05%  0.50
cross_slice     ALL         30  +0.014  [+0.004, +0.030]  *  +0.10%  0.83
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.37 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.26 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
