# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure               source  n_docs    Δppl            95% CI        rel   win
lora                  ALL        106  +0.772  [+0.631, +0.954]  *  +6.65%  1.00
within                ALL        106  +0.058  [+0.046, +0.073]  *  +0.53%  0.96
between_slices        ALL        106  +0.012  [+0.008, +0.018]  *  +0.11%  0.74
total                 ALL        106  +0.842  [+0.688, +1.043]  *  +7.25%  1.00
cross_doc             ALL         24  -0.002  [-0.006, +0.002]     -0.01%  0.38
cross_slice           ALL         30  +0.015  [+0.006, +0.029]  *  +0.10%  0.73
repeat_9_vs_1         ALL        106  +0.000  [-0.001, +0.001]     +0.00%  0.52
same_vs_other_source  ALL         30  +0.002  [-0.000, +0.004]     +0.02%  0.47
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.09 over 6 sources
- cross_slice: ρ = +0.20 over 6 sources
- within: ρ = +0.09 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/repeat_curve.png](figures/repeat_curve.png)
- [figures/stability.png](figures/stability.png)
- [figures/swap_matrix.png](figures/swap_matrix.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
