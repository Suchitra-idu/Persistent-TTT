# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure               source  n_docs    Δppl            95% CI        rel   win
lora                  ALL        106  +0.650  [+0.533, +0.800]  *  +5.62%  1.00
within                ALL        106  +0.034  [+0.026, +0.045]  *  +0.32%  0.89
between_slices        ALL        106  +0.002  [-0.000, +0.006]     +0.02%  0.54
total                 ALL        106  +0.687  [+0.560, +0.849]  *  +5.94%  1.00
cross_doc             ALL         24  -0.004  [-0.014, +0.005]     -0.03%  0.38
cross_slice           ALL         30  +0.005  [-0.001, +0.014]     +0.04%  0.60
repeat_9_vs_1         ALL        106  +0.000  [-0.001, +0.001]     +0.00%  0.50
same_vs_other_source  ALL         30  +0.010  [+0.004, +0.019]  *  +0.09%  0.73
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.49 over 6 sources
- cross_slice: ρ = +0.66 over 6 sources
- within: ρ = +0.20 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/repeat_curve.png](figures/repeat_curve.png)
- [figures/stability.png](figures/stability.png)
- [figures/swap_matrix.png](figures/swap_matrix.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
