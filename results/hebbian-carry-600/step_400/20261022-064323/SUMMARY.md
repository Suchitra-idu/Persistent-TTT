# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.904  [+1.541, +2.360]  *  +14.83%  1.00
within          ALL        106  +0.006  [+0.003, +0.009]  *   +0.06%  0.63
between_slices  ALL        106  +0.062  [+0.047, +0.081]  *   +0.57%  0.92
total           ALL        106  +1.972  [+1.600, +2.439]  *  +15.36%  1.00
cross_doc       ALL        100  -0.003  [-0.008, +0.002]      -0.03%  0.39
cross_slice     ALL        106  +0.062  [+0.048, +0.081]  *   +0.57%  0.92
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.60 over 6 sources
- cross_slice: ρ = +0.26 over 6 sources
- within: ρ = +0.26 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
