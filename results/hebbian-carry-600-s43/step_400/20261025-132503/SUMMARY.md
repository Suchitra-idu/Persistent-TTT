# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.894  [+1.535, +2.348]  *  +14.76%  1.00
within          ALL        106  +0.008  [+0.002, +0.015]  *   +0.07%  0.61
between_slices  ALL        106  +0.062  [+0.046, +0.084]  *   +0.57%  0.96
total           ALL        106  +1.964  [+1.594, +2.436]  *  +15.30%  1.00
cross_doc       ALL        100  -0.007  [-0.013, -0.002]  *   -0.07%  0.37
cross_slice     ALL        106  +0.062  [+0.047, +0.083]  *   +0.57%  0.96
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.14 over 6 sources
- cross_slice: ρ = +0.43 over 6 sources
- within: ρ = -0.37 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
