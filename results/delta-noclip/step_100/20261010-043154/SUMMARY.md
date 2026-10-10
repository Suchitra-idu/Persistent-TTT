# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.508  [+1.213, +1.883]  *  +11.74%  1.00
within          ALL        106  +0.003  [-0.006, +0.012]      +0.03%  0.62
between_slices  ALL        106  +0.002  [-0.018, +0.015]      +0.01%  0.63
total           ALL        106  +1.512  [+1.212, +1.885]  *  +11.78%  1.00
cross_doc       ALL         24  -0.005  [-0.016, +0.007]      -0.03%  0.29
cross_slice     ALL         30  +0.004  [-0.025, +0.028]      +0.02%  0.57
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.20 over 6 sources
- cross_slice: ρ = +0.09 over 6 sources
- within: ρ = +0.37 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
