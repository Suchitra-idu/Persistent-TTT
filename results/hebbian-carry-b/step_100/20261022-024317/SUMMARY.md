# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.450  [+1.171, +1.807]  *  +11.30%  1.00
within          ALL        106  +0.047  [+0.029, +0.067]  *   +0.41%  0.83
between_slices  ALL        106  +0.135  [+0.108, +0.169]  *   +1.19%  0.99
total           ALL        106  +1.633  [+1.319, +2.035]  *  +12.72%  1.00
cross_doc       ALL         24  +0.011  [+0.003, +0.021]  *   +0.07%  0.75
cross_slice     ALL         30  +0.151  [+0.101, +0.226]  *   +1.02%  1.00
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.09 over 6 sources
- cross_slice: ρ = +0.43 over 6 sources
- within: ρ = +0.66 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
