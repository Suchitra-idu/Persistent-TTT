# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure             source  n_docs    Δppl            95% CI         rel   win
lora                ALL        106  +2.007  [+1.628, +2.483]  *  +15.63%  1.00
within              ALL        106  +0.027  [+0.017, +0.039]  *   +0.25%  0.72
between_slices      ALL        106  +0.119  [+0.091, +0.155]  *   +1.10%  1.00
total               ALL        106  +2.153  [+1.747, +2.657]  *  +16.77%  1.00
cross_doc           ALL        100  -0.040  [-0.054, -0.029]  *   -0.37%  0.11
cross_slice         ALL        106  +0.119  [+0.093, +0.156]  *   +1.10%  1.00
carry_gain          ALL        106  +0.135  [+0.102, +0.174]  *   +1.26%  1.00
context_gain        ALL        106  +2.753  [+2.336, +3.179]  *  +25.65%  1.00
context_over_carry  ALL        106  +2.617  [+2.246, +3.062]  *  +24.70%  1.00
```

## Carry against full context

The carried state recovers 4.9% [4.3%, 5.7%] of the gain from placing the previous slice in attention.

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = -0.43 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.09 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
