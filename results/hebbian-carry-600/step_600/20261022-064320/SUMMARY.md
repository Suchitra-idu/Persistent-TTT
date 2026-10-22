# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure             source  n_docs    Δppl            95% CI         rel   win
lora                ALL        106  +1.971  [+1.597, +2.439]  *  +15.36%  1.00
within              ALL        106  +0.025  [+0.017, +0.035]  *   +0.23%  0.76
between_slices      ALL        106  +0.097  [+0.073, +0.128]  *   +0.89%  0.97
total               ALL        106  +2.093  [+1.701, +2.586]  *  +16.30%  1.00
cross_doc           ALL        100  -0.020  [-0.030, -0.013]  *   -0.19%  0.29
cross_slice         ALL        106  +0.097  [+0.074, +0.129]  *   +0.89%  0.97
carry_gain          ALL        106  +0.110  [+0.082, +0.145]  *   +1.02%  0.97
context_gain        ALL        106  +2.770  [+2.354, +3.202]  *  +25.72%  1.00
context_over_carry  ALL        106  +2.660  [+2.279, +3.113]  *  +24.96%  1.00
```

## Carry against full context

The carried state recovers 4.0% [3.3%, 4.7%] of the gain from placing the previous slice in attention.

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.14 over 6 sources
- cross_slice: ρ = +0.37 over 6 sources
- within: ρ = +0.03 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
