# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure             source  n_docs    Δppl            95% CI         rel   win
cross_doc           ALL        100  +0.010  [+0.004, +0.018]  *   +0.10%  0.64
cross_slice         ALL        106  +0.099  [+0.075, +0.129]  *   +0.89%  0.92
carry_gain          ALL        106  +0.112  [+0.084, +0.148]  *   +1.02%  0.92
context_gain        ALL        106  +2.903  [+2.489, +3.375]  *  +26.32%  1.00
context_over_carry  ALL        106  +2.791  [+2.387, +3.209]  *  +25.57%  1.00
```

## Carry against full context

The carried state recovers 3.9% [3.1%, 4.6%] of the gain from placing the previous slice in attention.

## Figures

- [figures/boundary_gaps.png](figures/boundary_gaps.png)
