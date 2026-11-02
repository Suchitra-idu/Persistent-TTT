# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure         source  n_docs    Δppl            95% CI         rel   win
lora            ALL        106  +1.732  [+1.399, +2.152]  *  +13.49%  1.00
within          ALL        106  -0.045  [-0.067, -0.028]  *   -0.41%  0.27
between_slices  ALL        106  +0.040  [+0.027, +0.056]  *   +0.36%  0.81
total           ALL        106  +1.727  [+1.399, +2.145]  *  +13.45%  1.00
cross_doc       ALL        100  +0.038  [+0.025, +0.052]  *   +0.34%  0.79
cross_slice     ALL        106  +0.040  [+0.028, +0.056]  *   +0.36%  0.81
```

## Domain correlates (Spearman vs fresh ppl)

- cross_doc: ρ = +0.20 over 6 sources
- cross_slice: ρ = +0.77 over 6 sources
- within: ρ = -0.66 over 6 sources

## Figures

- [figures/regime_gaps.png](figures/regime_gaps.png)
- [figures/boundary_gaps.png](figures/boundary_gaps.png)
- [figures/domain_correlates.png](figures/domain_correlates.png)
