# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                         source  n_docs    Δppl            95% CI        rel   win
dyneval_gain_self               ALL        200  +0.267  [+0.213, +0.325]  *  +1.48%  0.76
dyneval_gain_related            ALL        200  -0.232  [-0.277, -0.189]  *  -1.29%  0.15
dyneval_gain_random             ALL        200  -0.279  [-0.315, -0.248]  *  -1.55%  0.07
dyneval_gain_other_source       ALL        200  -0.210  [-0.239, -0.184]  *  -1.16%  0.06
dyneval_self_vs_related         ALL        200  +0.499  [+0.427, +0.575]  *  +2.73%  0.88
dyneval_related_vs_random       ALL        200  +0.047  [+0.002, +0.095]  *  +0.26%  0.56
dyneval_random_vs_other_source  ALL        200  -0.069  [-0.102, -0.041]  *  -0.38%  0.41
```

## Dynamic evaluation

Setting chosen on the tuning cases: adam lr=0.0001 steps=3.

## Figures

