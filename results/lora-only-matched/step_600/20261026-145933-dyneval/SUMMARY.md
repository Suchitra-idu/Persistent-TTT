# Evidence bundle

## Pooled estimates (95% paired bootstrap over documents)

```
measure                                 source  n_docs    Δppl            95% CI        rel   win
dyneval_self_gain_self                  ALL        200  +0.271  [+0.217, +0.329]  *  +1.50%  0.76
dyneval_self_gain_related               ALL        200  -0.236  [-0.280, -0.192]  *  -1.31%  0.14
dyneval_self_gain_random                ALL        200  -0.284  [-0.320, -0.251]  *  -1.57%  0.07
dyneval_self_gain_other_source          ALL        200  -0.217  [-0.247, -0.190]  *  -1.20%  0.09
dyneval_self_self_vs_related            ALL        200  +0.507  [+0.435, +0.583]  *  +2.77%  0.90
dyneval_self_related_vs_random          ALL        200  +0.048  [+0.005, +0.096]  *  +0.26%  0.54
dyneval_self_random_vs_other_source     ALL        200  -0.067  [-0.098, -0.038]  *  -0.37%  0.39
dyneval_related_gain_self               ALL        200  +0.224  [+0.201, +0.249]  *  +1.24%  0.97
dyneval_related_gain_related            ALL        200  +0.029  [+0.015, +0.044]  *  +0.16%  0.56
dyneval_related_gain_random             ALL        200  -0.019  [-0.027, -0.011]  *  -0.10%  0.36
dyneval_related_gain_other_source       ALL        200  -0.022  [-0.029, -0.015]  *  -0.12%  0.34
dyneval_related_self_vs_related         ALL        200  +0.196  [+0.170, +0.224]  *  +1.09%  0.91
dyneval_related_related_vs_random       ALL        200  +0.047  [+0.031, +0.063]  *  +0.26%  0.68
dyneval_related_random_vs_other_source  ALL        200  +0.003  [-0.006, +0.013]     +0.02%  0.51
```

## Dynamic evaluation

Settings chosen on the tuning cases:

- tuned for self: adam lr=3e-05 steps=10
- tuned for related: adam lr=3e-05 steps=3

## Figures

