# Checkpoint comparison

```
measure            source                  n_docs    Δppl            95% CI        rel   win
vs_cold_carry      RedPajamaArXiv              17  +0.136  [+0.097, +0.202]  *  +1.81%  1.00
vs_cold_carry      RedPajamaC4                 20  +0.769  [+0.538, +1.093]  *  +2.60%  1.00
vs_cold_carry      RedPajamaGithub             20  +0.086  [+0.052, +0.137]  *  +1.51%  0.95
vs_cold_carry      RedPajamaStackExchange       5  +0.307  [+0.092, +0.685]  *  +2.55%  1.00
vs_cold_carry      RedPajamaWikipedia          11  +0.508  [+0.384, +0.640]  *  +2.72%  1.00
vs_cold_carry      ALL                         73  +0.257  [+0.193, +0.337]  *  +2.13%  0.99
vs_cold_carry_off  RedPajamaArXiv              17  +0.094  [+0.074, +0.125]  *  +1.24%  1.00
vs_cold_carry_off  RedPajamaC4                 20  +0.283  [+0.194, +0.410]  *  +0.96%  1.00
vs_cold_carry_off  RedPajamaGithub             20  +0.046  [+0.032, +0.066]  *  +0.81%  1.00
vs_cold_carry_off  RedPajamaStackExchange       5  +0.185  [+0.066, +0.371]  *  +1.53%  1.00
vs_cold_carry_off  RedPajamaWikipedia          11  +0.267  [+0.205, +0.335]  *  +1.43%  1.00
vs_cold_carry_off  ALL                         73  +0.132  [+0.104, +0.166]  *  +1.09%  1.00
vs_lora_path       RedPajamaArXiv              17  +0.092  [+0.072, +0.122]  *  +1.21%  1.00
vs_lora_path       RedPajamaC4                 20  +0.266  [+0.182, +0.388]  *  +0.90%  1.00
vs_lora_path       RedPajamaGithub             20  +0.048  [+0.034, +0.067]  *  +0.85%  1.00
vs_lora_path       RedPajamaStackExchange       5  +0.140  [+0.056, +0.267]  *  +1.16%  1.00
vs_lora_path       RedPajamaWikipedia          11  +0.189  [+0.155, +0.226]  *  +1.01%  1.00
vs_lora_path       ALL                         73  +0.120  [+0.095, +0.149]  *  +0.99%  1.00
fresh_sanity       RedPajamaArXiv              17  +0.000  [+0.000, +0.000]     +0.00%  0.50
fresh_sanity       RedPajamaC4                 20  +0.000  [+0.000, +0.000]     +0.00%  0.50
fresh_sanity       RedPajamaGithub             20  +0.000  [+0.000, +0.000]     +0.00%  0.50
fresh_sanity       RedPajamaStackExchange       5  +0.000  [+0.000, +0.000]     +0.00%  0.50
fresh_sanity       RedPajamaWikipedia          11  +0.000  [+0.000, +0.000]     +0.00%  0.50
fresh_sanity       ALL                         73  +0.000  [+0.000, +0.000]     +0.00%  0.50
```

Unmatched documents: 0

## Config differences

- `carry_backprop`: `False` → `True`
- `carry_gate`: `False` → `True`
- `layer_indices`: `()` → `None`
- `output_gate_bias_init`: `-2.0` → `-6.0`
- `run_name`: `'lora-only-matched'` → `'hebbian-carry-600'`

## Figures

- [figures/checkpoints.png](figures/checkpoints.png)
