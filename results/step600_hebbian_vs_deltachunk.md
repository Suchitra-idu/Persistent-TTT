# Step 600: hebbian vs delta_chunk

Same 106 held-out documents, 6 sources. Δppl positive = mechanism lowers perplexity. `*` = 95% CI excludes zero.

| measure | hebbian (`ttt-v1.1`) | delta_chunk (`ttt-v1.1-clean-deltachunk`) |
|---|---|---|
| lora | +0.650 * | +0.772 * |
| within | +0.034 * | +0.058 * |
| between_slices | +0.002 | +0.012 * |
| total | +0.687 * | +0.842 * |
| cross_doc | -0.004 | -0.002 |
| cross_slice | +0.005 | +0.015 * |
| repeat_9_vs_1 | +0.000 | +0.000 |
| same_vs_other_source | +0.010 * | +0.002 |

Bundles: `ttt-v1.1/step_600/20261009-223740/`, `ttt-v1.1-clean-deltachunk/step_600/20261009-220029/`.
