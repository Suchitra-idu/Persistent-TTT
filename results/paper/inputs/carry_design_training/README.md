# Carry-design training runs, in-training evaluation logs

In-training evaluations (30 held-out documents, 5 per source, no intervals) from
the three attempts at a working carry, all Hebbian, all otherwise identical to
`ttt-v1.1` (`state_aware_gate=false, limit_docs=4000`). Copied from the Modal
run logs. The full 106-document evaluations of the final design are the
`paper_eval` bundles under `results/hebbian-carry-b/`.

| file | run | flags beyond ttt-v1.1 | outcome |
|---|---|---|---|
| `1_carrygate_bptt.log` | `hebbian-carrygate-bptt` | `carry_gate`, `carry_backprop` (gate open at 0.12, no carry scale) | carry doubled perplexity at step 100, run ended at about step 130 without a checkpoint |
| `2_carryscale_bptt.log` | `hebbian-carryscale-bptt` | as 1, plus zero-initialised carry scale | Hebbian blow-up until about step 140, out of memory at about step 195 |
| `3_carry_b.log` | `hebbian-carry-b` | as 2, plus `output_gate_bias_init=-6`, `carry_window=2` | stable from step 10, checkpoints every 50 steps |
| `4_carry_600.log` | `hebbian-carry-600` | same as 3, trained to step 600 | in-training evaluations every 50 steps to step 550; full evaluations at 400 and 600 under `results/hebbian-carry-600/` |
