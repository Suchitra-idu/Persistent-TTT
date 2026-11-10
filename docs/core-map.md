# Ring 0 — `ttt/core/`

Pure logic and math. No I/O, no GPU, no global state, no environment. CPU torch
is allowed (REBUILD_PLAN D1), because the TTT kernels are the code that most
needs property tests and reference oracles, and those live here.

## The mechanism

| File | Does |
|---|---|
| [`ttt_math.py`](../ttt/core/ttt_math.py) | The TTT kernels. `scan` is the Hebbian chunked scan (chunk deltas, exclusive cumsum, clip at read time, apply); `chunked_delta_scan` is the chunked delta rule. `frobenius_clip`, `chunk_deltas`, `to_chunks` are their parts. |
| [`carry.py`](../ttt/core/carry.py) | `advance` (`S <- decay*S + delta`), `state_ratio` (`‖ηS‖_F / ‖W0‖_F`), `mean_state_ratio`. |

## Vocabulary

| File | Does |
|---|---|
| [`types.py`](../ttt/core/types.py) | The shared types: `WorkItem` (a token range of one document), `Session` (items whose carry is kept), `Carry` (per-layer state keyed by base-model layer index), the row types `PplRow`, `EvalRow`, `SliceRow`, and the four regime names (`fresh`, `lora_only`, `cold_carry_off`, `cold_carry`). |

## Scheduling and data selection

| File | Does |
|---|---|
| [`schedule.py`](../ttt/core/schedule.py) | Cutting a document into item ranges: `derive_slice_count`, `slice_doc` (training), `equal_token_slices` (evaluation). |
| [`lr_schedule.py`](../ttt/core/lr_schedule.py) | Warmup + cosine, matching HuggingFace's curve, and the step count that sizes it. |
| [`sampling.py`](../ttt/core/sampling.py) | Which documents to look at, given an injected rng: permutations, per-source picks with shortfall reporting. |
| [`balance.py`](../ttt/core/balance.py) | Rebalancing a source-skewed pool to a preset's ratios. |
| [`tokens.py`](../ttt/core/tokens.py) | The one chars-per-token estimate, used to prefilter before tokenizing. |

## Measurement and reporting

| File | Does |
|---|---|
| [`stats.py`](../ttt/core/stats.py) | `Interval`, `Estimate`, the percentile bootstrap, paired gaps, win rates, ranks and Spearman correlation. |
| [`metrics.py`](../ttt/core/metrics.py) | Perplexity aggregation (token-weighted and geometric mean) and the gap decomposition (`lora`, `within`, `between`, `total`) per source. |
| [`report.py`](../ttt/core/report.py) | Table builders: rows in, strings out. Nothing prints. |
| [`naming.py`](../ttt/core/naming.py) | Parameter names: which modules get LoRA, which optimizer group a parameter is in, which tensors a checkpoint holds. |

## Config — `ttt/core/config/`

| File | Does |
|---|---|
| [`ttt.py`](../ttt/core/config/ttt.py) | `TTTConfig`: chunk size, η, update rule, gates, clip, carry decay, `carry_gate`, `carry_backprop`. `derive_layer_indices`. |
| [`train.py`](../ttt/core/config/train.py) | `TrainConfig`: data limits, learning rates per group, schedule, LoRA, strategy, carry window, seed, run name, in-training evaluation. |
| [`dataset.py`](../ttt/core/config/dataset.py) | `DatasetSpec`: how to read one corpus and label every row with its source. |
| [`presets.py`](../ttt/core/config/presets.py) | Source-mixing presets. `slim-research` (25% C4, 20% Github and ArXiv, 15% Book, 10% Wikipedia and StackExchange) is the paper's. |
| [`resolve.py`](../ttt/core/config/resolve.py) | Merging overrides into a frozen config, and `describe` for the manifest. |

All configs are frozen dataclasses. They are resolved once, in `ttt/cli.py`, and
never changed afterwards (D3).
