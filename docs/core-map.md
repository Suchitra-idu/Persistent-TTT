# Ring 0 map

Pure math and logic. May import `torch`, `math`, `dataclasses`. No I/O, no GPU,
no environment, no clock, no global state. Every file below has a matching
`tests/core/test_*.py`.

## The mechanism

| File | Holds | Key names |
|---|---|---|
| [ttt_math.py](../ttt/core/ttt_math.py) | The chunked scan | `scan`, `chunk_deltas`, `exclusive_cumsum`, `frobenius_clip`, `apply_state`, `stream_chunk_delta`, `stream_apply` |
| [carry.py](../ttt/core/carry.py) | The carry recurrence | `advance`, `state_ratio`, `mean_state_ratio`, `steady_state_scale` |

## Vocabulary

| File | Holds | Key names |
|---|---|---|
| [types.py](../ttt/core/types.py) | Every type the ports and plugins speak in | `DocRef`, `WorkItem`, `Session`, `Carry`, `GradStats`, `PplRow`, `EvalRow`, `REGIMES` |

`Carry` is keyed by base-model layer index and there is no other
representation — that is what makes D14 defect 2 unrepresentable.

## Scheduling

| File | Holds | Key names |
|---|---|---|
| [schedule.py](../ttt/core/schedule.py) | Cutting a doc into ranges | `equal_token_slices`, `slice_doc`, `derive_slice_count`, `covers` |
| [lr_schedule.py](../ttt/core/lr_schedule.py) | Warmup + cosine LR | `multiplier`, `warmup_steps`, `total_optimizer_steps` |

`equal_token_slices` is the reproducible eval path (no rng); `slice_doc` is the
random training path. `covers` is the definition of a valid partition, reused
by strategy contract suites.

## Data selection

| File | Holds | Key names |
|---|---|---|
| [tokens.py](../ttt/core/tokens.py) | The one chars→tokens heuristic | `estimate_tokens`, `min_chars_for`, `has_enough_tokens` |
| [sampling.py](../ttt/core/sampling.py) | Which docs to look at | `n_per_source_indices`, `stratified_indices`, `uniform_indices`, `permutation` |
| [balance.py](../ttt/core/balance.py) | Rebalancing a skewed pool | `balanced_indices`, `source_counts`, `SourceQuota` |

`balanced_indices` never invents rows and preserves pool order, so domains stay
interleaved. Callers must shuffle first.

## Measurement

| File | Holds | Key names |
|---|---|---|
| [metrics.py](../ttt/core/metrics.py) | Aggregation and the gap decomposition | `token_weighted_ppl`, `geometric_mean_ppl`, `gap_decomposition`, `summarise_by_source`, `seed_gap`, `clip_ratio` |
| [report.py](../ttt/core/report.py) | Table builders. Nothing prints | `Table`, `render`, `per_source_table`, `composition_table` |

## Naming and text

| File | Holds | Key names |
|---|---|---|
| [naming.py](../ttt/core/naming.py) | Parameter-name string logic | `lora_target_regex`, `classify_param`, `ttt_down_suffixes`, `is_checkpoint_key`, `strip_peft_prefix` |
| [sampling_text.py](../ttt/core/sampling_text.py) | The pure half of generation | `nucleus_filter`, `next_token`, `stop_token_ids`, `split_thinking`, `strip_chat_specials` |

`lora_target_regex` must never match a TTT layer's `down_proj` — an adapter on
a fast weight fails silently, so it is property-tested.

## Config

All frozen. Env and CLI resolve once, in Ring 5, and pass inward (D3).

| File | Holds |
|---|---|
| [config/ttt.py](../ttt/core/config/ttt.py) | `TTTConfig` — the mechanism's knobs, plus `derive_layer_indices` |
| [config/train.py](../ttt/core/config/train.py) | `TrainConfig` — outer-loop knobs. No session-mode booleans (D4), no strategy knobs |
| [config/dataset.py](../ttt/core/config/dataset.py) | `DatasetSpec` — how to read one corpus. Enforces "every row has a source" (D9) |
| [config/presets.py](../ttt/core/config/presets.py) | `SOURCE_PRESETS` and the `--only-sources` builder. Data, not a registry (D5) |
| [config/resolve.py](../ttt/core/config/resolve.py) | `merge` / `only_set` / `describe`. Generic, so plugin configs use it too |

`merge` rejects an unknown key rather than ignoring it: the failure it replaces
is a six-hour run at the default learning rate because a flag was misspelled.

## Not here yet

| Ring | Package | Phase |
|---|---|---|
| 1 | `ttt/extensions/` — dataset specs, strategies, the `nn.Module` | 2 |
| 2/3 | `ttt/ports/`, `ttt/adapters/` — 9 ports, real and fake | 3 |
| 4 | `ttt/app/` — the loops | 4 |
| 5 | `ttt/experiments/`, `cli.py` | 5 |
