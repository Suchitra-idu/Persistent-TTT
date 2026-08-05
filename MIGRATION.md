# Migration

What moved where, what was deliberately dropped, and what proves the two trees
agree. The rebuild contract is [`PLAN.md`](PLAN.md); this file is the diff.

The old flat modules now live in [`legacy/`](legacy/), moved but **unmodified**
(D8). That move was not cutover: `tests/parity/` still imports them from there.
Retiring them is a separate, explicit call — see [Cutover](#cutover).

---

## Where each old file went

| Old | New | Note |
|---|---|---|
| `inplace_ttt.py` | `ttt/core/ttt_math.py`, `ttt/core/carry.py`, `ttt/extensions/mechanism/inplace_ttt.py`, `ttt/adapters/torch_fast_weights.py` | Kernels to Ring 0, the `nn.Module` to Ring 1, the lifecycle to a Ring 3 adapter behind the `FastWeights` port |
| `train_modal.py` | `ttt/app/data_pipeline.py`, `stages.py`, `train_loop.py`, `eval_loop.py`, `pilot.py` + `ttt/experiments/train_v1.py`, `compounding_pilot_v1.py`, `sanity_check_v1.py` | 1735 lines became eight Ring 4 modules and three entrypoints |
| `infer_modal.py` | `ttt/app/session_eval.py`, `generate.py`, `chat.py` + `ttt/experiments/holdout_eval_v1.py`, `single_doc_eval_v1.py`, `holdout_generate_v1.py`, `chat_v1.py` | The 250 lines of table printing became `ttt/core/report.py` builders that return strings |
| `ttt_config.py` | `ttt/core/config/` (5 files) + `ttt/cli.py` + `ttt/extensions/datasets/` | Import-time env magic became one resolution in Ring 5 (D3) |
| `ttt_wiring.py` | `ttt/core/naming.py`, `ttt/adapters/checkpoint_io.py`, `ttt/adapters/model_builder.py` | Name logic is pure; persistence goes through the `Storage` port |
| `train_utils.py` | `ttt/core/schedule.py` + `ttt/extensions/strategies/` | Primitives in Ring 0, regimes as plugins (D4) |
| `data_utils.py` | `ttt/adapters/hf_data_source.py`, `hf_table.py` | Behind the `DataSource` and `Table` ports |
| `model_setup.py` | `ttt/adapters/model_builder.py` | |
| `observability.py` | `ttt/adapters/wandb_tracker.py`, `console_tracker.py` | `gpu_stats` and `param_health` are gone (D12) |
| `chat_utils.py` | `ttt/core/sampling_text.py` | Pure, so it moved inward |
| `chat_client.py` | `ttt/experiments/chat_repl.py` | Plus the D14 switch flags and `/ab` |
| `plot_pilot.py` | `ttt/experiments/plot_pilot.py` | |

`pipelines/pipeline.py` is out of scope — an independent upstream data-prep job
with no import relationship to the training code.

---

## What was dropped, and why

### Cut outright (D11)

| Cut | Consequence |
|---|---|
| `TTTConfig.v_source` (fixed to `hidden_state`) | `EmbeddingTap`, the embedding forward hook, `model._ttt_tap`, and the `_v_source` / `_v_left_context` dispatch all die. `v_source_norm` stays — it normalises hidden states |
| `TTTConfig.gate_reg_weight`, `gate_stats()` | Dead code: referenced in one comment, never called. The gate itself stays — it is the mechanism, not an ablation |
| `multi` and `single` strategies | arxiv-era regimes assuming uniformly long documents. Takes `make_slice_sessions`, `expected_items_per_doc`, and eight `TrainConfig` fields with them |
| `session_eval` entrypoint | Read `.txt` files from a directory — a manual arxiv-era workflow. The *measurement* survives as `ttt/app/session_eval.py` |
| Random slicing at eval | `equal_n_slices` is the reproducible path; the random one depended on `multi` |
| `bitsandbytes` PagedAdamW8bit | `torch.optim.AdamW`. 8-bit paging solved memory pressure that does not exist at 0.6B / ~30M trainable on an H100 |
| Pinned flash-attn wheel | `sdpa` everywhere. Removes a URL pinned to `torch2.8+cu12+cp311+cxx11abiTRUE`. **Tradeoff: some throughput loss at 16k context** |
| Source-preset registry | Plain data in `core/config/presets.py` (D5) |
| The `arxiv` dataset | SlimPajama is the corpus (D9). This is what let 35 `if "source" in ds.column_names` branches disappear |
| Loss masking | Already deleted in `00f2a53`; only its test file survived. It stays deleted (D7) |
| 12 of 25 logging keys | All `health/*`, `gpu/*`, `perf/*`, `anomaly/*`, `telemetry.alert`, per-layer `session/state_ratio_L{i}`, and the per-source `eval/<src>/*` keys (D12) |

Per-source eval numbers are **not lost** — they are the RQ3 deliverable. They
stop going to wandb, where they were ~40 keys of noise per eval, and come back
as `EvalReport.summaries` for the console table and the analysis records.

### Renamed (D10)

"paper" → "doc" throughout: `eval_n_papers` → `eval_n_docs`,
`n_papers_per_source` → `eval_n_docs_per_source`, `single_paper_eval` →
`single_doc_eval_v1`, `_eval_paper` → `eval_loop`. SlimPajama has documents.

### Still carried, undecided

`v_bidirectional` (knowingly invalid under next-token prediction),
`_stratified_sample_indices`, `resume_from`, and LoRA itself are all ported
as-is and listed in PLAN §9. None is a cleanup decision.

---

## Defects fixed in the port

| Defect | Where it was | What proves the fix |
|---|---|---|
| The stale scan oracle (max err 3.5e-3 against a 1e-6 tolerance) | `tests/test_scan_math.py` | `tests/core/test_carry.py` pins both limits: `decay=1.0` reproduces the pure sum the old oracle assumed, `decay<1.0` the EMA the mechanism runs. **The mechanism was right; the oracle was stale** |
| `run_holdout_eval`'s snapshot/restore had no test | `train_modal.py` | `tests/app/test_eval_loop.py::TestStateRestoration`, including one that raises mid-measurement |
| **D14.1** — the trained seed could not reach chat at all | `_stream_forward` never read `carried_delta` | `FastWeights.install(carry, family=STREAM)`; `tests/app/test_chat.py::TestSeedSwitch` |
| **D14.2** — two incompatible snapshot key schemes | `snapshot_carried_delta` keyed by layer index, `export_fast_weights` by enumeration; mixing them loaded 7 of 14 modules with **another layer's** delta, silently | One `Carry`, keyed by base-model layer index. `tests/parity/test_checkpoint_parity.py::test_the_carry_key_scheme_is_the_base_model_layer_index` |
| **D14.3** — no decay at the turn boundary | Chat accumulated a pure sum while training used an EMA | `FastWeights.decay_stream(factor=)`; `tests/app/test_chat.py::TestCrossTurnSwitch` |
| Gradient checkpointing was dropped in the port | `train_modal.py:698` enabled it; the rebuild did not, and a 16k document OOM'd an 80GB card on the first micro step — the scan state is ~2GB per TTT layer | `prepare_for_training`; `tests/adapters/test_model_builder.py::TestPrepareForTraining` |
| `model.train()` was never called | `from_pretrained` returns an eval-mode model, so LoRA dropout was inactive and `self.training` was False inside the mechanism | `test_it_puts_the_model_in_train_mode` |
| Per-turn gate std was `nan` every turn | `gate.std()` on one element — chat streams one token per call | Population std. `tests/parity/test_mechanism_parity.py::TestGate::test_the_gate_std_is_where_the_two_sides_deliberately_differ` |

---

## Behaviour that deliberately differs

Every one of these has a named test that fails if it is reverted.

| Change | Why |
|---|---|
| A nonfinite loss no longer defers the accumulation boundary | The old `continue` rolled that window's gradients into the next one — a double-size step, and a step count drifting out of sync with the LR schedule |
| A window with no finite loss does not step at all | AdamW would otherwise apply weight decay to every parameter on the strength of no gradient |
| `train/lr_*` is the rate the optimizer applied | The old loop called `scheduler.step()` after `optimizer.step()` and logged `get_last_lr()` — the *next* step's rate |
| `session_training` defaults to `True` | D4: it is the A1 ablation, not a mode. Pinned by `test_the_session_training_default_changed_deliberately` |
| Gate weight is zero-init, not `normal_(std=1e-3)` | D1: construction must not touch the global RNG. The gate has one output unit, so there is no symmetry to break |
| Eval regime names | `eval/carry_ppl` is the `cold_carry` regime; `carry` now means the *seeded* one. The metric keys are unchanged so dashboards keep working — see `docs/app-map.md` |

---

## What proves the two trees agree

`tests/parity/`, marked `parity` and run by default — 436 tests.

| Suite | Asserts |
|---|---|
| `test_mechanism_parity.py` | Both `InPlaceTTTMLP`s in **fp64 from bit-identical weights**, agreeing to **<1e-12** on: scan, stream, token-by-token stream, session carry at decay 1.0/0.9/0.0, clip on/off/biting, gate on/off, the bidirectional ablation, and across batch sizes |
| `test_wiring_parity.py` | LoRA target regex byte-identical across 3 depths × 2 strides; parameter classification over 13 synthetic names covering all four groups; checkpoint membership; token-weighted and per-source metric arithmetic to 1e-12 |
| `test_schedule_parity.py` | Session schedules identical under the same seed, for both surviving strategies |
| `test_cli_parity.py` | 28 shared config fields × 9 argument combinations against the old `_apply_cli_overrides` |
| `test_checkpoint_parity.py` | An old `ttt_params.pt` and `per_source_carries.pt` load into the new tree, and a new one loads back into the old — key sets identical, tensors unchanged |

Three things the parity suite deliberately does *not* assert as exact:

1. **Stream against scan** sits at float32 epsilon, not fp64 — `_commit_chunk`
   accumulates in fp32 on both sides by design. Old-vs-new stays exact because
   both downcast identically, which is why the comparison is old-vs-new rather
   than against an fp64 oracle.
2. **Gate std** differs by exactly the Bessel factor (sample vs population),
   asserted to 1e-12 as a *relationship* rather than an equality.
3. **`session_training`'s default**, above.

Parity pins `TTT_DATASET=slimpajama-6b` in `tests/parity/conftest.py` before any
module imports the old `ttt_config`, which reads it at import and defaults to
arxiv. Assertions still name the spec explicitly rather than trusting that.

---

## Cutover

Not done, and not automatic. Before deleting anything in `legacy/`:

1. `make check` green (currently **1944 tests, ~10s**) and `make guard` passing
   both plants.
2. `make test-gpu` on a real H100 — `tests/gpu/test_gpu_smoke.py`, which is the
   only tier that touches the actual base model: the identity check, a 20-step
   training run, and a checkpoint round trip. **This has not been run.**
3. A real training run against `ttt/experiments/train_v1.py` whose numbers you
   are willing to compare against an old-tree run.
4. Then delete `legacy/` entirely. `tests/parity/`
   goes with it — it cannot run without the old tree, and it is the one part
   of this suite designed to be deleted rather than maintained.

`legacy/tests/test_loss_mask.py` is already dead: it imports eight
functions deleted in `00f2a53` and the old suite does not collect without
`--ignore`. It can go at step 4 regardless.
