# Ring 5 map — the CLI and the experiments

Ring 5 is the composition root: the only place that decides which concrete
adapter goes into which port, and the only place that reads the environment.
Nothing imports it.

It is two layers, not one, and the import contracts say so. `ttt/cli.py` turns
arguments into frozen config and touches nothing else — the
`cli-only-resolves-config` contract forbids it from importing `ttt.app`,
`ttt.adapters` or `ttt.ports`. `ttt/experiments/` composes.

| File | Replaces | Entrypoint |
|---|---|---|
| `cli.py` | `_apply_cli_overrides` + `ttt_config`'s import-time env magic | — |
| `train_v1.py` | `train_modal.py::train` | `modal run --detach ttt/experiments/train_v1.py::train` |
| `holdout_eval_v1.py` | `infer_modal.py::holdout_eval` | `modal run ttt/experiments/holdout_eval_v1.py` |
| `single_doc_eval_v1.py` | `single_paper_eval` (D10 rename) | `modal run ttt/experiments/single_doc_eval_v1.py` |
| `holdout_generate_v1.py` | `infer_modal.py::holdout_generate` | `modal run ttt/experiments/holdout_generate_v1.py` |
| `compounding_pilot_v1.py` | `train_modal.py::compounding_pilot` | `modal run ttt/experiments/compounding_pilot_v1.py` |
| `sanity_check_v1.py` | `train_modal.py::sanity_check` | `modal run ttt/experiments/sanity_check_v1.py` |
| `chat_v1.py` | `infer_modal.py::TTTInference` (chat half) | `modal deploy ttt/experiments/chat_v1.py` |
| `chat_repl.py` | `chat_client.py` | `python -m ttt.experiments.chat_repl` |
| `plot_pilot.py` | `plot_pilot.py` | `python -m ttt.experiments.plot_pilot <json>` |
| `ruler_prepare_v1.py` | — | `modal run ttt/experiments/ruler_prepare_v1.py` |
| `ruler_eval_v1.py` | — | `modal run ttt/experiments/ruler_eval_v1.py` |
| `lang_corpus_prepare_v1.py` | — | `modal run ttt/experiments/lang_corpus_prepare_v1.py --role train` |
| `lang_scan_v1.py` | — | `modal run ttt/experiments/lang_scan_v1.py` |

## Passing configuration

Every knob travels in one `--flags` string, as `name=value` pairs:

```
modal run --detach ttt/experiments/train_v1.py --flags "num_epochs=2,strategy=hybrid"
```

Modal builds an entrypoint's CLI from its signature and cannot express
`**kwargs`, so a per-field surface is not available to it. `cli.parse_flags`
coerces each value using that field's declared default, and `cli.resolve`
rejects an unknown name locally — before anything reaches a GPU. An entrypoint's
own arguments (`--n-docs`, `--seeded`) stay separate, because those are not
config fields. `tests/architecture/test_entrypoints.py` enforces the shape.

`session_eval` has no equivalent: it read `.txt` files from a directory, an
arxiv-era manual workflow, and D11 cut it. The session-perplexity *measurement*
it drove survives as `ttt/app/session_eval.py`, which `single_doc_eval_v1` uses.

---

## `cli.py`

Two entry points, because a Modal function signature cannot express `None`:

- **`resolve(**kwargs)`** — the real surface. `None` means "not given". Unknown
  keys raise, naming the valid ones, because a typo costs GPU-hours.
- **`from_flags(**kwargs)`** — maps the sentinels a Modal entrypoint *can*
  express (`0`, `""`, `-1`) onto `None`, then calls `resolve`. The sentinel mess
  stops here.

The settable surface is **derived from the dataclasses**, not listed:

```python
_TRAIN_FIELDS = _settable(TrainConfig, handled={"strategy", "session_training",
                                                "source_preset", "micro_batch_size"})
```

so a new config field is reachable the moment it exists. Strategy knobs are the
union over registered strategies, so a new plugin's knobs auto-enrol — which is
D4's payoff: `--carry-min-tokens` configures the `Hybrid` instance, and passing
it alongside `--strategy everlasting` is an error rather than a silent no-op.

Three things resolve through a *rule* rather than a copy, which is why they are
named arguments instead of pass-through fields:

- **`strategy`** picks the plugin, and a source-scoped one defaults `eval_every`
  to 25 — it has ~10x fewer optimizer steps per epoch, so the usual 100 fires
  too rarely to see anything. An explicit `--eval-every` still wins.
- **`session_training`** is tri-state on the command line (`-1` unset) because
  it is the A1 ablation, not a mode.
- **`source_preset` / `only_sources`** produce *weights*, not a name. The old
  path registered a synthesised `_only_*` preset into a module-level dict as a
  side effect; the new one returns the mapping, which is what let D5 delete the
  preset registry.

`env_defaults(environ)` is the only environment read in the tree. `TTT_DATASET`,
`TTT_MODEL_SIZE` and `TTT_BASE_MODEL` are defaults an explicit argument beats —
the settings the laptop and the container must agree on, which is why
`modal_runtime.FORWARDED_KEYS` bakes exactly those into the image.

---

## The Engine

`_runtime.py` is the composition root and the one file in `experiments/` that
may change; the `*_v1.py` files are append-only. An `Engine` is a bag of ports
plus the two things only a real model can do — persist itself and be resumed —
so an experiment's `run(resolved, *, engine, source)` is pure wiring and the
whole thing runs on fakes with no GPU. That is what `tests/experiments/` does.

`build()` is the real one: transformers, PEFT, the torch adapters. It takes the
storage adapter *and* its mount path separately, because PEFT writes an adapter
directory and the Storage port speaks in blobs. That is a deliberate Ring-5-only
concession rather than a `root` attribute smuggled onto the port.

---

## RULER

Two phases, not one: `ruler_prepare_v1` (CPU only) synthesizes each task x
context-length example set once via `extensions/ruler_tasks/` and writes it
to `Storage` as JSONL; `ruler_eval_v1` (GPU) only reads those sets and scores
generations under each `FastWeights` regime — `COLD_CARRY` (TTT on) and
`FRESH` (TTT off). Splitting it this way means synthesis never runs inside a
billed GPU container, and every model/checkpoint compared runs against the
*same* fixed example set.

`RulerConfig.prompt_style` (`ruler_flags="prompt_style=..."`) picks how a
prepared example reaches the model: `base` (default) appends the task's own
completion cue (`RulerExample.answer_prefix`) straight onto the prompt — what
a non-instruct checkpoint needs, since it has no instruction-following to
lean on. `instruct` routes through the tokenizer's chat template instead,
for when the base model being evaluated is swapped for an instruct-tuned one.

`ruler_eval.py` runs the FastWeights STREAM family (`stream=True`), not
CARRY/`_scan_forward` — `_scan_forward` materializes one
`[num_chunks, d_model, d_ff]` tensor per patched layer for the whole prompt
at once, which OOMs on one H100 well before 32k tokens. `_stream_forward`
(the same path chat generation already uses) holds one running
`[d_model, d_ff]` state instead, committed and discarded chunk by chunk —
O(1) in sequence length. Known gap versus the scan path: a trailing partial
chunk is never committed, which is negligible since a prompt's last tokens
are its query/answer cue, never the needle.

---

## Language transfer

Targets `hybrid` (`core.config.train.DEFAULT_STRATEGY`): does training on a
language the model is weak at actually lower its perplexity? `everlasting`'s
per-source seeds are orthogonal to this question and are not the eval's main
path.

`core.config.lang_transfer.TRAIN_LANGUAGES` / `EVAL_LANGUAGES` are currently
the *same* ten languages — the worst base-model ppl readings a real
`lang_scan_v1` run found (see below). An earlier version of this eval trained
on one set and evaluated a disjoint set, to test transfer to an untrained
language; that showed a weak signal, while training directly on a weak
language showed a strong one, so the design moved to direct train+eval on
the worst offenders. `TRAIN_LANGUAGES` and `EVAL_LANGUAGES` stay two separate
names rather than collapsing into one, specifically so a future transfer
attempt is one assignment away — point `EVAL_LANGUAGES` at a tuple that pulls
in a few `CANDIDATE_LANGUAGES` instead of copying `TRAIN_LANGUAGES`.

`lang_corpus_prepare_v1` (CPU only) streams each language's
`wikimedia/wikipedia` config, stopping at `target_rows`, into two *separate*
corpora — `TRAIN_LANGS` and `EVAL_LANGS` (`extensions/datasets/
lang_transfer.py`) — not one shared spec. A single `DatasetSpec`'s
`holdout_boundary` is one cut point on one table; this eval needs languages
that are eval-only and must never reach training, which one cut point can't
express without fragile row-placement engineering.

Two steps per corpus, not one write: `prepare()` fetches each language into
its own file (`core.config.lang_transfer.corpus_dir`, resumable per
language), then `combine()` merges and shuffles all of a role's languages
into the single file a DatasetSpec actually reads (`combined_dir`). This
matters because `holdout_boundary`'s "the last N rows are a fair eval
sample" assumes rows are already mixed — true for a natively-mixed corpus
like SlimPajama, false for one built by concatenating one file per language,
where an unshuffled tail lands entirely inside whichever language sorts
last by filename. `TRAIN_LANGS` is read through both `load()` (training)
and `holdout()` (`train_v1`'s periodic in-loop eval, over its own holdout —
not `EVAL_LANGS` — sized as `TRAIN_HOLDOUT_FRACTION` of the raw train pool,
so it can't go stale if `DEFAULT_TARGET_ROWS` changes); `EVAL_LANGS` only
through `holdout()`, with `holdout_last_n` large enough that its whole
(small) pool is eligible.

Unlike `prepare()`, which is correctly keyed per language (`corpus_dir`) so a
changed language list only re-fetches what's new, `combine()` writes to one
fixed path per role and always rebuilds rather than checking it first —
the language list is not part of that path, so an exists()-guard would go on
serving whichever mix built the file first, which is exactly what happened
the first time `TRAIN_LANGUAGES` changed: a stale seven-language file with
none of the new ten in it, silently emptying the training pool. Rebuilding
is CPU-only and cheap, unlike the network fetch `prepare()` genuinely wants
to skip on a repeat run.

Tokenizer efficiency varies by up to 3x across this project's language sweep
— see `metrics.bits_per_byte` and the `bpb`/`Δbpb` columns
`report.per_source_table`/`slice_gap_table` add.
Raw perplexity is not comparable across these languages; bits-per-byte is.
`Doc.n_bytes` (the row's raw UTF-8 length, captured once in
`data_pipeline._encode`) is what makes that possible — it is 0 (untracked)
on the plain training path, which never needs it.

`lang_scan_v1` is the scouting pass that picks the next languages to add:
base model, no LoRA, no carry (`Compute.eval_loss(..., lora=False)`, carry
reset before every document) against `core.config.lang_transfer.
CANDIDATE_LANGUAGES` — languages Qwen3 itself claims to support, each with a
real `wikimedia/wikipedia` config, excluding the three whose article counts
are bot-inflated (`ceb`, `war`, `min`), `lmo`/`pag`/`bjn` (a real scan
returned under 2000 tokens total across 5 docs for each — mostly stubs,
too thin to trust), and anything already in `TRAIN_LANGUAGES`/
`EVAL_LANGUAGES`. Sorted worst-bpb-first
(`report.language_scan_table`) — bpb, not raw ppl, for the same
tokenizer-fairness reason as the trained eval above.

---

## Modal

`modal_runtime.py` holds the image, volumes, secrets and mounts. Two
dependencies the old image carried are gone (D11): `bitsandbytes`, and a
flash-attn wheel pinned to `torch2.8+cu12+cp311+cxx11abiTRUE` — the most fragile
line in the build. Attention is `sdpa` everywhere.

`modal_storage.py` is `LocalStorage` plus one thing: `commit`. Writes inside a
container are invisible to the next container until the volume is committed,
which is the entire reason the Storage port has that method. It passes the same
conformance suite as the other two adapters, against a volume double that counts
commits.

---

## What the tests cover

| Suite | Asserts |
|---|---|
| `tests/experiments/test_cli.py` | every resolution rule, and that a typo, a bad type and an invalid combination each raise |
| `tests/experiments/test_experiments.py` | every entrypoint runs end to end on fixture data; same seed, same losses |
| `tests/experiments/test_chat_repl.py` | command parsing, switch application, the A/B's restore-on-failure |
| `tests/experiments/test_runtime.py` | resume, the boot log, tracker selection, the pilot's JSON, the sanity check |
| `tests/parity/test_cli_parity.py` | 270 assertions against the *old* `_apply_cli_overrides` |

The sanity check is worth calling out: it used to be GPU-gated, so nothing
proved the TTT identity in CI. It now runs on `TinyCausalLM` on CPU and asserts
the stronger property — at `W_target = 0` the difference is **exactly** zero,
not merely under a tolerance. The `1e-3` tolerance stays for the real model,
where bf16 makes exact zero impossible.
