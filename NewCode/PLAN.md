# NewCode — Rebuild Plan

Rebuilding the In-Place TTT research codebase under `RESEARCH_ARCHITECTURE.md`
(the Research Hexagon). This document is the contract for the rebuild: what
gets built, in what order, and how each phase is proven correct before the
next one starts.

Status: **Phases 0–1 complete (2026-07-27)** — scaffold + enforcement, and all
of Ring 0. `make check` green: 407 tests, 5.5s. Phases 2–6 pending.

---

## 0. What exists today (the source of truth we are porting)

| File | Lines | What it really is |
|---|---|---|
| `inplace_ttt.py` | 508 | The TTT mechanism: `InPlaceTTTMLP` (scan + stream paths), carry lifecycle, snapshot/install, state norms, gate stats |
| `train_modal.py` | 1735 | Modal app + a 150-line data-loading swamp + a 350-line train loop + eval + compounding pilot + CLI |
| `infer_modal.py` | 1017 | Modal class `TTTInference` (ppl, session ppl, generate, chat) + 5 local entrypoints + ~250 lines of table printing |
| `ttt_config.py` | 370 | `DatasetSpec` + `DATASETS` + `SOURCE_PRESETS` + `TTTConfig` + `TrainConfig`, all env-var-driven at import time |
| `ttt_wiring.py` | 160 | LoRA regex, param grouping/classification, checkpoint save/load, per-source carry save/load |
| `train_utils.py` | 198 | Pure schedule/slicing math |
| `data_utils.py` | 100 | HF dataset open / source annotate / source filter / holdout split |
| `model_setup.py` | 56 | HF + PEFT + TTT patch assembly |
| `observability.py` | 126 | wandb wrapper + gpu stats + param health |
| `chat_utils.py` | 81 | top-p sampling, stop tokens, thinking split |
| `chat_client.py` | 151 | local REPL |
| `plot_pilot.py` | 64 | matplotlib pilot plot |
| `tests/` | 12 files | 159 passing, 1 failing, 1 dead (see §7) |

Out of scope: `pipelines/pipeline.py` (81 KB arXiv→HuggingFace data-prep, an
independent upstream job), the **`arxiv` dataset itself** (see D9 — SlimPajama
is the corpus going forward), and `docs/` (~300 KB of prose that describes the
old file layout and will need its own pass after the rebuild lands).

---

## 1. Architectural decisions (made up front, so they aren't re-litigated per file)

**D1 — Torch is allowed in Ring 0; frameworks are not.**
The spec says core is "pure, no framework." Read literally that would exile
the TTT scan math — the single most correctness-critical code in the repo —
out of the ring where property tests and reference oracles live. That is
backwards. The rule that actually carries the spec's intent is *no I/O, no
GPU, no global state, deterministic, laptop-runnable in milliseconds*, and
CPU torch satisfies all four. So:

- Ring 0 **may** import `torch`, `math`, `dataclasses`.
- Ring 0 **may not** import `transformers`, `peft`, `modal`, `datasets`,
  `wandb`, `numpy.random` globals, or touch `os.environ`, `time.time()`,
  `torch.cuda`, `torch.manual_seed`, or the filesystem.
- Both halves are machine-enforced (§4).

**D2 — Package root is `ttt/`, not bare `core/`.**
`import core` is a landmine in a repo that also runs on Modal. Everything
lives under one distribution package: `ttt.core`, `ttt.extensions`, …. This
is a one-level deviation from the spec's directory template; the rings are
unchanged. (Overridable — say the word and I'll use the bare template.)

**D3 — Config stops being import-time env-var magic.**
Today `ttt_config.py` reads `TTT_DATASET` / `TTT_MODEL_SIZE` at import and
builds module-level singletons `DATASET_SPEC`, `TTT_CFG`, `TRAIN_CFG` that
every module mutates via `dataclasses.replace`. In the rebuild, env/CLI
resolution happens exactly once in Ring 5 and produces a frozen config object
that is passed down. `TrainConfig` and `TTTConfig` become frozen. This is
spec §7 and it is the single biggest testability win in the port.

**D4 — Four session modes become four strategy plugins.**
`multi` / `single` / `hybrid` / `everlasting` are currently expressed as five
scattered switches: `_MODE_TO_FLAGS`, `_make_epoch_sessions`,
`_items_per_epoch`, `_print_session_composition`, `_mode_line` — plus boolean
flags on `TrainConfig` with a documented precedence order. All five collapse
into one `Strategy` protocol with `build` / `count` / `describe` / `compose`
and a `carry_scope` attribute (`"session"` vs `"source"`) that the train loop
dispatches on. Adding a fifth regime becomes one new file. This is the
clearest instance of "add, don't edit" in the whole codebase.

**D5 — Registries only where there are ≥2 real members.**
Built: `datasets` (2 — `slimpajama-6b` and a tiny synthetic `fixture` spec the
smoke/reproducibility tests run against) and `strategies` (2 — `hybrid` and
`everlasting`, per D11).
**Not** built (spec §8, one implementation is a function): the TTT mechanism
itself, the optimizer, the model forward, eval metrics, **and source presets**.
A preset is `dict[str, int]` — pure data with no behaviour, so a protocol plus
a registry plus a contract suite for it is ceremony. Presets live as a frozen
mapping in `core/config/presets.py` alongside the `--only-sources` builder.
Eval metrics stay pure functions in `core/metrics.py` with reference-oracle
tests, exactly as the spec's cookbook prescribes.

**D6 — Two compute-side ports, not one.**
`Compute` (loss / backward / optimizer step / grad stats) and `FastWeights`
(reset / advance / snapshot / install / state ratio / evolve+session toggles)
are separate protocols. That split is what lets the *entire* train loop —
including the carry lifecycle and the everlasting per-source carrier logic —
run against fakes with no torch model at all.

**D7 — Loss masking is not ported.** It was deliberately deleted in `00f2a53`
(from `train_modal.py`, `train_utils.py`, and `ttt_config.py`). Only the test
file survived the deletion. It stays deleted.

**D8 — NewCode is built side-by-side.** The old files are not touched until
the parity suite in Phase 6 is green. Cutover is a separate, explicit call.

**D9 — SlimPajama only; `source` becomes a universal invariant.**
The `arxiv` spec is dropped. That is not just one fewer file — arxiv was the
*only* single-source, `tokens_est`-carrying dataset, and its existence is why
35 call sites across `train_modal.py` / `infer_modal.py` / `ttt_config.py`
branch on `if "source" in ds.column_names` or `if tokens_est_column`. Two
invariants replace all of them:

- **Every row always carries a `source` label.** A multi-source spec extracts
  it from `meta`; a single-source spec (if one is ever added) labels every row
  with a constant. So `source` is never absent, the per-source eval / balance /
  composition paths lose their `has_source` fallbacks entirely, and
  `everlasting_carry`'s "requires a source column" runtime check becomes a
  registry contract-suite assertion instead.
- **Token counts are always estimated, never read from a column.**
  `tokens_est_column` is deleted from `DatasetSpec`. One pure
  `core/tokens.py::estimate_tokens` (the ~3.5–4 chars/token heuristic) serves
  the prefilter stage, the eval-holdout filter, and the inference holdout
  filter — three places that each carry their own two-branch version today. If
  a future spec ships a real count column, *that* is the second instance that
  earns the seam back (spec §8).

Defaults follow: `slimpajama-6b` is the default spec, `holdout_last_n=5000`,
`default_source_preset="slim-research"`.

**D10 — "paper" → "doc" throughout.** `eval_n_papers`, `n_papers_per_source`,
`paper_idx`, `session_papers_min/max`, `_eval_paper`, `_print_paper_preview`
are arxiv-era vocabulary for what SlimPajama calls documents. Renamed in the
rebuild. Cosmetic, but it is a rename that only becomes correct once arxiv is
gone, and doing it later means touching every ring again.

**D11 — Confirmed cut list.**
The carry *is* the research — session-persistent TTT — so nothing on the carry
path is touched. What comes out is everything that isn't on it:

| Cut | Consequence |
|---|---|
| `TTTConfig.v_source` — fixed to `"hidden_state"` | `EmbeddingTap` dies entirely (it exists only to feed the `"embedding"` variant), and with it the embedding forward hook, `model._ttt_tap`, and the `_v_source` / `_v_left_context` dispatch. `v_source_norm` stays — it normalises hidden states. |
| `TTTConfig.gate_reg_weight` and `gate_stats()` | `gate_stats()` is dead code (referenced in one comment, never called). `gate_reg_weight` defaults to 0.0 and is never exercised. Removes `_gate_l2`, `gate_reg_term`, and the loss-side branch. The gate itself stays — it is part of the mechanism, not an ablation. |
| `multi` + `single` strategies | arxiv-era regimes assuming uniformly long docs. Removes `make_slice_sessions`, `expected_items_per_doc`, and the `session_papers_min/max`, `slice_prob`, `slice_min/max`, `slice_min_tokens`, `single_paper_sessions`, `single_paper_slices_min/max` config fields. Registry keeps `hybrid` (SlimPajama training) + `everlasting` (seed production) = 2 members. |
| `session_eval` entrypoint | Reads `.txt` files from a directory — a manual arxiv-era workflow. |
| Random slicing at eval (`slice_papers` / `slice_seed`) | `equal_n_slices` is the reproducible path; the random one depended on the `multi` strategy being cut anyway. |
| `bitsandbytes` PagedAdamW8bit → `torch.optim.AdamW` | One dependency out. 8-bit paging solves memory pressure that doesn't exist at 0.6B / ~30M trainable on an H100. |
| pinned `flash-attn` wheel → `sdpa` everywhere | Removes a URL pinned to `torch2.8+cu12+cp311+cxx11abiTRUE`, the most fragile line in the image. Inference already used `sdpa`. **Tradeoff: some throughput loss at 16k context** — revisit if training wall-time becomes the bottleneck. |
| Source-preset registry | → plain data (D5). |

`single_paper_eval` is **renamed, not removed** — it becomes
`single_doc_eval_v1` (D10: SlimPajama has documents, not papers).

**D12 — Logging budget: 25 keys → 13.**
Telemetry today is dashboard-building for a 500-step run. The keep-list is
exact; anything not on it goes.

*Kept:* `micro/doc_loss`, `micro/state_ratio_mean`, `train/step`,
`train/loss`, `train/lr_lora`, `train/lr_wdown`, `train/lr_new`,
`train/grad_clip_ratio`, `grad/lora`, `grad/wdown`, `grad/new`, and the
**aggregate** in-loop eval numbers only (`eval/carry_ppl`,
`eval/carry_off_ppl`, `eval/fresh_ppl`, `eval/gap_within`,
`eval/gap_between`, `eval/gap_total`, `eval/state_ratio_final`, plus the
`eval_seed/*` aggregates).

*Cut:* all `health/*` (deletes `param_health` entirely), all `gpu/*`
(`gpu_stats`), all `perf/*`, all `anomaly/*` and `telemetry.alert`,
`session/state_ratio_L{i}` per-layer, `session/state_ratio_max`,
`session/sessions_done`, `micro/paper_tokens`, `micro/session_pos`,
`micro/session_n`, and the per-source `eval/<src>/*` keys.

Per-source eval numbers are *not lost* — they are the RQ3 deliverable. They
stop going to wandb (where they were ~40 keys of noise per eval) and go to
the console table and the analysis records instead. The two-x-axis
`define_metric` setup stays, since both `micro/*` and `train/*` survive.

**D13 — `docs/` is kept.** Not ported, not deleted. New documentation gets
written against the rebuilt tree later.

**D14 — Chat is kept and promoted to a first-class carry-ablation surface.**
Not a leftover demo: the point is to talk to the model with each carry
mechanism switched on and off independently, watch how the responses change,
and sanity-check that it isn't generating nonsense. That is a real
deliverable (it is the only qualitative window onto SO1's three components),
so it gets built properly rather than ported as-is.

*Three defects in the current chat path, all of which have to be fixed for
the ablation to even be possible:*

1. **The seed cannot reach chat at all today.** Chat runs the stream path
   (`stateful=True`), which reads `state.delta`. The trained per-source
   carrier is installed by `install_carried_delta`, which writes
   `carried_delta`. `_stream_forward` never reads `carried_delta`. So of the
   three components, "meta-learned per-source initialisation" is simply not
   reachable from chat — there is no code path connecting them.
2. **Two incompatible snapshot key schemes, and mixing them corrupts state
   silently.** `snapshot_carried_delta` keys by base-model layer index —
   `{1, 3, 5, …, 27}` at Qwen3-0.6B with stride 2. `export_fast_weights` keys
   by enumeration order — `{0, 1, …, 13}`. `import_fast_weights` does
   `if i in snapshot` over the enumeration. Feed it a `per_source_carries.pt`
   payload and it matches `i ∈ {1,3,5,7,9,11,13}`, loading 7 of 14 modules,
   each with **another layer's** delta. No error, no warning.
3. **No decay at the turn boundary.** Session training accumulates the carry
   as an EMA (`carried_decay=0.9`); the stream path accumulates a pure sum
   bounded only by the Frobenius clip. So chat drives `state_ratio` into a
   regime training never showed the model. This is a concrete, testable
   hypothesis for *why* long chats might degrade into nonsense, and it stays
   invisible until the two paths share a decay setting.

*The switch matrix.* Three independent toggles matching SO1's three
components, plus a control:

| Switch | Off | On |
|---|---|---|
| `within_turn` | fast weight frozen during the turn (`evolve=False`) | chunk-scan updates as tokens stream |
| `cross_turn` | state cleared at each turn boundary | accumulated delta survives into the next turn (with `carried_decay` applied — defect 3) |
| `seed` | start from zero | install the trained per-source carrier for a chosen source |
| `context` *(control)* | `none` — TTT is the only memory, the current research design | `full` — ordinary conversation history, to establish the model isn't broken before blaming the carry |

`seed` also accepts a mismatched source, which is the chat-side version of
the existing `force_source` swap test.

*Diagnostics per turn*, so "is the carry doing anything" is observable
rather than guessed: `state_ratio`, `pending_tokens / chunk_size`,
per-turn growth `‖ΔS_turn‖_F / ‖S‖_F`, and **gate mean/std** — already
captured under `no_grad` in `_gated` at inference and currently thrown away.
A gate pinned near zero means the TTT term is suppressed and the carry
cannot be affecting the output, which is the first thing to check when
responses look identical.

*Same-prompt A/B.* One command generates the same turn twice under two
switch settings with an identical sampling seed, and prints both. This is
the only way to attribute a difference to the carry rather than to sampling
noise, and it is the feature the current chat most obviously lacks.

*Cleanup that comes with it.* One generation loop, not two — today
`generate()` calls `model.generate()` while `chat_turn` hand-rolls a
token loop with `sample_top_p`; the hand-rolled one wins (explicit,
TTT-state-aware, no HF surprises) and `model.generate()` goes. The 97-line
`chat_turn` splits into `app/chat.py` (turn orchestration + switches),
`core/sampling_text.py` (pure, already planned), and
`adapters/torch_generation.py` (the token loop). Both state families move
behind the single `FastWeights` port with **one** `Carry` representation
keyed by base-model layer index, which is what kills defect 2 permanently
and makes defect 1 a one-line install.

*Expectation-setting:* the base is Qwen3 continual-pretrained on raw
SlimPajama text with LoRA — never instruction-tuned on chat traces, as
`chat_client.py` already warns. Some incoherence is the base model, not the
carry. That is exactly why the `context=full` control and the same-seed A/B
exist.

---

## 2. Target tree

```
NewCode/
  pyproject.toml               deps, pytest config, markers
  Makefile                     `make check` = lint-imports + arch tests + pytest
  .importlinter                the enforced Dependency Rule
  ARCHITECTURE.md              map + ring diagram, points at RESEARCH_ARCHITECTURE.md
  cli.py                       Ring 5 — single source of CLI args

  ttt/
    core/                      RING 0 — pure. property-tested + oracles.
      types.py                 WorkItem, Session, DocRef, Carry, GradStats, EvalRow, PplRow
      config/
        ttt.py                 TTTConfig (frozen) — no v_source, no gate_reg_weight (D11)
        train.py               TrainConfig (frozen) — session-mode booleans REMOVED (D4),
                               slice_*/session_papers_* REMOVED (D11)
        dataset.py             DatasetSpec (frozen)
        presets.py             frozen source-weight mapping + only_sources builder (D5)
        resolve.py             pure merge: defaults + overrides -> frozen config
      ttt_math.py              chunk deltas, exclusive cumsum, frobenius clip, apply
      carry.py                 EMA carry accumulation (carried_decay), state-ratio math
      schedule.py              slice_doc, equal_token_slices, derive_slice_count, partitions
      tokens.py                estimate_tokens — the one char/token heuristic (D9)
      sampling.py              n_per_source, stratified, uniform index selection
      balance.py               preset weights + source labels + target -> kept indices
      metrics.py               token-weighted ppl, 3-mode & 5-mode gap decomposition,
                               per-source aggregation, seeded-pass deltas
      naming.py                LoRA target regex, param classification, peft prefix strip
      sampling_text.py         top-p/top-k sampling, stop-token set, thinking split
      report.py                pure table BUILDERS (rows in, list[list[str]] out)

    extensions/                RING 1 — plugins. one contract suite each.
      datasets/
        _registry.py           DatasetSpec protocol + DATASETS + @register
        slimpajama_6b.py       the corpus (default spec)
        fixture.py             tiny synthetic multi-source spec for smoke/repro tests
      strategies/
        _registry.py           Strategy protocol + STRATEGIES + @register
        hybrid.py              length-gated: short=whole-doc, long=k slices
        everlasting.py         whole-doc, carry_scope="source"
      mechanism/
        inplace_ttt.py         InPlaceTTTMLP nn.Module over core.ttt_math kernels
                               (no EmbeddingTap — died with v_source, D11)

    ports/                     RING 2 — interfaces, defined in core types
      compute.py               Compute
      fast_weights.py          FastWeights
      tracker.py               Tracker
      storage.py               Storage
      table.py                 Table (len/column/select/row/column_names)
      data_source.py           DataSource
      tokenizer.py             Tokenizer
      clock.py                 Clock
      rng.py                   Rng

    adapters/                  RING 3 — real AND fake, side by side
      torch_compute.py         / fake_compute.py         (scripted losses; AdamW, D11)
      torch_fast_weights.py    / fake_fast_weights.py    (dict-of-floats carry)
                               one Carry repr, layer-index keyed, both state
                               families behind one port (D14 defect 2)
      torch_generation.py      / fake_generation.py      (scripted tokens)
      wandb_tracker.py         / in_memory_tracker.py    + console_tracker.py
                               (13-key budget, no alerts, no param_health — D12)
      modal_storage.py         / local_storage.py        / in_memory_storage.py
      hf_table.py              / list_table.py
      hf_data_source.py        / fake_data_source.py
      hf_tokenizer.py          / fake_tokenizer.py       (deterministic, no download)
      system_clock.py          / fake_clock.py
      numpy_rng.py             / scripted_rng.py
      model_builder.py         HF + TTT patch + PEFT assembly (the framework seam)
      checkpoint_io.py         ttt_params.pt + per_source_carries.pt on Storage
      modal_runtime.py         modal.App, Image, Volumes, Secrets
                               (no flash-attn wheel, no bitsandbytes — D11)

    app/                       RING 4 — orchestration only, all-fake tested
      data_pipeline.py         PIPELINE = [split_holdout, filter_sources, shuffle,
                               prefilter, balance, tokenize, drop_short, rebalance, cap]
      stages.py                each stage: Data -> Data, independently testable
      train_loop.py            accum / clip / step / log / save / eval / carry lifecycle
      eval_loop.py             3-mode zero-seed pass + seeded pass + per-source
      session_eval.py          session-perplexity use case (drives infer tables)
      pilot.py                 cold / persist / seeded compounding pilot
      chat.py                  turn orchestration + the 4 carry switches (D14)
      generate.py              the one token loop, shared by chat and completions

    experiments/               RING 5 — append-only compositions
      train_v1.py              modal fn: the current training entrypoint
      holdout_eval_v1.py
      single_doc_eval_v1.py    renamed from single_paper_eval (D10/D11)
      holdout_generate_v1.py
      compounding_pilot_v1.py
      sanity_check_v1.py
      chat_repl.py             local REPL (was chat_client.py) — switch flags,
                               /ab same-seed A/B, per-turn carry diagnostics
      plot_pilot.py
                               (session_eval_v1 removed — D11)

  tests/
    RULES.md                   the 10 rules, verbatim from spec §10
    architecture/              ring-boundary + banned-idiom tests (grep-based)
    core/                      property tests + reference oracles
    extensions/                contract suites (parametrized over registries)
    ports/                     conformance suites (abstract class per port)
    app/                       orchestration tests on all-fake adapters
    experiments/               smoke + same-seed reproducibility
    parity/                    OLD-vs-NEW numeric equivalence (Phase 6, then frozen)
    gpu/                       marked `gpu`, deselected by default
```

---

## 3. The data pipeline (spec §5) — the biggest single cleanup

`train_modal.py::load_token_dataset` is a 145-line function with two
rebalance passes, an over-fetch multiplier, an `only_mode` branch, and a
char-count fallback all interleaved. It becomes an explicit stage list:

```python
PIPELINE = [
    split_holdout,        # newest holdout_last_n rows reserved for eval
    filter_sources,       # spec.include_sources
    shuffle,              # seeded, .select over a permutation
    prefilter,            # core.tokens.estimate_tokens >= min_doc_tokens (D9)
    balance,              # preset ratios, over-fetched by PRESET_OVERSAMPLE
    tokenize,             # the one impure stage — goes through the Tokenizer port
    drop_short,           # exact token count
    rebalance,            # exact preset ratios after the source-biased drop
    cap,                  # limit_docs, applied last for only-sources mode
]
```

`prefilter` is single-branch now that `tokens_est_column` is gone, and
`balance` / `rebalance` no longer need their "does this dataset have sources"
guard clauses — both are consequences of D9.

Each stage is `Data -> Data` where `Data` carries a `Table` (port) plus the
resolved config plus an accumulating log of what each stage kept/dropped.
Stages are pure and individually tested against `ListTable`; the whole
pipeline runs in a test in milliseconds with no `datasets` install.

---

## 4. Enforcement (spec §9)

`.importlinter`:
```ini
[importlinter:contract:rings]
type = layers
layers =
    ttt.experiments
    ttt.app
    ttt.adapters
    ttt.ports
    ttt.extensions
    ttt.core

[importlinter:contract:app-never-touches-a-concrete-adapter]
type = forbidden
source_modules = ttt.app
forbidden_modules = ttt.adapters
```
(The spec's own snippet is not sufficient here — a pure `layers` contract puts
`adapters` *below* `app`, which would legalise `app → adapters`. The explicit
`forbidden` contract is what actually enforces "via interfaces, never a
concrete adapter directly." Verified in Phase 0: a planted `app → adapters`
import leaves the `rings` contract **kept** and breaks only the `forbidden`
one. Note also that layers are *newline*-separated — the spec writes them on
one line joined by `|`, which in import-linter means "independent siblings in
a single layer" and would forbid `app → core`.)

Plus forbidden contracts keeping `transformers` / `peft` / `modal` / `wandb` /
`datasets` out of `core`, `extensions`, and `ports`.

Plus `tests/architecture/` — mechanical grep tests that import-linter can't
express: no `os.environ`, `time.time()`, `torch.cuda`, `torch.manual_seed`,
`random.seed`, or `open(` in rings 0–2; every port has a conformance suite;
every registry has a contract suite.

`make check` = `lint-imports && pytest -m "not gpu and not slow"`. A boundary
violation is a failing test, not a review comment.

---

## 5. Phases

Each phase is independently verifiable and leaves the tree green. I'll stop
after each one and report.

### Phase 0 — Scaffold + enforcement  *(small)*
Package skeleton, `pyproject.toml` (adds `hypothesis` and `import-linter` as
dev deps — neither is installed today), pytest markers and default
deselection, `.importlinter`, `tests/RULES.md`, `ARCHITECTURE.md`, `Makefile`.
**Done when:** `make check` is green on an empty tree, and a deliberately
planted `core → transformers` import makes it fail.

### Phase 1 — Ring 0 core  *(large)*
All of `ttt/core/`. Ports the math out of `inplace_ttt.py`, `train_utils.py`,
`ttt_wiring.py`, the metric arithmetic buried in `train_modal.py` and
`infer_modal.py`'s printing functions, and `chat_utils.py`.
Tests: property tests (partitions cover the range, are disjoint, meet the
minimum; balance never exceeds pool size; gaps decompose additively) and
reference oracles (naive sequential apply-then-update vs the chunked scan;
naive per-token ppl vs token-weighted aggregation).
**Done when:** ~130 core tests green in under 2s, including a
`carried_decay`-aware oracle (see §7, defect 1).

### Phase 2 — Ring 1 extensions  *(small–medium)*
Two registries + four plugins, and the TTT `nn.Module` rebuilt on the
Phase-1 kernels.
Tests: one contract suite per registry, parametrized over
`REGISTRY.values()`, so a new plugin auto-enrolls. Strategy contract:
every session is non-empty, every doc appears exactly once, items are
contiguous and cover their doc, `count()` matches `len(build())` for exact
modes, `describe()` is a non-empty single line, builds are deterministic per
seed. Dataset contract: every spec yields a `source` label on every row
(D9) — this is where `everlasting_carry`'s old runtime check now lives.
**Done when:** contract suites green, and old-vs-new schedule parity holds
under identical seeds for both surviving modes.

### Phase 3 — Rings 2/3 ports + adapters  *(large)*
Nine ports, and for each at least one real and one fake adapter.
Tests: one abstract conformance suite per port; every adapter (real and fake)
subclasses it. Real adapters that need network/GPU are marked `integration`
or `gpu` and deselected by default; the fakes always run.
**Done when:** every adapter passes its port's conformance suite, and the
architecture test asserting "every port has a conformance suite" passes.

### Phase 4 — Ring 4 app  *(large — the heart)*
Data pipeline + stages; train loop; eval loop; session eval; pilot; chat;
generate.
Tests, all on fakes with zero GPU: accumulation boundary fires at exactly
`grad_accum_steps`; a nonfinite loss skips backward but still advances
session state and increments the counter; checkpoints save on cadence and at
the final step; eval fires on cadence and restores module state exactly
(the current code's snapshot/restore in `run_holdout_eval` is subtle and
deserves a real test); everlasting-carry installs the right source's carrier
before the forward and snapshots it back after; carry resets at session
boundaries. The whole app suite runs in well under a second.
Chat (D14) is tested here too, against `fake_generation` + `fake_fast_weights`:
each of the four switches independently changes the recorded state
transitions; `cross_turn=False` leaves zero residual state at the turn
boundary while `True` leaves the decayed carry; a seeded session starts from
the installed carrier and an unseeded one from zero; and the same-seed A/B
produces identical token streams when both arms have identical switches.
**Done when:** a complete simulated training run — multiple epochs, both
strategies, eval and save cadences firing — executes in CI in milliseconds,
and the chat switch matrix is covered with no model loaded.

### Phase 5 — Ring 5 experiments, CLI, Modal  *(medium)*
`cli.py` as the single arg surface; the Modal runtime adapter; one
append-only experiment file per current entrypoint; the chat REPL; the plot
script.
Tests: CLI-args → resolved-frozen-config snapshot tests (this replaces
`_apply_cli_overrides`, currently the most-tested function in the repo);
experiment smoke on tiny fake data; same-seed reproducibility.
**Done when:** every current entrypoint has an equivalent, and each one's
resolved config is asserted equal to what the old CLI path produced.

### Phase 6 — Parity, checkpoint compat, cutover  *(medium — the trust phase)*
This is what makes the rewrite safe to believe.
1. **Numeric parity suite** (`tests/parity/`): import the *old* modules and
   the *new* ones side by side, drive both with identical fp64 CPU weights
   and inputs, and assert agreement to `<1e-12` on: scan path, stream path,
   stream-matches-scan, session carry with and without decay, clip on/off,
   gate on/off. Parity is run with the old side pinned to
   `v_source="hidden_state"`, since that is the only mode the new code has
   (D11).
2. **String/structure parity**: LoRA target regex byte-identical; param
   classification identical over a synthetic name list; session schedules
   identical under the same seed; metric values identical on recorded
   fixtures captured from the old code. The old modules default to
   `TTT_DATASET=arxiv`, so every parity test pins the SlimPajama spec
   explicitly rather than relying on either side's import-time default.
3. **Checkpoint compatibility**: an existing `ttt_params.pt` /
   `per_source_carries.pt` loads into the new model with matching keys.
4. **Gated GPU smoke** (`-m gpu`, manual): the `sanity_check` identity test
   and a 20-step training run on one H100.
5. Docs + a `MIGRATION.md` recording what moved where and what was
   intentionally dropped.
**Done when:** parity is green and you decide whether to retire the old files.

---

## 6. Effort shape

| Phase | Source files | Test files | Relative size |
|---|---|---|---|
| 0 | 6 | 2 | S |
| 1 | 16 | 13 | L |
| 2 | 7 | 4 | S–M |
| 3 | 29 | 11 | L |
| 4 | 8 | 11 | L |
| 5 | 10 | 5 | M |
| 6 | 2 | 6 | M |

~78 source files, ~52 test files, none of them large — that's the point of
"one concept = one file."

D11/D12 take roughly 500 lines off the port before it is written (two
strategies, the embedding tap and its plumbing, the preset registry, the
gate-reg path, one entrypoint, `param_health`, and the eval slicing branch),
plus two dependencies. Phase 2 shrinks the most: two registries instead of
three, four plugins instead of eight.

---

## 7. Defects found in the current code (carried into the rebuild as fixes)

1. ~~**`tests/test_scan_math.py::test_scan_matches_sequential_reference_session_carry`
   fails today**~~ (max err 3.5e-3 vs a 1e-6 tolerance). The reference oracle
   models the carry as a pure sum; `_scan_forward` has applied a
   `carried_decay=0.9` EMA since that config field was added. The oracle was
   never updated. The mechanism is probably right and the oracle stale — but
   *nothing in the repo currently proves which*.
   **Settled in Phase 1.** `tests/core/_oracles.py::sequential_carry` takes
   `decay` as a parameter and `tests/core/test_carry.py` pins both limits:
   `decay=1.0` reproduces the pure sum the old oracle assumed, `decay<1.0`
   reproduces the EMA the mechanism runs, and a third test asserts the two
   disagree — which is exactly the 3.5e-3 the old suite reported. The
   mechanism was right; the oracle was stale.

2. **`tests/test_loss_mask.py` is dead** — it imports eight functions deleted
   from `train_utils.py` in commit `00f2a53`, so the suite doesn't even
   collect without `--ignore`. Not ported (D7); I'll flag the old file for
   deletion at cutover.

3. `run_holdout_eval`'s snapshot/restore of `carried_delta` and
   `_next_carried` around the eval pass has no test. It is exactly the kind
   of orchestration bug the spec's Ring-4 fake-adapter tests exist to catch;
   Phase 4 covers it.

---

## 8. Resolved (2026-07-26)

1. **`pipelines/pipeline.py` is out of scope.** The arXiv→HF data-prep job
   stays where it is; it has no import relationship to the training code.
   Separate pass later if wanted.
2. **Package root is `ttt/`** — D2 confirmed.
3. **Cutover deferred.** Build side-by-side; the old files are not deleted in
   Phase 6. Retiring them is a separate explicit call after parity is green
   and a real GPU run has been done against NewCode.
4. **arXiv is dropped; SlimPajama is the corpus.** See D9/D10 for the
   invariants and the rename this unlocks.
5. **Cut list confirmed** (D11) and **logging budget set** (D12).
6. **`docs/` is kept**, to be rewritten later (D13).
7. **Chat is kept and upgraded** into a carry-ablation surface (D14), with
   three defects on that path fixed as part of the port.

---

## 9. Still open — carried forward, not decided

These were raised as cut candidates and are **not** in the D11 list. Ported
as-is unless you say otherwise; none of them blocks starting Phase 0, but the
first one changes the shape of Phases 2–5 substantially, so it is worth
settling before Phase 2.

1. ~~**The streaming / `stateful=True` path, generation, and chat.**~~
   **Resolved: kept and upgraded — see D14.** The reference table below stays
   because the naming distinction it records is load-bearing for the rebuild.

   **The streaming state is not the carry.** The mechanism holds two disjoint
   state families, and the old names actively obscure it:

   | | `carried_delta` — **the carry, the research** | `state.delta` — streaming state |
   |---|---|---|
   | lives in | `_scan_forward` | `_stream_forward` / `_commit_chunk` |
   | active when | `session_mode=True` | `stateful=True` |
   | managed by | `reset_session_state`, `advance_session_state`, `snapshot_carried_delta`, `install_carried_delta` | `reset_fast_weights`, `export_fast_weights`, `import_fast_weights` |
   | used by | training TBPTT, per-source everlasting carriers, in-loop eval, 5-mode eval, the pilot | `generate()` and `chat_turn()` — nothing else |

   Verified: `_scan_forward` never references `self.state`, `_stream_forward`
   never references `carried_delta` — the two share no state at all. Only
   four call sites in the repo set `stateful=True`, all in `infer_modal.py`
   (`generate`, `chat_reset`, `chat_turn`). Every measurement path runs
   `stateful=False`; the pilot says so in a comment.

   Cutting the streaming path therefore removes **chat and free-text
   generation only** — `TTTState`, `_stream_forward`, `_commit_chunk`,
   `export/import_fast_weights`, `reset_fast_weights`,
   `reset_v_left_context`, `stream_pending_progress`,
   `state_norms(source="stream")`, `chat_utils.py`, `chat_client.py`,
   `holdout_generate`, `chat_turn`, `chat_reset`, `save_session` — not the
   carry, and not any measurement path.

   **The vocabulary gets fixed in the rebuild.** `carry` means
   `carried_delta` and nothing else; the streaming family is named
   `stream_state` throughout. No function called `*_fast_weights` survives —
   that name straddles both families, and the ambiguity is what made "kill
   the streaming path" read as "kill the carry", *and* what let the two
   incompatible key schemes of D14 defect 2 coexist unnoticed.
2. **`v_bidirectional`.** Your own comment: "breaks chunk-causality under
   standard NTP; carry can leak ground-truth right-context." A knowingly
   invalid ablation, one branch in `_targets`, three tests.
3. **Training resume (`resume_from`).** Optimizer momentum isn't preserved, so
   a resume was never a faithful continuation anyway. Loading a checkpoint
   *for eval / the pilot* stays regardless.
4. **`LORA-ONLY` eval config and `compare_ppl`.** With `evolve=False`, FULL's
   `fresh` regime already is "TTT silent"; LORA-ONLY differs only by the
   trained `W_down`. Costs a Modal container per eval.
5. **`_stratified_sample_indices` and uniform holdout sampling.** Unreachable
   at default config (`eval_n_docs_per_source=1`), and the proposal specifies
   ≥30 docs per source — per-source *is* the sampling policy.
6. **LoRA / PEFT itself.** The largest possible cut (~150 lines of prefix
   juggling, the two-file checkpoint dance, a 3-group optimizer, one
   dependency) but a research decision, not a cleanup one.
7. **JSONL eval records vs. printed tables.** ~350 lines of f-string table
   code across six printers, each with two variants, recomputing the Δ
   decomposition in three places. Emitting one record per
   `(config, source, doc, regime, seed)` would collapse them to one generic
   renderer and move analysis off the Modal container.
8. **Two additions the proposal requires but the code lacks:** bootstrap 95%
   CIs over doc-order seeds (`core/stats.py`, needed for SO3/A1–A4) and a
   sweep runner for RQ4's chunk_size × carried_decay grid. Not in the tree
   above; say the word and I'll add them.
