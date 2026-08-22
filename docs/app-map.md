# Ring 4 map — the loops

`ttt/app/` is orchestration and nothing else. It reaches the outside world only
through Ring 2 protocols, so every module here runs against fakes with no GPU,
no network and no disk — which is what the whole suite does, in milliseconds.

Ring 4 may not import a framework at all, **torch included**. That is enforced
(`.importlinter`, `app-stays-framework-free`) and `make guard` proves it by
planting an `import torch` in `ttt/app/` and expecting the lint to reject it.

| File | Does | Talks to |
|---|---|---|
| `stages.py` | The nine pipeline stages, each `Data -> Data` | `Table`, `Tokenizer`, `Rng` |
| `data_pipeline.py` | `PIPELINE`, the holdout split, `Doc` | `DataSource` + the above |
| `train_loop.py` | Accumulate, clip, step, log, save, eval, carry | `Compute`, `FastWeights`, `Tracker` |
| `eval_loop.py` | Six-regime held-out perplexity, gap decomposition | `Compute`, `FastWeights` |
| `session_eval.py` | Per-item perplexity with the carry persisting | `Compute`, `FastWeights` |
| `pilot.py` | Cold / persist / seeded compounding, over `session_eval` | — |
| `generate.py` | The one token loop | `Generation` |
| `chat.py` | Turn orchestration and the four D14 switches | `Generation`, `FastWeights`, `Tokenizer`, `Rng` |

---

## The pipeline

`load_token_dataset` was 145 lines with two rebalance passes, an over-fetch
multiplier, an `only_mode` branch and a char-count fallback all interleaved.
It is now nine functions with one `Data -> Data` signature:

```
split_holdout → filter_sources → shuffle → prefilter → balance
              → tokenize → drop_short → rebalance → cap
```

Three of the old branches are gone rather than moved:

- **`if "source" in ds.column_names`** — D9 makes the label universal, so
  `filter_sources`, `balance` and `rebalance` have no `has_source` fallback.
- **`if tokens_est_column`** — D9 makes the estimate universal, so `prefilter`
  is single-branch over `core.tokens`.
- **`only_mode`** — an `--only-sources` pick is a preset with equal weights, and
  `core.balance` already drops sources a preset omits. `cap` is last in the list
  anyway, so the "defer the limit" special case dissolves.

**Why two balancing passes.** `drop_short` is source-biased: Books survive it
because they are long, C4 does not because it has many short documents. A mix
balanced before tokenizing is not one after. So `balance` over-fetches by
`PRESET_OVERSAMPLE`, and `rebalance` re-applies the exact ratios once the true
token counts are known.

**Ratios are only exact under a limit.** With no `limit_docs`, both balancing
passes target the pool size, and a majority source's quota is bounded only by
what it has — so the mix moves toward the preset without reaching it. Set
`limit_docs` when the ratio matters. This is the old behaviour, now visible in
the stage log rather than buried in a print.

**`Table.filter` exists for the prefilter.** `column()` materialises; SlimPajama-6B
is millions of rows of text and would not fit. `filter(name, predicate)` is
row-wise on both adapters (`dataset.filter` on the HF side), so the pass that
shrinks the corpus never holds it. `tokenize` does read a column, by which point
the over-fetch has bounded the pool.

---

## The train loop

Three decisions the old loop made implicitly, now explicit and tested:

**A nonfinite loss does not defer the accumulation boundary.** The old code
`continue`d before the boundary check, so a nonfinite landing on the last micro
of a window rolled that window's gradients into the next one — a double-size
step, and a step count that drifts out of sync with the LR schedule. The rebuilt
loop still skips the backward, still advances the carry, still counts the micro,
and still steps at the boundary with one micro's contribution missing.

**A window with no finite loss at all does not step.** Otherwise AdamW would
apply weight decay to every parameter on the strength of no gradient.

**The rate logged is the rate used.** The old loop called `scheduler.step()`
after `optimizer.step()` and logged `get_last_lr()`, which is the *next* step's
rate. `_learning_rates` takes the count before the step, so `train/lr_*` is what
the optimizer actually applied.

Checkpointing and eval are injected callables, not imports: serialization needs
torch and lives in `ttt/adapters/checkpoint_io.py`, which Ring 4 may not touch.
Ring 5 supplies both.

---

## Eval, and what the metric names mean

The metric keys predate the regime vocabulary and disagree with it. Both are
kept — the keys because a dashboard reads them, the regimes because
`core.metrics` decomposes over them:

| Metric key | Regime | Starts from | Between slices |
|---|---|---|---|
| `eval/carry_ppl` | `cold_carry` | zero | persists |
| `eval/carry_off_ppl` | `cold_carry_off` | zero | reset |
| `eval/fresh_ppl` | `fresh` | — (TTT off) | — |
| `eval_seed/carry_ppl` | `carry` | the trained per-source carrier | persists |
| `eval_seed/carry_off_ppl` | `carry_off` | the carrier, reinstalled | reset |

`fresh` is not measured twice: with `evolve=False` the TTT branch is bypassed
entirely, so a seed cannot reach it.

`eval_seed/gap_seed` compares the seeded documents **against themselves cold**,
never against the whole pool — the seeded subset is whichever sources happen to
have a carrier, which is not a random sample of the holdout.

Per-source numbers are the RQ3 deliverable and are *not* logged: D12 cut them
from wandb, where they were ~40 keys of noise per eval. They come back as
`EvalReport.summaries`, for the console table and the analysis records.

**Eval restores what it found.** It snapshots the carry, runs, and reinstalls in
a `finally` — so firing it mid-training is a no-op against the loop that called
it. The old `run_holdout_eval` did this too and had no test; it has two now,
including one that raises mid-measurement.

---

## Chat: the switch matrix (D14)

| Switch | Off | On |
|---|---|---|
| `within_turn` | fast weight frozen for the turn | chunk-scan updates as tokens stream |
| `cross_turn` | state cleared at the turn boundary | delta survives, **decayed** |
| `seed` | start from zero | install a named source's trained carrier |
| `context` | `none` — TTT is the only memory | `full` — ordinary history, the control |

All three D14 defects are fixed here, and each fix has a test that fails when
the fix is reverted:

1. **The seed could not reach chat.** Chat runs the stream path, which reads the
   streaming delta; the trained carrier was written to the carry. One `Carry`
   representation plus `install(carry, family=STREAM)` connects them.
2. **Two incompatible key schemes.** Killed in Phase 3 — there is one `Carry`,
   keyed by base-model layer index, and nothing enumeration-keyed to mix it with.
3. **No decay at the turn boundary.** `decay_stream(factor=...)` applies the same
   EMA training uses, so chat stops driving `state_ratio` into a regime the model
   never saw. Passing `decay=1.0` reproduces the old unbounded sum, which is what
   the defect looked like.

Per-turn diagnostics answer "is the carry doing anything" rather than leaving it
to be guessed: `state_ratio`, `pending_tokens / chunk_size` (a zero state ratio
means either "frozen" or "chunk not full yet" — this disambiguates), the share of
the state this turn added, and gate mean/std. A gate pinned near zero means the
TTT term cannot be reaching the output, which is the first thing to check when
two arms of an A/B read identically.

`ab_turn` runs the same turn on two sessions. With an identical `sampling.seed`
both arms draw the same tokens, so a difference attributes to the switches rather
than to sampling noise.
