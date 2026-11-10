# Evaluation suites

Every measurement in the paper comes from one entrypoint,
[`ttt/experiments/paper_eval_v1.py`](../ttt/experiments/paper_eval_v1.py). It
loads one checkpoint, runs the suites you name with `--suites`, and writes a
*bundle*: one folder holding the raw per-document rows, the tables computed
from them, a manifest and a summary. This page says what each suite measures,
which module does it, and where it appears in the paper.

All gains are differences of perplexity, positive when the second condition is
lower. Intervals are paired percentile bootstrap intervals over documents
(`ttt/core/stats.py`, `ttt/app/evidence.py`).

## The regimes suite (`regimes`)

Module: [`ttt/app/eval_loop.py`](../ttt/app/eval_loop.py). Each held-out
document is cut into 8 equal slices (`schedule.equal_token_slices`) and read
slice by slice under four conditions:

| Regime | Model | Fast weights | Carry between slices |
|---|---|---|---|
| `fresh` | the pretrained base model (`pretrained_mlps`, LoRA off) | off | no |
| `lora_only` | the trained model with fast weights off | off | no |
| `cold_carry_off` | the trained model | on, reset at every slice | no |
| `cold_carry` | the trained model | on | yes, from earlier slices |

The gaps between them (`evidence.REGIME_GAPS`):

| Measure | Compares | In the paper |
|---|---|---|
| `lora` (`training`) | `fresh` → `lora_only` | what training changed, fast weights off |
| `within` | `lora_only` → `cold_carry_off` | "within a slice" |
| `between_slices` | `cold_carry_off` → `cold_carry` | "between slices", the headline within-document gain |
| `total` | `fresh` → `cold_carry` | everything together |

The earlier slices are never in the attention context, so `between_slices` is
what the carry adds when that text is gone. `regime_slices` holds the same
rows per slice position, for "does the gain grow along the document".

Comparing two checkpoints (for example the lead over the LoRA-only control) is
done afterwards, from two bundles, by `paper_analysis.matched_estimates` and
[`compare_checkpoints_v1.py`](../ttt/experiments/compare_checkpoints_v1.py).

## The boundary suite (`boundary`)

Module: [`ttt/app/boundary_eval.py`](../ttt/app/boundary_eval.py). For each
source, a chain of documents (`--n-chain`, 20 for the paper's gated runs) is read
one after another, every document cut into slices, under three arms:

| Arm | Reset between documents | Reset between slices |
|---|---|---|
| `persist` | no | no |
| `doc_reset` | yes, as In-Place TTT does | no |
| `slice_reset` | yes | yes |

`cross_doc` is `doc_reset` → `persist`: the carry from earlier documents of the
same source, the paper's "into another document". The first document of a chain
is skipped, because it is identical under both arms.

## The repeat suite (`repeat`)

Module: [`ttt/app/repeat_eval.py`](../ttt/app/repeat_eval.py). Each document is
read 10 times with a carry that is never reset. Repeat 0 is the cold reading,
the reference for every later one. It records the state ratio
`‖η S‖_F / ‖W0‖_F` (`carry.state_ratio`) at each reading; the paper uses it for
"What the carried state passes on" (the second reading adds a little, later ones
nothing, while the state grows 6.5-fold over ten readings).

## The context suite (`context`)

Module: [`ttt/app/context_eval.py`](../ttt/app/context_eval.py). Each slice after
the first is scored on the same tokens three ways: alone with the state reset,
alone with the carried state, and with the previous slice placed before it in
attention. The paper's "attention lowers perplexity about 25 times more than
the carry" comes from here.

## The relatedness suite (`relatedness`)

Modules: [`ttt/app/relatedness_eval.py`](../ttt/app/relatedness_eval.py), cases
built by [`ttt/experiments/_wiki_relatedness.py`](../ttt/experiments/_wiki_relatedness.py).
It uses English Wikipedia (November 2023). For each case:

- the **target** is the second 1024-token block of an article;
- a **prime** is a 1024-token block that the carry reads first, then the carry is
  kept and the target scored. The primes are, from most to least related:
  `self` (the target article's first block), `related` (the article the target
  mentions most often by title, at least twice, titles of at most four words),
  `random` (an article that neither mentions nor is mentioned by the target),
  `other_source` (a document from another SlimPajama source);
- the **reset** arm scores the target with no prime.

Optional arms: `generic` (`--n-generic 200`: the mean carried state of 200 other
articles, each layer rescaled to their median norm, `generic_carry`),
`in_context` (`--in-context`: prime and target as one input, carry reset), and
`chain_k` (`--chain-lengths 1,2,5,10`: k random articles carried before the
target).

Contrasts: `self_vs_related`, `related_vs_random`, `random_vs_other_source`
(`CONTRASTS`), and the four preregistered ones in `HEADLINE`
(`self_vs_generic`, `related_vs_generic`, `related_vs_random`,
`carry_vs_in_context`), also at the Bonferroni-adjusted level 98.75%.

Cases are built once and saved with `--cases-file`, so every model is scored on
the identical cases; a saved file is rejected if the settings that choose its
articles differ. `--overlap-seeds 42,43` removes articles whose opening shares
at least five ten-word sequences with a Wikipedia training document of either
seed.

**Development and confirmatory sets.** The development cases use shards 0–1 and
seed 0. The preregistered confirmatory cases use `--wiki-first-shard 2
--wiki-case-seed 2027 --unique-related --n-boot 10000 --cases-file fresh`, with
`eval_holdout_seed=1` and `--exclude-holdout-seed 0` for the 73 new held-out
documents. Bundles of the confirmatory pass end in `-fresh` and are never mixed
with development ones (`paper_analysis_v1.is_development`).

## The token-gain suite (`token_gain`)

Module: [`ttt/app/token_gain.py`](../ttt/app/token_gain.py). The relatedness
cases scored per token, so a gain can be located in the target. The analysis,
the preregistered Addendum A, is [`ttt/app/token_gain_analysis.py`](../ttt/app/token_gain_analysis.py),
run by [`token_gain_analysis_v1.py`](../ttt/experiments/token_gain_analysis_v1.py);
it writes `tables/addendum_a.json` into the bundle (the 54–56% of the gain on
the first 31 tokens, the tokens in neither prime, the generic share).

`--target-tokens 32` and `128` score only the target's first tokens; those are
the short-text bundles (`-short32-fresh`, `-short128-fresh`).

## The dynamic-evaluation baseline (`dyneval`)

Modules: [`ttt/app/dynamic_eval.py`](../ttt/app/dynamic_eval.py),
[`ttt/experiments/_dynamic_adapter.py`](../ttt/experiments/_dynamic_adapter.py).
On the LoRA-only model, Adam steps on the LoRA adapters over the prime, then the
target is scored and the adapters restored. The learning rate and number of steps
are tuned on 40 separate cases (`--n-tune`) for each objective.

## The training probe across documents

[`mixed_probe_v1.py`](../ttt/experiments/mixed_probe_v1.py) compares the bundles
of the probe trained across documents (`mixed-carry-600`) with the gated carry
on the same cases, for the paper's "Training across documents".

## What a bundle holds

```
results/<run>/step_<n>/<stamp>[-<tag>]/
    raw/<suite>.json      one row per document (or case, token) and condition
    tables/*.csv, *.md    estimates with intervals, per source and pooled
    manifest.json         the command, the resolved config, the plan, git state
    SUMMARY.md            the pooled estimates
```

Re-render tables from the raw rows, no GPU:
`python -m ttt.experiments.paper_eval_v1 results/<run>/step_<n>/<stamp>`.
