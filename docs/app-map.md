# Ring 4 — `ttt/app/`

The loops. Each module reaches the outside world only through the ports, so all
of them run against the fake adapters in the tests, with no GPU, network or disk.
Ring 4 may not import torch directly or any adapter (`.importlinter`;
`make guard` plants such an import and expects the check to fail).

## Data

| File | Does |
|---|---|
| [`stages.py`](../ttt/app/stages.py) | The nine pipeline stages, each `Data -> Data`: `split_holdout`, `filter_sources`, `shuffle`, `prefilter`, `balance`, `tokenize`, `drop_short`, `rebalance`, `cap`. |
| [`data_pipeline.py`](../ttt/app/data_pipeline.py) | `load` runs the stages; `documents` turns the table into `Doc`s; `holdout` draws the held-out documents per source (`eval_holdout_seed`, 20 per source in the paper's evaluations). |

The training rows and the held-out rows are split first, by position (the last
15,000 rows are held out), so no held-out document can reach training.
`balance` over-fetches and `rebalance` re-applies the source ratios after
tokenizing, because dropping short documents removes more of some sources than
others.

## Training

| File | Does |
|---|---|
| [`train_loop.py`](../ttt/app/train_loop.py) | `train`: sessions from the strategy, one forward and backward per item, gradient accumulation, clipping, the optimizer step, the carry lifecycle (`reset_carry` per session, `advance_carry` per item, `detach_carry` per carry window and per step), logging, checkpoints, and in-training evaluation. Non-finite losses and gradients are skipped and counted, never stepped on. |

## Evaluation suites

Each returns rows, one per document (or case) and condition; [evaluation.md](evaluation.md)
says what each measures.

| File | Suite |
|---|---|
| [`eval_loop.py`](../ttt/app/eval_loop.py) | `regimes`: the four regimes over 8 slices per document, and the gap decomposition. Restores the carry it found, so it is safe to call mid-training. |
| [`session_eval.py`](../ttt/app/session_eval.py) | per-item perplexity with the carry persisting; the building block of the boundary and repeat suites |
| [`boundary_eval.py`](../ttt/app/boundary_eval.py) | `boundary`: chains of same-source documents, carried or reset at the boundary |
| [`repeat_eval.py`](../ttt/app/repeat_eval.py) | `repeat`: one document read ten times with an unreset carry |
| [`context_eval.py`](../ttt/app/context_eval.py) | `context`: the carry against the previous slice in attention |
| [`relatedness_eval.py`](../ttt/app/relatedness_eval.py) | `relatedness`: case selection (`pick`, `mentions`), the primed arms, the generic carry, the overlap check (`shingles`, `overlaps`), the contrasts and the preregistered headline |
| [`token_gain.py`](../ttt/app/token_gain.py) | `token_gain`: per-token losses on the relatedness cases |
| [`dynamic_eval.py`](../ttt/app/dynamic_eval.py) | `dyneval`: the dynamic-evaluation baseline, its grid and tuning |

## Statistics over the rows

| File | Does |
|---|---|
| [`evidence.py`](../ttt/app/evidence.py) | Rows in, estimates out: the paired bootstrap over documents, the regime gaps (`REGIME_GAPS`), the boundary and repeat estimates, per-source summaries, JSON row I/O. |
| [`paper_analysis.py`](../ttt/app/paper_analysis.py) | Cross-checkpoint estimates: rebasing on a control's `fresh` rows, the lead over a matched model, per-position estimates, clip saturation, state statistics. |
| [`token_gain_analysis.py`](../ttt/app/token_gain_analysis.py) | The preregistered Addendum A: where in the target the gain falls, tokens in neither prime, the generic share, with cluster bootstrap over shared primes. |
