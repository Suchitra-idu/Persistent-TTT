# Paper framing: handoff for the next session

Read this first. Then `paper/TODO.md` (the plan), `results/step600_hebbian_vs_deltachunk.md` (the numbers), `docs/research.md` and `docs/mechanism.md` (the mechanism and prior findings).

## The paper

A negative-result paper, properly framed. Claim: persistent fast-weight state (hebbian and delta_chunk TTT) on top of a LoRA-fine-tuned LM gives no measurable cross-document benefit, and the within-document gain is small next to LoRA's. Scoped to: one base model, one corpus family (SlimPajama-6B, 6 sources), 600-step training, one seed per setting.

Never write "TTT does not work". Every claim is "in this setup". Target is a Q2 venue; the user has about 2 weeks from 2026-10-10, works part-time, and has budget for **one** LoRA-only training run (600 steps). Everything else must be evaluation-only. Do not propose new training, an lr grid, extra seeds, or retraining hebbian.

## Why a negative framing

The user reports from earlier tests that LoRA with a higher lr outperforms LoRA+TTT by a large margin. LoRA's lr is deliberately low in this repo so the update signal reaches the TTT layers. That is a baseline-handicap confound, so the one training run should be **LoRA-only (no TTT layers) at the tuned higher lr, 600 steps**, reported honestly as a tuned baseline and not an lr-matched ablation. First check whether the earlier high-lr runs left a usable checkpoint or logs; if so, evaluate that and skip training.

## Results so far (step 600, 106 held-out docs, 6 sources)

Δppl, positive = mechanism lowers perplexity, `*` = 95% paired-bootstrap CI excludes zero.

| measure | hebbian | delta_chunk |
|---|---|---|
| lora | +0.650 * | +0.772 * |
| within | +0.034 * | +0.058 * |
| between_slices | +0.002 | +0.012 * |
| total | +0.687 * | +0.842 * |
| cross_doc | -0.004 | -0.002 |
| cross_slice | +0.005 | +0.015 * |
| repeat_9_vs_1 | +0.000 | +0.000 |
| same_vs_other_source | +0.010 * | +0.002 |

Reading: LoRA is ~90% of the gain; within-doc TTT is +0.3-0.5%; cross-doc carry is null under both rules and flat across repeats; the only carry signal is hebbian `same_vs_other_source` (+0.09%, significant but negligible, flag it as such). Domain correlates use 6 sources, so treat them as noise. Delta_chunk has the better raw numbers.

## Checkpoints and where they live

Modal volume `ttt-checkpoints`. Each complete checkpoint has `adapter/`, `ttt_params.pt`, `per_source_carries.pt`.

- Hebbian clean run: `ttt-v1.1/step_N` (steps up to 847 exist, incl. 600, 650, 800). Trained with `update_rule=hebbian,state_aware_gate=false,save_every=50,eval_every=50,limit_docs=4000`, stopped at 600. Its `step_600/ttt_params.pt` sha256 starts `b463fe3a` and matches `backups/hebbian-clean-step600`.
- Delta_chunk run: `ttt-v1.1-clean-deltachunk/step_N` (up to 1000). Trained with `update_rule=delta_chunk,state_aware_gate=false,save_every=50,eval_every=50,limit_docs=4000,run_name=ttt-v1.1-clean-deltachunk`, stopped at 600.
- Other volume entries: `ttt-v1`, `Hebbian50-backup`, `freezeHebbian50-backup`, `delta-backup/step_600` (params and carries only), `lang_corpus*`. Not used; the freeze run is explicitly not wanted.
- Local `backups/*` hold only `ttt_params.pt` and `per_source_carries.pt`, **no LoRA adapter**, so they cannot be evaluated alone. `legacy/hebbian-freeze50-step600/` is the freeze run (has an adapter) and is a different run (different hash). Do not use it.

## Results on disk

`results/` is gitignored. Layout: `results/<run>/step_N/<stamp>/{SUMMARY.md, manifest.json, run.log, tables/, figures/, raw/}`.

- `results/ttt-v1.1/step_600/20261009-223740/` hebbian
- `results/ttt-v1.1-clean-deltachunk/step_600/20261009-220029/` delta_chunk
- `results/step600_hebbian_vs_deltachunk.md` side-by-side
- Per-source tables are in each `tables/*.csv`; not yet analysed.
- Raw rows are also on the volume under `evidence/`. Rebuild tables and figures without a GPU: `python -m ttt.experiments.paper_eval_v1 results/<run>/step_N/<stamp>`.

## How to run an evaluation

```
PYTHONPATH=. setsid nohup .venv/bin/modal run --detach ttt/experiments/paper_eval_v1.py \
  --checkpoint <run>/step_N --flags "<the flags it was trained with>" \
  > <log> 2>&1 < /dev/null &
```

- Use the venv's modal (`/home/suchitra/MyStuff/TTT/Our-TTT/.venv/bin/modal`).
- Pass the training flags; for delta_chunk include `run_name=ttt-v1.1-clean-deltachunk`.
- Always `--detach` and `setsid`: the first delta_chunk run was cancelled (cause unknown) while the local process hung and had to be killed and relaunched.
- A checkpoint with no TTT layers runs the `regimes` suite only (use `--suites regimes`).
- One full run is 4 suites on an H100 (4h cap) and took well under an hour for 106 docs.
- Use `modal volume ls ttt-checkpoints <path>` to inspect the volume; `modal volume get` to download. A partial download of the whole volume (387 MB) was started and stopped; `checkpoints/modal-volume-snapshot/` is a leftover and can be deleted.
- The Modal workspace shown in run URLs changed between the two runs (aiteera-llc, then suchitra-idu); not investigated.

## Repo conventions that apply

- No excessive comments or docstrings; one-line docstrings; rationale lives in `PLAN.md`/`ARCHITECTURE.md`. Enforced by `tests/architecture/test_comment_budget.py`.
- Active research is `update_rule=hebbian`. Delta/delta_chunk stay in code and are being used here only as the error-correcting contrast. Do not raise `carried_decay` above 0.9 for hebbian without saying it is untested and likely unsafe at the current `clip_tau` (a real run with 0.98 blew up, `docs/research.md` finding 7).
- Git: branch `before_lng`, many uncommitted changes plus untracked `backups/`, `graphs/`, `paper/`. `.gitignore` now excludes `checkpoints/` and `results/`. Nothing has been committed by this session. Commit trailers per the session reminder.

## Open decisions for the user

- Include delta_chunk in the paper (recommended: yes, the null replicating across both rules strengthens the claim).
- Whether a usable high-lr LoRA checkpoint already exists (decides whether the training run is needed).
- Venue and format: confirm it can be submitted to within the two weeks.

## Next actions, in order

1. Check for an existing high-lr LoRA checkpoint; else launch the one LoRA-only training run.
2. While it trains: positive control (repeated/near-duplicate doc), later checkpoints (hebbian step 800, delta_chunk step 1000), larger eval set, zeroed/shuffled-carry controls.
3. Analyse per-source tables; fix the single primary comparison (`cross_doc`, pooled) before looking at new runs.
4. Draft as results land; see the day-by-day plan in `paper/TODO.md`.
