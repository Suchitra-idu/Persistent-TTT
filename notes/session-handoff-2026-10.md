# Session handoff, October 2026

State of the research, every decision, and the next steps, written so work can
resume without the chat history. Source of truth for numbers is `results/`.

## People, constraints, preferences

- Researcher: M A S I Malimbada (KIC-BSCCSAI-25.1-P-014), supervisor Prof. Roshan D. Yapa.
- Budget is tight. Ask before launching GPU work; keep runs short; stop runs with a watcher.
- Writing rules for the thesis and anything for others: no em dashes, no en dashes, no colons in prose.
- Figures as PDF, not PNG.
- `results/` is the source of truth. Every number in the thesis must trace to a file there.
- Thesis cites only works in `proposal.md` and `litReview.tex`; main focus is our own method.
- In-training eval (`eval_every`) is wanted; keep it on (50 steps).
- Goal: publish within about 6 months (conference or journal, top tier welcome), free or
  minimal cost, for master's or PhD applications. Also wants research opportunities.

## Infrastructure facts

- Launch with `~/venv-ardupilot/bin/modal` (the project `env/` has no torch).
- Modal workspace `aiteera-llc`, volume `ttt-checkpoints`. The `suchitra-idu` workspace has a
  different volume; both contain a `ttt-v1.1` folder holding different runs.
- Hebbian `ttt-v1.1` checkpoints after step 650 are from an older run; excluded.
- Launch detached: `setsid nohup ... modal run --detach ttt/experiments/train_v1.py --flags ... &`
  and a detached watcher script that stops the app after the target checkpoint.
- Always `save_every=50` for experimental runs (two runs were lost to save_every=200).
- Epoch = 1050 optimizer steps; never use `max_steps` to shorten a run (it rescales the LR
  schedule); stop the app instead.
- Local tests: `~/venv-ardupilot/bin/python -m pytest tests -o addopts="" -q -W ignore`;
  `tests/parity/test_cli_parity.py` fails on HEAD too (pre-existing).
- TeX is not installed locally; the thesis is compiled on Overleaf.

## Runs and results (all on the same 106 held-out docs, 6 SlimPajama sources)

| run | what | key result |
|---|---|---|
| `lora-only-matched` | control, no TTT layers, same data/LR/seed | base ppl GM 12.838 |
| `ttt-v1.1` | Hebbian | vs LoRA +0.07..0.11; within +0.014..0.034; cross-doc ~0 |
| `ttt-v1.1-clean-deltachunk` | delta_chunk | vs LoRA +0.19..0.23; within +0.06..0.28; cross-doc ~0 |
| `delta-noclip` | delta without clip | gate closes (0.00), all TTT effects 0, -0.167 worse |
| `hebbian-noclip` | Hebbian without clip | collapses (loss ~8.1 at step 200) |
| `hebbian-carry-b` | carry design to 200 | between-slice +0.135 @100, +0.099 @200 |
| `hebbian-carry-600` | carry design to 600 | between-slice +0.062 @400, +0.097 @600; cross-doc (100 docs) +0.010 @200, -0.003 @400, -0.020 * @600; vs LoRA +0.234 @600; carry recovers 4.0% of previous-slice-in-context gain |

Carry design flags (Hebbian): `carry_gate=true,carry_backprop=true,output_gate_bias_init=-6`
(plus code defaults `carry_window=2`). Full flags string used:
`update_rule=hebbian,state_aware_gate=false,limit_docs=4000,carry_gate=true,carry_backprop=true,output_gate_bias_init=-6,eval_every=50,save_every=50,log_every=10,run_name=<name>`

Failed attempts (documented in `results/paper/inputs/carry_design_training/README.md`):
1. carry path without zero-init scale doubled perplexity (tau=20 is 4-5x W0's largest
   singular value, `inputs/w0_spectral.json`);
2. with scale but gate bias -2: early Hebbian blow-up until ~step 140, then OOM at ~195
   (fixed by `carry_window=2` and `expandable_segments`).

Pipeline bugs found and fixed: LoRA dropout active in eval; `fresh` regime kept trained W_down
(old bundles rebased onto the LoRA-only bundle's true base).

## Claim and framing (agreed)

"When a carried fast weight helps and when it hurts": a naive carry fails for a structural
reason (saturated read-time clip); a trained, gated carry is a small but reliable memory of
related text (same document, survives 600 steps) and interference for unrelated text
(cross-document harm after full training); attention on the preceding text is far stronger
(carry recovers ~4%). Thesis subtitle updated to match.

## Code added this session (uncommitted unless the user committed it)

- `ttt/core/config/ttt.py`: `carry_gate`, `carry_backprop`.
- `ttt/core/config/train.py`: `max_steps`, `carry_window`.
- `ttt/extensions/mechanism/inplace_ttt.py`: carry path, zero-init `output_gate_carry_scale`,
  `detach_carry`, `RecomputeAsForward` / `checkpoint_context_fn` (gradient-checkpoint fix),
  `pretrained_mlps`.
- Ports: `Compute.backward(retain_graph=)`, `Compute.eval_loss(score_from=)`,
  `FastWeights.detach_carry()`.
- `ttt/app/eval_loop.py`: per-slice carry gain metrics; `ttt/app/context_eval.py`;
  `ttt/app/paper_analysis.py`.
- `ttt/experiments/paper_analysis_v1.py`, `thesis_figures_v1.py`, `_paper_figures.py`.
- `paper_eval_v1.py`: opt-in `context` suite.

## Outputs

- `results/paper/`: tables T1..T11, figures F1..F15 (PDF), `FINDINGS.md`, `inputs/`.
- `paper/main.tex` + `paper/references.bib` + `paper/figures/`: thesis (report class, ~10.4k words).
- Regenerate: `python -m ttt.experiments.paper_analysis_v1 --control lora-only-matched --run hebbian=ttt-v1.1 --run delta_chunk=ttt-v1.1-clean-deltachunk --steps 200,400,600 --w0-norms results/paper/inputs/w0_norms.json`
  then `python -m ttt.experiments.thesis_figures_v1`.

## Publication plan (proposed, user deciding)

- arXiv preprint now (needs endorsement for cs.LG / cs.CL).
- Option A: TMLR (free, fast, judged on correctness) submitted about December 2026.
- Option B: ACL 2027 via ARR, deadline 4 Jan 2027 (Findings as fallback); TMLR if rejected.
- Cannot submit to two venues at once.

## Next experiments (agreed as needed, not yet run)

Budget estimate about 8 GPU-hours, about $32-40 total.
1. DONE 2026-10-25, see FINDINGS.md 4d (A and B synergistic, C alone ~nothing). Original plan: Ablation, 200 steps each: "C only" (gate -6, no carry_gate, no carry_backprop) first, it is
   the key control; then A+C and B+C. A+B is covered by failed attempt 2.
2. DONE 2026-10-25, see FINDINGS.md 4e (replicates). Original plan: Second seed of the carry design to 600 (`seed=43`), evals at 400 and 600.
3. DONE 2026-10-25, see FINDINGS.md 4c (Wikipedia, not code; code carry too weak). Original plan: Relatedness test (eval only, on `hebbian-carry-600/step_600`, and `ttt-v1.1/step_600`):
   data `codeparrot/github-code-clean` loaded as plain parquet shard
   (`data/train-00000-of-00880.parquet`, has repo_name, path, code, language; not gated;
   the dataset's loading script is unusable with datasets>=4). For each Python target file,
   score its second half after priming the carry with: nothing, its own first half, another
   file of the same repo, a file of another repo (same language), a non-code document. Equal
   prime length (~1024 tokens), drop boilerplate and duplicates, ~100 targets from ~50 repos.
   Implement as `ttt/app/relatedness_eval.py` + opt-in `relatedness` suite in paper_eval.

## Pending housekeeping

- Delete junk checkpoints on `aiteera-llc`: `smoke-*`, and `delta-carry-bptt` if unused.
- Commit the code above.
- Supervisor email drafted (in chat); framing paragraph updated to the "when it helps" framing.

## Status 2026-10-25

All planned experiments are done: relatedness (4c), ablation (4d), second seed (4e).
Thesis zip v8 includes all three. Modal workspace in use switched to `vihanganoob123`
for the seed-43 run; its checkpoints live on that workspace's volume.

## Status 2026-10-26

- Dynamic-evaluation baseline done (FINDINGS 4f): carry transfers more to related and random
  articles, dyneval fits the same document more. Run sharded (`--dyneval-grid`, `--dyneval-fixed`,
  `--case-slice`, `--tag`, `merge_bundles_v1`) after a preemption lost a single-container run.
- Repeat suite on the carry design: no accumulation.
- Thesis reframed: subtitle "A Carried Fast Weight Transfers to Related Documents and Hurts
  Unrelated Ones"; abstract, contributions, conclusion lead with cross-document transfer.
- `lora-only-matched/step_600` and `hebbian-carry-600/step_600` copied to the vihanganoob123
  volume (sha256 verified).

## Status 2026-10-27

- Thesis rebuilt with a full literature review from `Notes/ALL_NOTES.md` (70 bib entries); old version in git 992d756.
- Audit applied. Subtitle now "Transfer and Interference of a Carried Fast Weight Across Documents".
  Claim is "transfers by relatedness; harms after other-source text or a long same-source chain".
  Ch. 6 is "Why the Naive Carry Fails"; the redesign keeps a saturated clip (carry state ratio
  median ~8k s42 / ~15k s43). Boundary-vs-relatedness conflict on Wikipedia stated as open.
- Reviewer gaps still open: equal-memory KV-cache baseline, dyneval on the carry model,
  word-overlap control, behavioural probe after eviction, more seeds.
