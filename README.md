# Session-persistent test-time training

Code for the paper *Session-Persistent Test-Time Training for Language Models:
Transfer and Interference of a Carried Fast Weight Across Documents*
(`Paper_TMLR/main.tex`).

Test-time training (TTT) updates some weights of a language model while it
reads. In-Place TTT does this inside the MLP blocks of a pretrained transformer
and resets that fast weight at every document boundary. This code carries it
forward instead: a **gated carry** passes the fast weight built on earlier text
to later text through its own zero-initialised, gated path, trained by
backpropagation across slice boundaries. It is built on Qwen3-0.6B, trained on
SlimPajama-6B and evaluated in perplexity on held-out documents and on English
Wikipedia.

## The method in brief

Each of the 14 odd-numbered MLP layers becomes

```
out = W_down z + g(h) η S_in z + α g_c(h) η clip(S_carried) z
```

- `S_in` is written inside the current slice, 50 tokens at a time.
- `S_carried <- 0.9 S_carried + ΔS_item` accumulates the earlier slices.
- `g`, `g_c` are learned gates; `α` starts at zero, so the layer starts as In-Place TTT.

[docs/mechanism.md](docs/mechanism.md) maps every part to the code.

## Layout

```
ttt/
  core/         pure math and logic: the TTT kernels, the carry, statistics, configs
  extensions/   the TTT layer, session strategies, dataset specs
  ports/        interfaces to everything costly (compute, fast weights, data, storage, ...)
  adapters/     real and fake implementations of the ports; model builder; Modal runtime
  app/          the data pipeline, the training loop, every evaluation suite
  cli.py        flags in, frozen configuration out
  experiments/  the entrypoints: train, evaluate, and the analysis scripts for the paper
tests/          about 1,600 fast tests, mirroring ttt/
results/        the result bundles of every run in the paper, and results/paper/ (tables, figures)
Paper_TMLR/     the paper
docs/           the documentation
```

Imports point only inward, from `experiments` down to `core`; `make check`
enforces it. [docs/codebase.md](docs/codebase.md) explains the structure and
follows a training run and an evaluation through it.

## Setup

```
python3.11 -m venv .venv
make install-dev
make check
```

Training and evaluation run on [Modal](https://modal.com) with an H100. They
need a Modal account (select it with `MODAL_PROFILE`), the volumes
`ttt-checkpoints` and `hf-hub-cache`, and the secrets `wandb` and `huggingface`.
The analysis of existing results needs none of this.

## Usage

Train:

```
modal run --detach ttt/experiments/train_v1.py --flags \
  "carry_gate=true,carry_backprop=true,output_gate_bias_init=-6,run_name=my-gated-carry"
```

Evaluate a checkpoint:

```
modal run ttt/experiments/paper_eval_v1.py --checkpoint my-gated-carry/step_600 \
  --suites regimes,boundary --out-dir results --flags "<the training flags>"
```

Regenerate the paper's tables and figures from `results/`, on a CPU:

```
python -m ttt.experiments.cross_run_tables_v1
python -m ttt.experiments.report_figures_v1
```

The exact commands for every run in the paper, the meaning of each result
directory, and which script writes which table are in
[docs/reproducing.md](docs/reproducing.md).

## Documentation

Start at [docs/README.md](docs/README.md).
