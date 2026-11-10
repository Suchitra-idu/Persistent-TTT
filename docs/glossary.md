# Glossary

## The method

| Term | Means |
|---|---|
| **fast weight** `S` | A weight the model writes during the forward pass, from the text it reads, not by gradient descent |
| **In-Place TTT term** | `g(h) η S_in z`: the fast weight written inside the current item, read through the main gate |
| **carry**, `S_carried` | The fast weight kept from earlier items of a session: `S_carried <- 0.9 S_carried + ΔS_item` |
| **ungated carry** | `carry_gate=false`: the carry is added to the state the scan starts from. In-Place TTT with the reset removed |
| **gated carry** | `carry_gate=true`: the carry is read through its own clip, gate `g_c` and zero-initialised scale `α`, beside the In-Place TTT term. The paper's method also sets `carry_backprop=true` and gate bias `-6` |
| **training through the carry** | `carry_backprop=true`: the carry stays in the autograd graph for `carry_window=2` items, so a later item's loss trains an earlier item's write |
| **chunk** | `chunk_size=50` tokens; one fast-weight write. A chunk reads only earlier chunks |
| **η** | the fixed fast-weight step size, 2 |
| **target `V`** | what the fast weight is trained to produce: a causal depthwise convolution (kernel 8) over the normalised hidden states, then `W_target`, initialised to zero |
| **W0 / W_down** | the `down_proj` weight of a TTT layer |
| **update rule** | `hebbian` (default; `ΔS = VᵀZ/C`) or `delta_chunk` (`ΔS = (V − V̂)ᵀZ/C`, error from the chunk-start state) |
| **main gate `g`** | `output_gate`: a per-token sigmoid on the In-Place TTT term, weights zero, bias `output_gate_bias_init` |
| **carry gate `g_c`, scale `α`** | `output_gate_carry`, `output_gate_carry_scale`: the gated carry's read |
| **clip, τ** | rescaling a state so `‖ηS‖_F ≤ 20`, keeping its direction |
| **state ratio** | `‖ηS‖_F / ‖W0‖_F`, the carry's size relative to the base weight |
| **carried_decay** | 0.9, the factor on the carry each time an item is added |

## Training

| Term | Means |
|---|---|
| **doc** | one document |
| **item** | a contiguous token range of one document: one forward and one backward |
| **slice** | an item cut out of a longer document |
| **session** | the items over which the carry is kept; it is reset between sessions |
| **strategy** | how documents become sessions: `hybrid` (one document per session) or `mixed` (five documents, crossing boundaries) |
| **parameter groups** | `lora`, `wdown`, `new`, each with its own learning rate |
| **step** | one optimizer step, after `grad_accum_steps=16` items |

## Data

| Term | Means |
|---|---|
| **source** | the domain of a document: `RedPajamaArXiv`, `RedPajamaBook`, `RedPajamaC4`, `RedPajamaGithub`, `RedPajamaStackExchange`, `RedPajamaWikipedia` |
| **preset** | per-source mixing weights; `slim-research` is the paper's |
| **holdout** | the last 15,000 rows of the dataset, never trained on |
| **development holdout** | 20 held-out documents per source (6 for Book), holdout seed 0 |
| **confirmatory holdout** | 73 new held-out documents, holdout seed 1 excluding seed 0's documents |

## Evaluation

| Term | Means |
|---|---|
| **regime** | a way of reading a held-out document: `fresh`, `lora_only`, `cold_carry_off`, `cold_carry` ([evaluation.md](evaluation.md)) |
| **between-slice gain** | `cold_carry_off` → `cold_carry`: what the carry adds within a document |
| **cross-document gain** | `doc_reset` → `persist` in the boundary suite |
| **suite** | one kind of measurement in `paper_eval_v1`: `regimes`, `boundary`, `repeat`, `context`, `relatedness`, `token_gain`, `dyneval` |
| **case** | in the relatedness suite, one target article and its primes |
| **prime** | the text the carry reads before the target: `self`, `related`, `random`, `other_source`, `generic`, `chain_k` |
| **bundle** | one folder of results for one checkpoint and pass: raw rows, tables, manifest |
| **fresh pass, `-fresh`** | the preregistered confirmatory evaluation |
| **pins** | `results/paper/inputs/bundle_pins.json`: which bundle each table reads |
| **ppl** | perplexity, `exp(mean token loss)`; lower is better |
| **gain** | a drop in perplexity, positive when the second condition is lower |

## Architecture

| Term | Means |
|---|---|
| **ring** | one of the six layers of `ttt/`; imports point inward ([codebase.md](codebase.md)) |
| **port** | an interface to something costly or nondeterministic |
| **adapter** | an implementation of a port; each port has a real one and a fake one |
| **conformance suite** | the tests every adapter of a port must pass |
| **registry** | name → plugin, for strategies and datasets |
| **contract suite** | the tests every registered plugin must pass |
| **oracle** | a slow, obviously correct reimplementation that a fast kernel is checked against |
| **D1–D14** | the design decisions in [REBUILD_PLAN.md](../REBUILD_PLAN.md), cited in code by number |
