# From the paper's findings to the code

The paper (`Paper_TMLR/main.tex`) asks two questions:

- **Q1.** How much does carrying the fast weight contribute to language modelling?
- **Q2.** Which domains benefit most?

Every answer is a perplexity difference on held-out text. This page lists each
finding with the suite that measures it ([evaluation.md](evaluation.md)), the
runs it compares ([reproducing.md](reproducing.md#run-names-and-what-they-are))
and the table or figure it comes from.

| Finding | Suite and measure | Runs | Output |
|---|---|---|---|
| The gated carry lowers perplexity on the continuation of a document by about 1% | `regimes`, `between_slices` | `hebbian-carry-600`, `-s43`, `gated-carry-1050`; confirmatory `-fresh` bundles | T11, T14; F15 |
| Removing the reset alone gains little | `regimes` and `boundary` | `ttt-v1.1`, `ttt-v1.1-clean-deltachunk` | T1, T3, T11 |
| Only the full design recovers the gain between slices | `regimes`, `boundary` at step 200 | `ablate-*`, `hebbian-carry-b` | T13 |
| Across documents the effect follows relatedness | `relatedness` | gated carry, development and `-fresh` | T12; F18; `tables/relatedness.csv` of the `-fresh` bundles |
| A small part of the transfer is document-specific (related against the generic carry) | `relatedness`, `HEADLINE` contrasts | `-fresh` bundles | `tables/relatedness.csv` |
| The gain sits at the start of the target | `token_gain`, Addendum A | `-tokengain-fresh`, `-tokengain-generic-fresh` | `addendum_a.json`; F19 |
| Training across documents removes the harm after other-source text | `relatedness`, `regimes` | `mixed-carry-600`, `-s43` against the gated carry | `mixed_probe_v1` |
| Most of the lead over LoRA-only comes from training the down-projections | `regimes`, compared across checkpoints | `lora-only-matched`, `pmatched-control-600`, gated carry | T1; `results/compare/` |
| Dynamic evaluation as a baseline | `dyneval` | `lora-only-matched` | T12, T15 |
| Attention is a far stronger channel when the text is available | `context` | gated carry | the bundles' `context_gaps` tables |
| The carried state's size barely reaches the output | `boundary`, `repeat` state ratios | ungated and gated runs | T6, T16 |
| Domains (Q2) | `regimes` and `boundary`, per source | gated carry | T7, T8 and the bundles' per-source tables |

## What the evidence can and cannot support

Every measurement is perplexity, which supports claims of stream and domain
adaptation, not of memory or recall. The intervals are over documents, not over
training runs. The preregistered contrasts are the four in
`relatedness_eval.HEADLINE`, at a Bonferroni-adjusted 98.75%; everything outside
them is reported without adjustment. The preregistration and its deviations are
in `plan/PREREGISTRATION.md`.
