# Preregistration: fresh evaluation pass

Committed and pushed before any fresh-set evaluation is run. Every analysis below is fixed here;
anything not listed is exploratory and will be labelled so. Deviations, if any, are listed in a
"Deviations" section appended after the run, never by editing this text.

Written 2026-11-02. Plan context: `plan/PLAN.md` §1 and §4 (Tier 1).

## 1. Questions

1. Does the carry's gain across Wikipedia documents survive a generic Wikipedia carry, i.e. is any
   of it document-specific?
2. Is the relatedness ordering (same > related > random > other source) present on fresh data?
3. How does the carry compare with placing the same prime in attention?
4. Does harm grow with the number of earlier documents carried?
5. Does training the carry across document boundaries (mixed-session probe) remove the harm while
   keeping the gain?

## 2. Models (all step 600, no other checkpoints)

| Label | Checkpoint | Use |
|---|---|---|
| carry s42 | `hebbian-carry-600/step_600` | all suites |
| carry s43 | `hebbian-carry-600-s43/step_600` | all suites |
| mixed s42 | `mixed-carry-600/step_600` (strategy `mixed`, otherwise the s42 flags) | relatedness, chain, regimes |
| LoRA-only | `lora-only-matched/step_600` | regimes reference |

## 3. Data

**Wikipedia relatedness cases.** `wikimedia/wikipedia` 20231101.en, shards 2 and 3 only
(development used shards 0 and 1). Targets have at least 12,000 characters, primes at least 6,000.
For each target, in a seeded order (seed 2027):
- scored text: the target's second 1,024-token block;
- same: the target's first 1,024 tokens;
- related: the article the target mentions most often by whole title (titles of at most 4 words),
  at least twice; **each related article is used for one target only**;
- random: an article that neither mentions nor is mentioned by the target;
- other source: a 1,024-token block of a fresh SlimPajama holdout document from a non-Wikipedia
  source.

200 cases are reported. A case is skipped, and the next in the seeded order taken, if its target,
related or random article shares at least 5 ten-word sequences (lower-cased, from its first 2,000
characters) with any Wikipedia document among the 4,000 SlimPajama training documents selected with
seed 42 or seed 43 (first 700 tokens, decoded). SlimPajama's Wikipedia text carries no title line and
comes from an earlier dump, so neither titles nor exact hashes can identify the same article; shared
word sequences can. The number skipped is reported. Cases are built once, saved to the checkpoint
volume (`evidence/cases/fresh.json`), and reused for every model.

**SlimPajama regimes.** A new holdout seed (`eval_holdout_seed=1`) from the holdout pool, which is
disjoint from training by construction. Up to 40 documents per source are drawn, any of the 106
development documents (holdout seed 0) are removed by token sequence, and the first 20 per source
are kept (fewer for Book if the pool is short; the count is reported). These documents also supply
the other-source primes.

## 4. Arms

Every arm scores the same target tokens; only what precedes them differs.

- **reset**: carry reset, no prime.
- **same, related, random, other source**: carry primed on that block, prime not in attention.
- **generic**: carry set to the mean of the carried states of 200 random shard-2/3 articles that
  are neither targets nor primes and pass the overlap check, each layer rescaled to the median
  Frobenius norm of those 200 single carried states. Built separately for each model from the same
  200 articles.
- **in context**: carry reset; the related prime and the target read as one input, only the target
  tokens scored. The within-input fast weights see the prime as they would any earlier text of the
  same input.
- **chain k** (k = 1, 2, 5, 10): carry primed on k articles in sequence, then the target scored.
  Each case has its own 10 chain articles (first 1,024 tokens each), none a target, prime,
  generic-carry or other case's chain article; chain k uses the first k. The carry is restored
  after each scoring, so scoring the target never feeds the chain.

## 5. Measure and statistics

- Δppl of an arm against a reference: the difference of geometric-mean perplexities over cases,
  positive when the arm has lower perplexity, as in the development results. Headline numbers also
  give it as a percent of the reference perplexity.
- Intervals: paired percentile bootstrap over cases, 10,000 resamples, resampling seed 0. Win rate
  is the share of cases on which the arm is lower.
- Each seed is reported separately; seeds are never pooled.
- No p-values.

## 6. Headline contrasts (main models, each seed)

| # | Contrast | Reported with |
|---|---|---|
| H1 | same minus generic | 95% and Bonferroni-adjusted 98.75% interval |
| H2 | related minus generic | 95% and 98.75% |
| H3 | related minus random | 95% and 98.75% |
| H4 | related (carry) minus in context | 95% and 98.75% |

Every arm against reset is also reported with a 95% interval.

## 7. Decision rules

- **Document-specific transfer.** The phrase appears in the title and abstract only if the 98.75%
  interval of H2 excludes zero, positive, in both seeds. Otherwise the claim is "domain-level
  adaptation", and H3 is reported as the document-specific gap.
- **Replication of the ordering.** "The relatedness ordering replicates on fresh data" is written
  only if same > related > random > other source holds in point estimates in both seeds and the
  95% intervals of related minus random and random minus other source exclude zero in both seeds.
- **Chain length.** If chain-10 minus chain-1 is negative with a 95% interval excluding zero in both
  seeds, the boundary-versus-relatedness conflict is explained as accumulated interference.
  Otherwise it is stated as open.
- **Mixed-session probe** (mixed s42 against carry s42, paired on the same cases). Four contrasts,
  each the mixed model's gain over its own reset arm minus the carry model's gain over its own,
  with both models resampled on the same cases: other source, chain 10, related, and the
  between-slice gain on fresh SlimPajama; each with a 95% interval. "Training across documents
  removes the harm" is written only if the other-source and chain-10 contrasts are positive with intervals excluding zero and the related
  contrast's interval does not lie wholly below zero. If the mixed model's related and
  between-slice gains are both within 0.01 of zero, the result is reported as "the probe learned to
  ignore the carry". Every mixed result is labelled "in a single-seed probe".

## 8. Fixed in advance, not tuned

Prime and target lengths (1,024 tokens), the minimum-mention rule, the generic-carry size (200),
the chain lengths, the bootstrap settings and the case seed. Nothing is tuned on the fresh cases.
Development-set results stay reported as development results.
