---
title: "Session-Persistent Test-Time Training for Language Models: Methodology and Empirical Analysis"
subtitle: "Undergraduate Research Proposal"
author: "[Your Name]"
date: "[DD Month 2026]"
lang: en
geometry: margin=2.5cm
fontsize: 12pt
linestretch: 1.5
---

\newpage

# Cover Sheet

**Undergraduate Research Proposal**

*[KIC + NIBM Logos]*

**Proposed Study Title**

Session-Persistent Test-Time Training for Language Models: Methodology and Empirical Analysis

**Name of the Researcher**

[Your Name]

Kandy Innovation Centre,
National Institute of Business Management,
1095, Kandy – Colombo Rd, Getambe.

**Date of Submission:** [DD Month 2026]

\newpage

# Table of Contents

*[Auto-generate in MS Word: References → Table of Contents → Automatic Table 1.]*

\newpage

# 1. Details of the Individual Research Project

## Proposed Study Title

Session-Persistent Test-Time Training for Language Models: Methodology and Empirical Analysis

## Details of the Researcher

- **Name:** [Your Name]
- **Institutional Affiliation:** Kandy Innovation Centre, National Institute of Business Management, 1095, Kandy – Colombo Rd, Getambe.
- **Student Registration Number:** [Reg. No.]
- **Telephone No:** [Mobile]
- **E-mail:** chat.wilerathna@gmail.com

## Details of Internal Supervisor(s)

- **Name:** [Supervisor Name]
- **Institutional Affiliation:** Kandy Innovation Centre, National Institute of Business Management, 1095, Kandy – Colombo Rd, Getambe.
- **Telephone No (s):** Mobile: [ ] / Official: [ ]
- **E-mail (s):** [ ]

## Name of External Supervisor(s)

- **Name:** [If any, otherwise "Not Applicable"]
- **Institutional Affiliation:** [ ]
- **Telephone No (s):** Mobile: [ ] / Official: [ ]
- **E-mail (s):** [ ]

\newpage

# 2. Introduction

## Background Information

Modern large language models (LLMs) generate predictions with weights that are fixed at inference. Once training completes, every response the model produces comes from the same static parameters, regardless of what domain the input belongs to or what the user has recently been writing. This static-weight assumption underpins today's serving infrastructure but limits how well a model can specialise to the specific context in which it is being used.

Test-time training (TTT) relaxes this assumption. First proposed for image classification under distribution shift (Sun et al., 2020), TTT updates a subset of a model's parameters at inference using a self-supervised loss on the incoming input. In sequence modelling, Sun et al. (2024) introduced TTT-Linear and TTT-MLP, a family of sequence layers whose hidden state is itself a small learner updated as new tokens arrive. Behrouz et al. (2025) generalised these ideas in Titans, a family of neural long-term memory modules that memorise at inference. Fast-weight programmers (Schlag et al., 2021) and meta-learning frameworks such as Model-Agnostic Meta-Learning (Finn et al., 2017) provide theoretical grounding for both the update rules and the outer-loop training signal that shapes them.

Most of the above formulate TTT as a new sequence-layer architecture competing with attention. A more recent line — In-Place TTT (Feng et al., 2026) — instead retrofits pretrained transformer models by updating existing MLP down-projections with chunk-parallel rank-one writes during the forward pass. This makes TTT applicable to any pretrained transformer without redesigning the model from scratch, and reports substantial gains at long context (e.g., RULER-16k accuracy improving from 6.58 to 19.99). However, In-Place TTT's fast weight is reset per document: adaptation obtained on one input does not persist to the next. The natural next question — whether the adapted state can be usefully carried across items within a user's session — is left as future work.

## Research Problem

Existing plug-in test-time training methods operate at the granularity of a single document. Their adapted state is discarded at each document boundary, and no published method for pretrained language models both persists the fast weight across items in a session and initialises that persistent state from a meta-learned per-source starting point. As a result, the potential benefits of such session-persistent adaptation, the conditions under which it helps, and its interactions with source domain, session length, and configuration choices remain uncharacterised.

## Rationale

Understanding session-persistent test-time training matters for three reasons. First, from a scientific standpoint, characterising how test-time adaptation compounds across items reveals which parts of language-model quality come from static pretrained knowledge versus dynamically accumulated adaptation. Second, from a methodological standpoint, integrating session-persistent state and meta-learned initialisation into a plug-in retrofit framework broadens the design space of test-time training beyond single-document adaptation. Third, from a practical standpoint, a well-analysed method — one that documents when it helps, when it does not, and why — is more useful to future researchers and practitioners than a black-box mechanism claiming aggregate improvements. This project provides both the method and the analysis.

\newpage

# 3. Objectives

## Primary / Overall Objective

To develop and empirically analyse a session-persistent test-time training method for pretrained language models, characterising its behaviour across source domains, session lengths, and configuration axes.

## Specific Research Objectives

**SO1.** Design a session-persistent test-time training method that extends the plug-in test-time training paradigm (In-Place TTT) with (a) cross-document persistence of the fast weight, and (b) a meta-learned per-source initialisation for that persistent state.

**SO2.** Implement the method as a retrofit on a pretrained transformer language model and establish an evaluation harness that isolates each component's contribution.

**SO3.** Conduct an empirical analysis characterising the method's behaviour across source domains, session lengths, and key configuration choices, using rigorous per-source token-weighted metrics with bootstrap confidence intervals.

**SO4.** Synthesise the analysis into a discussion of when the method helps, when it does not, and how offline meta-learned initialisation interacts with online cross-item adaptation.

\newpage

# 4. Proposed Research Methods

## General Description of the Study Field

The study lies at the intersection of test-time adaptation, meta-learning, and language modelling. Experiments will be conducted on decoder-only transformer language models in the 0.6B–1.7B parameter range — small enough to run multiple training and evaluation configurations within a modest cloud-compute budget, and large enough to exhibit the phenomena reported in the recent test-time-training literature. Evaluation will follow the token-weighted perplexity (PPL) and negative-log-likelihood (NLL) conventions used by Sun et al. (2024) and Feng et al. (2026).

## Research Questions

The analysis is organised around four research questions:

- **RQ1.** How much does session-persistent state (fast-weight persistence across documents) contribute to language-modelling quality beyond within-document test-time adaptation?
- **RQ2.** Does a meta-learned per-source initialisation add benefit over a zero-initialised persistent carry, and how does that benefit change as session length grows?
- **RQ3.** Does the method's per-token benefit vary systematically across source domains (e.g., prose-heavy vs. code corpora)?
- **RQ4.** How sensitive is the method to core configuration choices — chunk size, decay of the persistent carry, and choice of seed source?

Because the specific architectural instantiation is expected to evolve during the project, the research questions are framed at the level of the method's behaviour rather than any particular implementation detail.

## Methods for Data Collection *(links to SO2, SO3)*

Training and evaluation data will be drawn from SlimPajama (Together AI / DKYoon subsample), a publicly available multi-source language-modelling corpus that includes ArXiv, Books, C4, GitHub, StackExchange, and Wikipedia. The multi-source composition is essential to RQ3, which asks how the method's benefit varies across source domains. A deterministic held-out split (last-N-documents policy) will be used to construct evaluation sets. Held-out documents will be tokenised on-the-fly and sampled per-source with sufficient documents (a minimum target of 30 per source) to permit bootstrap confidence intervals on token-weighted metrics.

Additionally, a small set of synthetic multi-document sessions will be constructed by concatenating held-out documents from a single source, in randomised orders drawn from multiple seeds. These sessions are the primary experimental unit for RQ1 and RQ2.

No human-subjects data or personally identifiable information will be collected. All datasets are publicly available under permissive research licences; ethical clearance is not required.

## Methods for Analyses of Data *(links to SO3, SO4)*

The analysis will report token-weighted NLL and PPL per document, per source, and per configuration, together with bootstrap 95% confidence intervals over doc-order seeds.

The empirical characterisation will be organised into four analyses, each mapped to a research question:

- **A1 (RQ1).** *Ablation of cross-document persistence.* Compare NLL under three regimes: fresh (no test-time adaptation), within-document only (adaptation resets per document), and session-persistent (adaptation carries across documents). The gap between the second and third isolates the cross-document contribution.
- **A2 (RQ2).** *Session-length dynamics with and without the seed.* Report per-document NLL as a function of position within a synthetic session, contrasting zero-initialised persistence and meta-learned seeded persistence. Characterise how the seed's contribution changes as the session grows.
- **A3 (RQ3).** *Per-source behaviour.* Report the analysis of A1 and A2 broken down by source, testing whether the method's benefits are domain-specific.
- **A4 (RQ4).** *Configuration sensitivity.* Sweep the two most important configuration parameters (chunk size and the decay coefficient of the persistent carry) to characterise how they modulate the observed behaviour.

## Methods for Implementation of the Solution *(links to SO1, SO2)*

The method will be implemented on top of an existing in-place test-time training layer that performs chunk-parallel rank-one updates to the MLP down-projection of selected transformer layers. Three logically independent components will be exposed as configuration switches so that ablations can turn each on and off in isolation:

- **Within-document adaptation** — the standard within-forward chunk-scan update. This inherits from the In-Place TTT baseline.
- **Cross-document persistence** — a mechanism that carries the fast weight across documents within a session rather than resetting it per document.
- **Meta-learned per-source initialisation** — an outer-loop training procedure that maintains a per-source persistent carrier during training and uses it as the initialisation of the persistent state at evaluation.

The training loop and evaluation harness will support toggling each component independently, which is necessary for the ablation analyses (A1–A2). The specific architectural details of the fast-weight update rule may evolve during the project as pilot results inform design choices; the analysis framework is designed to remain valid across such changes.

## Expected Results and Applications

- **Under SO1–SO2:** a working implementation of the method at 0.6B (and, resources permitting, 1.7B) parameter scale, integrated with the training pipeline and the evaluation harness. The implementation will be released as open source.
- **Under SO3:** a per-source, per-session-length, per-configuration characterisation surface reported with confidence intervals — i.e., a set of empirical plots and tables answering RQ1–RQ4.
- **Under SO4:** a discussion synthesising the analyses into practical guidance for when session-persistent test-time training helps and when it does not, and a characterisation of how offline meta-learned initialisation interacts with online cross-document adaptation.

## Methods of Comparing Models and Results

Comparisons are internal to the analysis: within a configuration, the three ablation regimes (fresh / within-only / session-persistent) are compared against each other; across configurations, the per-source characterisation is compared across sources, session lengths, chunk sizes, and decay values. Where applicable, a baseline drawn from prior test-time training work (e.g., In-Place TTT run on the same base model and data) will be included as a reference. The project does not aim to top a leaderboard benchmark; the axis of comparison is the mechanism's behaviour, not aggregate score.

## Resources

- **Compute:** Modal cloud GPU credits, estimated at ~USD 40–60 for the full set of runs (multiple sources × configurations × doc-order seeds × two model sizes at 300–500 training steps each), available through the researcher's existing account.
- **Models:** Publicly available pretrained Qwen 3 checkpoints (0.6B and 1.7B) accessed via Hugging Face.
- **Datasets:** SlimPajama-6B (public); no purchase required.
- **Software:** PyTorch, Hugging Face `transformers`, the researcher's existing test-time training training/evaluation codebase.
- **Storage:** Modal volumes, within existing account limits.
- **Personnel:** The researcher, with internal supervisor guidance.
- **Ethics approval:** Not required. The study uses publicly available textual corpora only. No human, animal, or personally identifiable data is involved.

\newpage

# 5. Timeline

Nine-week schedule; a Windows Visio Gantt chart will be included in the submitted document.

| Week | Task                                                                                                        | Deliverable                        |
|------|-------------------------------------------------------------------------------------------------------------|------------------------------------|
| 1    | Literature check; finalise evaluation harness; freeze the ablation-toggle interface (SO2)                   | Working evaluation infrastructure  |
| 2    | Preliminary training runs at 0.6B; verify all three components toggle cleanly                               | Baseline reference results         |
| 3    | Analysis A1 — cross-document persistence ablation, all sources                                              | Persistence-contribution plots     |
| 4    | Analysis A2 — session-length dynamics with and without meta-learned seed                                    | Per-position NLL curves            |
| 5    | Analysis A3 — per-source characterisation (extending A1 and A2)                                             | Per-source tables and plots        |
| 6    | Analysis A4 — configuration sensitivity (chunk size, decay)                                                 | Sensitivity plots                  |
| 7    | Synthesis; supervisor review; targeted follow-up runs to close gaps                                         | Draft §Results, §Discussion        |
| 8    | Draft §Introduction, §Related Work, §Method; supervisor review                                              | Revised full draft                 |
| 9    | Final formatting; submission                                                                                | Final document                     |

\newpage

# 6. Status of Permits and Clearance

Not applicable. The study uses only publicly available textual corpora and pretrained model checkpoints. No permits or ethical clearance are required.

\newpage

# 7. References

Behrouz, A., Zhong, P., & Mirrokni, V. (2025). Titans: Learning to memorize at test time. *arXiv preprint arXiv:2501.00663*.

Feng, S., et al. (2026). In-Place Test-Time Training. *arXiv preprint arXiv:2604.06169*.

Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1126–1135).

Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. In *Proceedings of the 38th International Conference on Machine Learning* (pp. 9355–9366).

Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). Test-time training with self-supervision for generalization under distribution shifts. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 9229–9248).

Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., & Guestrin, C. (2024). Learning to (learn at test time): RNNs with expressive hidden states. *arXiv preprint arXiv:2407.04620*.

Together AI. (2023). RedPajama & SlimPajama: Open reproductions of the LLaMA pretraining corpus. Retrieved from https://github.com/togethercomputer/RedPajama-Data.

\newpage

# 8. Annexure

**Ethical Clearance Application:** Not applicable to this project. The study uses only publicly available textual corpora and pretrained model checkpoints; no human subjects, personal data, or animal subjects are involved. A one-paragraph declaration to this effect can be attached in place of a formal application if the institution requires it on file.
