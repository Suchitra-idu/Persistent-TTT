# FILE: INDEX.md

# Notes index — 80 papers

One `_importance.md` file per paper in this folder, each with **Citation /
Important parts for my research / What to cite / Verification**. Built by
reading every PDF in `../TTT Papers/` against `../Praposal.md` and
`../FuturePlan.md` (the real empirical status: LoRA-only beats LoRA+TTT,
carry is flat from the first repeat, RULER is flat/slightly regressed, and
the active update rule is `hebbian` — pure outer-product, no error
correction). Every note was fact-checked against its own source PDF, not
generated from memory.

## Read these first — the flat-carry explanation

These five converge on the same mechanistic story, independently, from five
different papers. Together they're close to a formal argument for *why*
`hebbian` produces a flat, non-compounding carry:

- **[Using fast weights to attend to the recent past](Using%20fast%20weights%20to%20attend%20to%20the%20recent%20past_importance.md)** (Ba & Hinton 2016) — closed-form: under repeated near-identical input, a decayed Hebbian accumulator converges to a fixed point almost immediately, not a compounding trajectory. The strongest single piece of theory in the whole set.
- **[Learning What to Remember: Test-Time Training via Context Distillation](Learning%20What%20to%20Remember%3A%20Test-Time%20Training%20via%20Context%20Distillation_importance.md)** (Wang et al. 2026) — Proposition 1 ties carry strength to a *nonzero error signal*; hebbian has none.
- **[Mamba](Mamba%3A%20Linear-Time%20Sequence%20Modeling%20with%20Selective%20State%20Spaces_importance.md)** — Theorem 1 + boundary-resetting discussion: a non-selective (non-gated) state update can't filter useful from irrelevant history.
- **[xLSTM](xLSTM%3A%20Extended%20Long%20Short-Term%20Memory_importance.md)** — own ablation shows non-gated updates fail to revise stale stored associations.
- **[State-space modeling in long sequence processing (survey)](State-space%20modeling%20in%20long%20sequence%20processing%3A%20a%20survey%20on%20recurrence%20in%20the%20transformer%20era_importance.md)** — frames DeltaNet's rule as explicitly "error-correcting" vs. plain accumulation, i.e. names the exact axis `hebbian` sits on the wrong side of.

Also central, not for the flat-carry story but because they're load-bearing
for the mechanism/method itself:

- **[In-Place Test-Time Training](In-Place%20Test-Time%20Training_importance.md)** (Feng et al. 2026) — the architecture this proposal directly extends. Confirms unconditional fast-weight reset at document boundaries (§3.4, Algorithm 1) and confirms it never compares to LoRA (verified against the full reference list).
- **[Titans](Titans%3A%20Learning%20to%20Memorize%20at%20Test%20Time_importance.md)** (Behrouz et al. 2025) — confirmed via full-text scan: never operates across independently-bounded documents/sessions, only within one long context. Sharpens the Research Gap paragraph.
- **[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](Learning%20to%20%28Learn%20at%20Test%20Time%29%3A%20RNNs%20with%20Expressive%20Hidden%20States_importance.md)** (Sun et al. 2025) — TTT-Linear/MLP; confirms DeltaNet = TTT-Linear at batch size 1 minus LN/residual.
- **[Linear Transformers Are Secretly Fast Weight Programmers](Linear%20Transformers%20Are%20Secretly%20Fast%20Weight%20Programmers_importance.md)** (Schlag et al. 2021) — the associative-memory reading of what a fast weight *is*.

## Already-cited-in-proposal papers, verified/extended

Praposal.md cites these directly; each note checks the proposal's existing
claim against the source and adds detail the proposal doesn't currently have.

| Paper | What got verified/added |
|---|---|
| [Test-Time Training with Self-Supervision...](Test-Time%20Training%20with%20Self-Supervision%20for%20Generalization%20under%20Distribution%20Shifts_importance.md) | Sun 2020, the original TTT formulation |
| [Titans](Titans%3A%20Learning%20to%20Memorize%20at%20Test%20Time_importance.md) | see above |
| [In-Place Test-Time Training](In-Place%20Test-Time%20Training_importance.md) | see above |
| [Let's (not) just put things in Context (qTTT)](Let%27s%20%28not%29%20just%20put%20things%20in%20Context%3A%20Test-time%20Training%20for%20Long-context%20LLMs_importance.md) | Bansal 2025 — query-only, attention-only, single-context |
| [End-to-End Test-Time Training for Long Context](End-to-End%20Test-Time%20Training%20for%20Long%20Context_importance.md) | Tandon 2025 — footnote 4 confirms documents are filtered specifically to avoid cross-boundary resets; RULER template for framing our own flat result |
| [Learning What to Remember (Context Distillation)](Learning%20What%20to%20Remember%3A%20Test-Time%20Training%20via%20Context%20Distillation_importance.md) | Wang 2026 — see Prop. 1 above |
| [Lifelong In-Context Learning...Parametric Forms of Attention](Lifelong%20In-Context%20Learning%20with%20Transformers%20Requires%20Parametric%20Forms%20of%20Attention_importance.md) | McDermott 2026 |
| [Transformer-XL](Transformer-XL%3A%20Attentive%20Language%20Models%20beyond%20a%20Fixed-Length%20Context_importance.md) | Dai 2019 — confirmed SG(·) stop-gradient = activation persistence, zero weight update; flags a confound (segment recurrence alone fixes fragmentation without needing long-range dependency) worth a Discussion mention |
| [RULER](RULER%3A%20What%27s%20the%20Real%20Context%20Size%20of%20Your%20Long-Context%20Language%20Models%3F_importance.md) | Hsieh 2024 — held-out probe suite |
| [Dynamic Evaluation of Neural Sequence Models](Dynamic%20Evaluation%20of%20Neural%20Sequence%20Models_importance.md) | Krause 2018 — the evaluation-design ancestor |
| [Continual Test-Time Adaptation in Computer Vision](Continual%20Test-Time%20Adaptation%20in%20Computer%20Vision%3A%20Methods%2C%20Benchmarks%2C%20and%20Future%20Directions_importance.md) | Maharana 2026 — CTTA survey; parameter-restoration/gradient-degeneracy findings parallel our flat-carry story |
| [Catastrophic forgetting in connectionist networks](Catastrophic%20forgetting%20in%20connectionist%20networks_importance.md) | French 1999 |

## Fast weights / update-rule family

[A Modern Self-Referential Weight Matrix](A%20Modern%20Self-Referential%20Weight%20Matrix%20That%20Learns%20to%20Modify%20Itself_importance.md) ·
[Fine-Tuning Pre-trained Transformers into Decaying Fast Weights](Fine-Tuning%20Pre-trained%20Transformers%20into%20Decaying%20Fast%20Weights_importance.md) ·
[Parallelizing Linear Transformers with the Delta Rule (DeltaNet)](Parallelizing%20Linear%20Transformers%20with%20the%20Delta%20Rule%20over%20Sequence%20Length_importance.md) ·
[Gated Delta Networks](GATED%20DELTA%20NETWORKS%3A%20IMPROVING%20MAMBA2%20WITH%20DELTA%20RULE_importance.md) ·
[Gated Linear Attention Transformers](Gated%20Linear%20Attention%20Transformers%20with%20Hardware-Efficient%20Training_importance.md) ·
[Transformers are RNNs (linear attention)](Transformers%20are%20RNNs%3A%20Fast%20Autoregressive%20Transformers%20with%20Linear%20Attention_importance.md) ·
[RWKV](RWKV%3A%20Reinventing%20RNNs%20for%20the%20Transformer%20Era_importance.md) ·
[HOPFIELD NETWORKS IS ALL YOU NEED](HOPFIELD%20NETWORKS%20IS%20ALL%20YOU%20NEED_importance.md) ·
[Hybrid computing with dynamic external memory (DNC)](Hybrid%20computing%20using%20a%20neural%20network%20with%20dynamic%20external%20memory_importance.md)

## Test-time training / test-time adaptation (broader)

[Test-Time Training with Masked Autoencoders](Test-Time%20Training%20with%20Masked%20Autoencoders_importance.md) ·
[The Surprising Effectiveness of TTT for Few-Shot Learning](The%20Surprising%20Effectiveness%20of%20Test-Time%20Training%20for%20Few-Shot%20Learning_importance.md) (Akyürek — TTT here is ephemeral, per-task, discarded, no persistence) ·
[TENT](TENT%3A%20FULLY%20TEST-TIME%20ADAPTATION%20BY%20ENTROPY%20MINIMIZATION_importance.md) ·
[Efficient Test-Time Adaptation of Vision-Language Models](Efficient%20Test-Time%20Adaptation%20of%20Vision-Language%20Models_importance.md) (cache-capacity-vs-noise ablation, good adjacent-domain precedent) ·
[s1: Simple test-time scaling](s1%3A%20Simple%20test-time%20scaling_importance.md) (decoding-time compute, *not* weight updates — a useful contrast to define TTT against)

## Long context: architecture & benchmarks

[Compressive Transformers](COMPRESSIVE%20TRANSFORMERS%20FOR%20LONG-RANGE%20SEQUENCE%20MODELLING_importance.md) (has a direct online-update-vs-boundary-reset discussion, §5.5.1) ·
[Recurrent Memory Transformer](Recurrent%20Memory%20Transformer_importance.md) ·
[MEMORIZING TRANSFORMERS](MEMORIZING%20TRANSFORMERS_importance.md) ·
[Efficient Streaming LMs with Attention Sinks](EFFICIENT%20STREAMING%20LANGUAGE%20MODELS%20WITH%20ATTENTION%20SINKS_importance.md) ·
[REFORMER](REFORMER%3A%20THE%20EFFICIENT%20TRANSFORMER_importance.md) ·
[Big Bird](Big%20bird%3A%20Transformers%20for%20longer%20sequences_importance.md) ·
[H2O](H2O%3A%20Heavy-Hitter%20Oracle%20for%20Efficient%20Generative%20Inference%20of%20Large%20Language%20Models_importance.md) ·
[YaRN](YARN%3A%20EFFICIENT%20CONTEXT%20WINDOW%20EXTENSION%20OF%20LARGE%20LANGUAGE%20MODELS_importance.md) ·
[LongRoPE](LongRoPE%3A%20Extending%20LLM%20Context%20Window%20Beyond%202%20Million%20Tokens_importance.md) ·
[Long Range Arena](LONG%20RANGE%20ARENA%3A%20A%20BENCHMARK%20FOR%20EFFICIENT%20TRANSFORMERS_importance.md) ·
[LongBench](LongBench%3A%20A%20Bilingual%2C%20Multitask%20Benchmark%20for%20Long%20Context%20Understanding_importance.md) (has a memorization-vs-understanding diagnostic worth reusing) ·
[EFFICIENTLY MODELING LONG SEQUENCES (S4)](EFFICIENTLY%20MODELING%20LONG%20SEQUENCES%20WITH%20STRUCTURED%20STATE%20SPACES_importance.md) ·
[Universal Transformers](Universal%20transformers_importance.md)

## Continual learning / catastrophic forgetting

[Overcoming catastrophic forgetting (EWC)](Overcoming%20catastrophic%20forgetting%20in%20neural%20networks_importance.md) ·
[Revisiting Catastrophic Forgetting in LLM Tuning](Revisiting%20Catastrophic%20Forgetting%20in%20Large%20Language%20Model%20Tuning_importance.md) ·
[Continual learning with deep generative replay](Continual%20learning%20with%20deep%20generative%20replay_importance.md) ·
[iCaRL](iCaRL%3A%20Incremental%20classifier%20and%20representation%20learning_importance.md) ·
[Continual Learning of LLMs: A Comprehensive Survey](Continual%20Learning%20of%20Large%20Language%20Models%3A%20A%20Comprehensive%20Survey_importance.md) (has an "anticipatory recovering" phenomenon — precedent for non-monotonic carry) ·
[Towards Lifelong Learning of LLMs: A Survey](Towards%20Lifelong%20Learning%20of%20Large%20Language%20Models%3A%20A%20Survey_importance.md) (confirmed via keyword search: never discusses test-time/session persistence — a clean negative-result citation) ·
[Mind the Gap](Mind%20the%20Gap%3A%20Assessing%20Temporal%20Generalization%20in%20Neural%20Language%20Models_importance.md)

## Meta-learning / in-context-learning theory

[Model-agnostic meta-learning (MAML)](Model-agnostic%20meta-learning%20for%20fast%20adaptation%20of%20deep%20networks_importance.md) ·
[Optimization as a model for few-shot learning](Optimization%20as%20a%20model%20for%20few-shot%20learning_importance.md) ·
[Meta-Learning in Neural Networks: A Survey](Meta-Learning%20in%20Neural%20Networks%3A%20A%20Survey_importance.md) ·
[Meta-Learning the Difference](Meta-Learning%20the%20Difference%3A%20Preparing%20Large%20Language%20Models%20for%20Efficient%20Adaptation_importance.md) (dynamic LoRA beats static LoRA — comparator for the LoRA-vs-LoRA+TTT result) ·
[Transformers Learn In-Context by Gradient Descent](Transformers%20Learn%20In-Context%20by%20Gradient%20Descent_importance.md) (von Oswald — isolates carry-driven gains from ordinary ICL) ·
[What learning algorithm is in-context learning?](What%20learning%20algorithm%20is%20in-context%20learning%3F%20Investigations%20with%20linear%20models_importance.md) (Akyürek — same ICL-vs-carry validity role) ·
[A Survey on In-context Learning](A%20Survey%20on%20In-context%20Learning_importance.md)

## Foundational / scaling / efficiency (background, cite as needed)

[Attention Is All You Need](Attention%20is%20all%20you%20need_importance.md) ·
[Language Models Are Few-Shot Learners (GPT-3)](Language%20models%20are%20few-shot%20learners_importance.md) ·
[Training Compute-Optimal LLMs (Chinchilla)](Training%20Compute-Optimal%20Large%20Language%20Models_importance.md) ·
[FlashAttention](FLASHATTENTION%3A%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness_importance.md) ·
[LoRA](LORA%3A%20LOW-RANK%20ADAPTATION%20OF%20LARGE%20LANGUAGE%20MODELS_importance.md) (directly used — `lora_only` ablation) ·
[GQA](GQA%3A%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints_importance.md) ·
[RedPajama](RedPajama%3A%20an%20Open%20Dataset%20for%20Training%20Large%20Language%20Models_importance.md) ·
[A Practical Survey on Faster and Lighter Transformers](A%20Practical%20Survey%20on%20Faster%20and%20Lighter%20Transformers_importance.md) ·
[Language model evaluation beyond perplexity](Language%20model%20evaluation%20beyond%20perplexity_importance.md) (methodological caution — we lean on perplexity) ·
[On the Dynamic Adaptation of Stochastic Language Models](On%20the%20Dynamic%20Adaptation%20of%20Stochastic%20Language%20Models_importance.md) (Kneser 1993 — earliest cache-adaptation precedent)

---

*One paper from `../literature_review.md` has no note here: SlimPajama-6B
(Yoon 2023) is a Hugging Face dataset card, not a paper — nothing to read.*

## Added 2026-10-27 — closest recent work, memory and claim calibration

PDFs are in `../New Papers/`. Notes were written from the PDFs; each note's Verification section says which parts were read.

- **[Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training](Beyond%20Perplexity%3A%20A%20Behavioral%20Evaluation%20Framework%20for%20Deployment-Memory%20Claims%20in%20LLM%20Test-Time%20Training_importance.md)** — Song et al. 2026 — claim levels S/B/D; perplexity-only TTT work is S-level; names "evidence migration"
- **[Test-Time Training with Next-Token Prediction](Test-Time%20Training%20with%20Next-Token%20Prediction_importance.md)** — Ouyang et al. 2026, Findings of EMNLP — In-Place TTT successor; restores weights after every sample; includes Qwen3-0.6B
- **[Can a Language Model Continually Learn Facts in its Weights?](Can%20a%20Language%20Model%20Continually%20Learn%20Facts%20in%20its%20Weights%3F_importance.md)** — O'Neill et al. 2026 — sequential fact writes retain 1–46%; later writes govern retention
- **[Delayed Supervision for Test-Time Language Models](Delayed%20Supervision%20for%20Test-Time%20Language%20Models_importance.md)** — Kim & Kothari 2026 — delayed QA loss trains test-time memory to retain facts within one stream
- **[Adapting Language Models to Compress Contexts](Adapting%20Language%20Models%20to%20Compress%20Contexts_importance.md)** — Chevalier et al. 2023 (AutoCompressors) — summary vectors; stop-gradient after 2 steps; fused across documents
- **[Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass](Generative%20Adapter%3A%20Contextualizing%20Language%20Models%20in%20Parameters%20with%20a%20Single%20Forward%20Pass_importance.md)** — Chen et al. 2024 — forward-pass outer-product adapter; SVD normalisation keeps only directions
- **[Cartridges: Lightweight and general-purpose long context representations via self-study](Cartridges%3A%20Lightweight%20and%20general-purpose%20long%20context%20representations%20via%20self-study_importance.md)** — Eyuboglu et al. 2025 — offline-trained KV cache per corpus; 38.6× less memory than ICL
- **[MemoryLLM: Towards Self-Updatable Large Language Models](MemoryLLM%3A%20Towards%20Self-Updatable%20Large%20Language%20Models_importance.md)** — Wang et al. 2024, ICML — fixed memory pool updated by forward passes with exponential forgetting
- **[Larimar: Large Language Models with Episodic Memory Control](Larimar%3A%20Large%20Language%20Models%20with%20Episodic%20Memory%20Control_importance.md)** — Das et al. 2024 — episodic memory written by least squares; one-shot edits and forgetting
- **[Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](Leave%20No%20Context%20Behind%3A%20Efficient%20Infinite%20Context%20Transformers%20with%20Infini-attention_importance.md)** — Munkhdalai et al. 2024 — compressive associative memory across segments; Hebbian or delta; gated
- **[Test-Time Training Done Right](Test-Time%20Training%20Done%20Right_importance.md)** — Zhang et al. 2025 (LaCT) — large-chunk TTT; L2-normalised fast weights against magnitude explosion

# FILE: A Modern Self-Referential Weight Matrix That Learns to Modify Itself_importance.md

# A Modern Self-Referential Weight Matrix That Learns to Modify Itself

**Citation:** Kazuki Irie, Imanol Schlag, Róbert Csordás, Jürgen Schmidhuber, 2022, ICML (Proceedings of the 39th International Conference on Machine Learning, PMLR 162)

## Important parts for my research
- Reviews the DeltaNet fast-weight update (Schlag et al. 2021): input projected to key `k_t`, value `v_t`, query `q_t`, and scalar learning rate `β_t` (Eq. 1); a "retrieved old value" `v̄_t = W_{t-1}φ(k_t)` is computed (Eq. 2); the fast weight matrix updates via `W_t = W_{t-1} + σ(β_t)(v_t − v̄_t) ⊗ φ(k_t)` (Eq. 3) — i.e., the delta rule writes the *difference* between the new target value and what the current weights already retrieve for that key, not a raw outer product. This is the explicit error-correction term absent from a pure Hebbian rule (relevant to explaining why Hebbian carry is flat: Hebbian just accumulates `v_t ⊗ φ(k_t)` with no correction for what's already stored, so repeated near-duplicate keys don't refine the state, they just re-add roughly the same term, consistent with a carry that doesn't grow).
- Fast weight matrix `W_0` is initialized to **zero** in both DeltaNet and SRWM.
- States the `φ` (elementwise positive, sums-to-one, e.g. softmax) applied to both writes (Eq. 3) and reads (Eqs. 2, 4) "is crucial for stability when the delta rule is used" (citing Schlag et al. 2021 for details) — direct statement that write/read normalization is what keeps delta-rule updates from blowing up.
- SRWM extends this so the "slow" projection weights themselves are also generated on the fly by the same self-referential matrix (Eqs. 5–8), collapsing the slow/fast programmer hierarchy into one matrix.
- **State-reset ablation (directly relevant to flat/non-compounding carry):** they tested resetting the SRWM's weight updates after every fixed time span (length = backprop truncation span) instead of carrying them across a whole (variable-length) episode. This ablation "failed to leverage the SRWM mechanism," scoring 28.5/16.1 (train/test) versus 35.0/20.0 for the full carry-across-episode model and 34.6/15.3 for "Fake SR" (no self-modification at all) — i.e., frequent-reset carry collapses to no-better-than-no-carry performance. Direct empirical precedent that carry needs to persist over a long enough span to matter, and that breaking persistence early erases any benefit.
- Explicitly frames the whole approach as a form of dynamic evaluation: "in the common method of dynamic evaluation (Mikolov et al., 2010; Krause et al., 2018), [error feedback] is also used to adapt the weights at test time using the fixed gradient descent algorithm. A self-referential approach could also adapt the corresponding adaptation algorithm."
- Notes standard limitation that NNs (including their self-modifying model) lack systematic/length generalization, and that "there is no guarantee that the SRWM's self-modifications preserve the original objective function it is trained for" (citing Hubinger et al. 2019) — a caution about assuming self-modification is always beneficial.
- Reports fast weight models (DeltaNet, SRWM) "clearly outperform" feedforward/LSTM baselines in multi-task RL but by modest, single-digit-to-low-double-digit normalized-score margins (e.g. aggregated test score 20.0 for SRWM vs 18.6 DeltaNet vs 16.4 feedforward vs 15.7 LSTM) — an example of a fast-weight mechanism producing only modest empirical gains even when it "works."

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The delta rule updates fast weights by writing the *difference* between a target value and the value the current weights already retrieve for that key, giving an explicit error-correction term that a pure Hebbian (outer-product-only) update lacks" (Discussion — explains flat carry; Irie et al. 2022, citing Schlag et al. 2021's delta rule as formalized in Irie et al. 2022 Eqs. 1–4).
- "Prior work frames fast-weight self-modification during autoregressive sequence processing as an instance of dynamic evaluation (Mikolov et al. 2010; Krause et al. 2018), directly supporting the use of a dynamic-evaluation-style protocol to test cross-document carry" (Methodology).
- "An ablation that reset fast-weight updates on a short fixed cadence rather than carrying them across a full episode collapsed to baseline (no-self-modification) performance, showing that breaking persistence too early erases any benefit of a carried fast weight" (Related Work / Discussion — precedent for carry-length-dependent benefit).
- "Even successful fast-weight self-modification mechanisms report only modest gains over non-adaptive baselines in multi-task settings (single-digit-point normalized score improvements), suggesting small or null effect sizes are typical for this class of method, not necessarily a sign of a broken mechanism" (Discussion — framing a modest/null result as consistent with the literature).

## Verification
- Verified directly from the PDF: Eqs. 1–4 (DeltaNet) and Eqs. 5–8 (SRWM) as printed in Sec. 2–3; the φ-for-stability sentence in Sec. 2; the state-reset ablation paragraph and its exact numbers (28.5 (1.2) and 16.1 (2.2) vs. Table 3's 35.0 (1.6)/20.0 (1.8) for SRWM and 34.6 (1.8)/15.3 (1.9) for Fake SR) in Sec. 4.3 "Ablation on State Reset"; the dynamic evaluation paragraph in Sec. 6 "Dynamic Evaluation"; the Hubinger et al. citation in Sec. 7 Conclusion/limitations paragraph.
- All numbers quoted above are read directly from Table 3 and the ablation paragraph text, not inferred.

# FILE: A Practical Survey on Faster and Lighter Transformers_importance.md

# A Practical Survey on Faster and Lighter Transformers

**Citation:** Quentin Fournier, Gaétan Marceau Caron, Daniel Aloise, 2023, arXiv:2103.14636v2 (ACM Computing Surveys format)

## Important parts for my research
- Taxonomy of efficiency methods: general (gradient checkpointing, reversible layers, parameter sharing, pruning, distillation, mixed-precision/quantization, micro-batching, MoE) vs. specialized "x-formers" (sparse attention, factorized/low-rank attention, architectural change).
- Transformer-XL (Dai et al. 2019): segment-based recurrence — FIFO memory caches hidden states of the previous window; attention reads this cache but gradients do not flow into it. Achieves 4x the relative effective context length (RECL) of vanilla Transformer at same budget, but is bounded to the FIFO window range and cannot capture dependencies beyond it — a clean architectural analogy for "state carried across chunks but not indefinitely," useful contrast to session-persistent TTT (fast weight carries indefinitely, no FIFO eviction).
- Compressive Transformer (Rae et al. 2020) extends Transformer-XL: when the FIFO memory is full, oldest representations are compressed (pooling, 1D/dilated conv, or attention-based) into a secondary compressed memory instead of being discarded, trained with an auxiliary reconstruction loss.
- Linear/kernelized attention (Katharopoulos et al. 2020 "Transformers are RNNs"): rewrites softmax attention as a kernel feature map φ, giving Attention_i = φ(Q_i)(Σφ(K_j)V_jᵀ) / φ(Q_i)(Σφ(K_j)) — an explicitly recurrent/linear form where Σφ(K_j)V_jᵀ is an additive state analogous to a fast-weight matrix updated by outer products, directly relevant to hebbian-style associative memory framing. Performer (Choromanski et al. 2021) approximates the softmax kernel via random Fourier features (orthogonal Gaussian projections) instead of elu(x)+1.
- Long-Range Arena benchmark table (Tay et al. 2021, reproduced): BigBird 55.01% avg score (best), vanilla Transformer 54.39%, Linformer/Performer/Linear Transformer much faster (up to ~8x steps/sec at 4K length) but slightly lower average scores (~50-51%) — a reusable evidence point that efficiency variants trade a few points of quality for large throughput gains.

## What cite
- (Related Work) Transformer-XL/Compressive Transformer as the standard "memory across chunks within one continuous stream" baseline against which session-persistent cross-document carry should be distinguished — these carry hidden-state cache, not model weights, and are gradient-free/frozen at read time, bounded by FIFO eviction.
- (Related Work / Methodology) Katharopoulos et al.'s linear-attention-as-RNN reformulation as the standard equation for framing fast-weight-style associative state (Σφ(K)V^T) — useful for explaining hebbian outer-product updates by analogy.
- (Discussion) Long-Range Arena scores as a citable number for efficiency/quality trade-off framing if discussing computational cost of TTT compared to standard efficient-attention alternatives.

## Verification
- All specifics (Transformer-XL RECL 4x figure, Compressive Transformer memory-full compression mechanism, linear attention equation, LRA table values) directly read from PDF pages 26 and 24-25 and Table 3 (p.28). No inferred numbers.

# FILE: A Survey on In-context Learning_importance.md

# A Survey on In-context Learning

**Citation:** Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, Baobao Chang, Xu Sun, Lei Li, Zhifang Sui, 2024, arXiv:2301.00234v6 (Peking University et al.)

## Important parts for my research
- Formal definition: ICL is training-free — the model M takes a demonstration context C (task instruction + k examples) and query x, and predicts ŷ = argmax_yj P(yj | C, x) with no backward-gradient parameter update (Eq. 1-2, following Brown et al. 2020). This is the key contrast point with TTT: ICL conditions on context in the forward pass only, whereas TTT explicitly updates weights via a self-supervised loss.
- "Gradient Descent View" (§5.2.2): Dai et al. (2023a) show a dual form between Transformer attention and gradient descent, arguing GPT-style ICL behaves like implicit fine-tuning; von Oswald et al. (2023) show linear attention with hand-constructed weights reproduces models learned by gradient descent. This is the standard citation for "attention implicitly performs a weight update," useful for framing fast-weight/TTT mechanisms as an explicit, mechanistic version of what ICL does implicitly.
- Reported finding: increasing the number of in-context demonstrations does not reliably improve performance and can be detrimental (cited under Long-context ICL, §7) — a directly analogous phenomenon to a flat/non-compounding carry, worth noting as a parallel "more context ≠ better" result in a different mechanism.
- Bayesian view: ICL as implicit Bayesian inference over a latent task concept (Xie et al. 2022) — an alternative theoretical lens distinct from the gradient-descent view.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Definitional distinction that ICL performs no parameter update (Eq. 1-2) — positions TTT/fast-weight carry as mechanistically different from and complementary to ICL.
- (Related Work / Discussion) Dai et al. (2023a) / von Oswald et al. (2023) dual-form-of-attention-as-gradient-descent results, as the established link between attention and implicit weight updates that motivates explicit fast-weight TTT.
- (Discussion — explains flat carry, by analogy) The finding that more ICL demonstrations do not monotonically help and can hurt, as a parallel non-monotonic-context phenomenon from a different literature.

## Verification
- Definition equations, gradient-descent-view claims, and the long-context ICL degradation finding were read directly from PDF pages 2 and 7-8. Paper is tangential to core research (no TTT/fast-weight/carry content); scope kept accordingly narrow.

# FILE: Attention is all you need_importance.md

# Attention Is All You Need

**Citation:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, 2017, NeurIPS (31st Conference on Neural Information Processing Systems, NIPS 2017), also arXiv:1706.03762

## Important parts for my research
- Defines the position-wise feed-forward network (FFN) sub-layer exactly as: FFN(x) = max(0, xW1 + b1)W2 + b2 (Eq. 2), described as "two convolutions with kernel size 1," applied identically and independently at each sequence position, with d_model=512 input/output and d_ff=2048 inner dimensionality in the base model — this is the two-matrix (up-projection, down-projection) MLP structure that In-Place TTT's fast weights modify (the "down-projection" is W2 here), so this is the canonical definition of the module being adapted.
- Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V (Eq. 1); multi-head attention runs h=8 parallel heads with d_k=d_v=d_model/h=64, concatenated and projected by W^O (base model config in Table 3).
- Self-attention layer complexity is O(n²·d) per layer with O(1) sequential operations and O(1) maximum path length between any two positions (Table 1) — the standard complexity citation motivating why long-context/efficient-attention and non-attention (recurrent/fast-weight) alternatives are studied at all.
- Encoder/decoder each use residual connections around every sub-layer followed by LayerNorm: LayerNorm(x + Sublayer(x)) — standard pre-existing structure any adaptation (LoRA, TTT fast weight) must be inserted around or into.
- Sinusoidal positional encoding (not relevant to fast weights directly, but establishes the base architecture has no persistent state across the sequence beyond attention over the visible context window — motivating why nothing here "carries" past a context boundary without an added mechanism.)
- Base model trained for 100K steps (12h) on 8 P100 GPUs; achieves 27.3/38.1 BLEU on EN-DE/EN-FR (Table 2); big model 28.4/41.8 BLEU. These numbers are not otherwise relevant to TTT but confirm this is the base Transformer architecture being cited for background, not a TTT-adjacent contribution.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The Transformer architecture's feed-forward sub-layer is two affine transformations around a ReLU, FFN(x) = max(0, xW1+b1)W2+b2, applied identically at every sequence position (Vaswani et al. 2017); it is the down-projection matrix W2 of this module that In-Place TTT (Feng et al. 2026) and this work's fast-weight mechanism update at inference time" (Background/Methodology — grounds what layer is being modified).
- "Self-attention layers scale quadratically in sequence length, O(n²·d) per layer (Vaswani et al. 2017), the standard motivation cited across the efficient-attention and long-context literature this proposal builds on" (Background).
- "The base Transformer architecture has no mechanism for retaining information across independently-processed inputs beyond what is explicitly re-supplied in context (Vaswani et al. 2017); persistence across document/session boundaries is not a capability of the base architecture and must be added" (Introduction/Motivation — sets up the gap the proposal addresses).

## Verification
- Verified directly from the PDF: Eq. 1 (attention formula) and Eq. 2 (FFN formula) as printed in Sec. 3.2.1 and 3.3; d_model=512, d_ff=2048, h=8, d_k=d_v=64 from Sec. 3.1/3.2.2 text and Table 3 "base" row; Table 1's O(n²·d) self-attention complexity and O(1) sequential-ops/max-path-length entries; BLEU scores 27.3/38.1 (base) and 28.4/41.8 (big) from Table 2; 100K steps/12 hours training detail from Sec. 5.2.
- No claims in this note required inference beyond what is printed in the text/tables; all are read directly.

# FILE: Big bird: Transformers for longer sequences_importance.md

# Big Bird: Transformers for Longer Sequences

**Citation:** Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed, 2020, NeurIPS 2020 (Google Research), arXiv:2007.14062v2

## Important parts for my research
- BigBird's sparse attention = random attention (r random keys per query) + sliding window (width w) + global tokens (g tokens attending to/from everything), reducing complexity from quadratic to linear in sequence length while remaining a universal approximator of sequence functions and Turing complete (given O(1) global tokens).
- Two global-token variants: ITC (internal transformer construction — designate existing tokens as global) vs ETC (extended transformer construction — add extra global tokens, effectively extra storage locations "similar to memory," per the authors' own framing).
- Ablation (Table 1, @512 length): window-only attention gets MLM bitrate 58.3 vs BERT-base's 64.2; random-only gets 60.1; random+window combined gets 62.7 — random and local attention alone are each insufficient to match full attention, motivating the combination.
- Handles sequences up to 8x longer than prior full-attention models on the same hardware; achieves SOTA/near-SOTA on HotpotQA, Natural Questions, TriviaQA, WikiHop QA and long-document summarization (Arxiv, PubMed, BigPatent) and genomics tasks (promoter-region prediction F1 99.9, chromatin-profile prediction).
- Explicit no-free-lunch result: constructs a task (finding the furthest vector by inner product, related to the Orthogonal Vector Conjecture) that any sparse-attention mechanism with Õ(n) edges needs Ω̃(n^(1-o(1))) layers to solve, while full attention solves it in O(1) layers — sparsity has a provable cost on some tasks.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) BigBird as the representative "extend usable context length via sparse/global attention within one forward pass" approach — contrasts with session-persistent TTT's strategy of carrying fast weights across independently-bounded documents rather than extending the single attended context window.
- (Related Work) The no-free-lunch/OVC lower bound result as a citable formal argument that any linear-complexity context mechanism (attention-based or fast-weight-based) trades some expressivity for efficiency, useful context for discussing why persistence-based approaches might show flat/limited gains on some tasks.

## Verification
- All figures (MLM bitrate ablation, 8x context claim, promoter F1 99.9, no-free-lunch OVC-based lower bound) read directly from PDF pages 1, 4, 6, 9. Paper is tangential to the TTT/fast-weight/carry research core — no persistence or fast-weight mechanism is discussed; scope kept accordingly limited.

# FILE: COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING_importance.md

# Compressive Transformers for Long-Range Sequence Modelling

**Citation:** Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap, 2019, arXiv:1911.05507v1 (DeepMind / UCL)

## Important parts for my research
- Architecture (Algorithm 1): extends Transformer-XL's FIFO memory (n_m slots) with a second FIFO *compressed memory* (n_cm slots). When memory is full, the oldest n_s activations are compressed via f_c: R^(ns×d) → R^(⌊ns/c⌋×d) (compression rate c) instead of being discarded — max/mean pooling, 1D/dilated conv, or "most-used" (usage-based retention, inspired by DNC garbage collection) were tested; a learned conv with an attention-reconstruction loss (Algorithm 2, reconstructing content-based attention over old memories from compressed ones, with stop-gradient into the main network) performed best (0.973 bpc on Enwik8 vs 0.984 for plain conv+BPTT).
- Directly relevant training-dynamics finding (§5.5.1, "Optimisation Schedule"): reducing the learning rate toward zero during long-context training harms both TransformerXL and Compressive Transformer performance — attributed to a distributional shift arising from *online updating of parameters while processing long contiguous articles*. The authors explicitly note: "Updating the parameters only at article boundaries (and then resetting the state) could be one solution for long-range memory models, but this would slow down learning significantly" — i.e., they identify but do not pursue document-boundary-aware state resets, leaving a gap our cross-document persistence work addresses directly.
- The model is trained WITHOUT state resetting across the corpus (memory/compressed memory persist continuously across article/document boundaries during training) — an activation-level (not weight-level) precedent for "don't reset at document boundaries." At generation time authors note the model is "slightly out of sample" when given short contexts because memory/compressed memory are normally never empty during training (always full) — a caveat about train/inference distribution mismatch that carries over to any persistence scheme.
- Cites and briefly benchmarks dynamic evaluation (Krause et al. 2019) directly: "a very naive dynamic evaluation approach of loading a model checkpoint and continuing training over one epoch of the test set" gives WikiText-103 test perplexity 16.1, compared to Krause et al.'s 16.4 using a more sophisticated dynamic-eval approach on TransformerXL — both beat static SOTA (17.1 for their own Compressive Transformer). Confirms dynamic evaluation (continuing training/updating at test time) as an established, separate methodology from static context modelling — directly the family of technique (Krause 2018/2019 dynamic eval) this project's evaluation design extends.
- Results: SOTA 0.97 bpc on Enwik8, 17.1 ppl on WikiText-103 (1.2pp better than TransformerXL, mostly from ~20% improvement on infrequent/rare words, only 2.6% on frequent words — Table 7). Introduces PG-19 (28,752 pre-1919 Gutenberg books, 11GB, avg. 69K words/doc) as a long-range book-domain LM benchmark — relevant precedent for a Books-domain long-context eval set.
- Maximum temporal range formula: TransformerXL = l×n; Compressive Transformer = l×(n_m + c·n_cm) — for c>1 the compressed store increases effective temporal range at equal attention/compute cost.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Direct precedent for "carrying state across document/article boundaries without reset" at the activation-memory level, explicitly distinguishing our contribution: this paper carries a compressed KV/activation cache (frozen, retrieved via attention, no gradient into distant memories), not a fast weight matrix updated by a hebbian/gradient-style rule.
- (Related Work / Discussion — explains flat carry / instability) Their explicit acknowledgment that boundary-aware resets vs. continuous online updates trade off learning speed against generalization, and that models are "out of sample" at short/empty-memory contexts — directly useful for discussing why carry effects might be flat, unstable, or session-length-dependent in our own setup.
- (Methodology) Confirmation that dynamic evaluation (Krause et al. 2018/2019) is an established alternative to static perplexity evaluation, with concrete comparative numbers (16.1 vs 16.4 vs 17.1 ppl on WikiText-103) — reusable as a citable precedent for the magnitude of gains dynamic/continual test-time updating can produce.
- (Methodology) PG-19 as a citable long-context book-domain benchmark if books-domain framing is needed.

## Verification
- All claims (architecture algorithm, compression functions/losses, optimisation-schedule discussion, dynamic eval numbers 16.1/16.4/17.1, bpc/ppl results, temporal range formula, PG-19 stats) read directly from PDF pages 2-9. The "out of sample" / no-state-resetting note is from Supplementary Section E (page 16). No inferred numbers.

# FILE: Catastrophic forgetting in connectionist networks_importance.md

# Catastrophic Forgetting in Connectionist Networks: Causes, Consequences and Solutions

**Citation:** French, R. M. (1999). Catastrophic Forgetting in Connectionist Networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

## Important parts for my research
- Defines catastrophic interference/forgetting: in networks with a single set of shared, overlapping (distributed) weights, learning a new pattern set can abruptly and near-totally erase previously learned associations, unlike the gradual forgetting seen in human learning (Barnes & Underwood 1959 retroactive-interference curves).
- Canonical demonstration (McCloskey & Cohen 1989): a backprop net that had perfectly learned "ones" addition facts (1+1..9+1, 1+2..1+9) then learned "twos" facts — recall on ones facts dropped from 100% to 20% within 1-5 learning trials on twos, to 1% by 5 more trials, to 0% (no correct answers) by trial 15.
- Root cause identified as the *overlap of internal distributed representations* between old and new patterns — the same property that gives distributed nets their generalization ability. This is the "stability-plasticity dilemma" (Grossberg).
- "Weight cliffs" (Box 1, citing Kolen & Pollack 1990): weight-space is not smoothly structured; even tiny changes to initial weights of a trivial 2-2-1 XOR network cause huge, discontinuous changes in convergence behavior — explains why a small new-pattern gradient step can catastrophically relocate a network's solution.
- **Fast/slow weights (Hinton & Plaut 1987, cited as ref 12):** two separate weight sets — one that changes rapidly but also decays rapidly back toward zero ("fast weights"), one that changes slowly and decays only slowly ("slow weights"); the effective weight used is a combination of both. This reduced catastrophic interference in their experiments but French notes it is "frequently cited" yet "not yet thoroughly explored," with likely storage-capacity limits — unclear if it scales to truly sequential learning beyond one or two new patterns.
- Solutions taxonomy: (1) reduce representational overlap via orthogonal/sparse coding (French's "activation sharpening," ALCOVE's localist receptive fields) — but sparsity trades away generalization/discrimination ability; (2) dual-network architectures with separate fast-storage and slow-consolidation areas (McClelland, McNaughton & O'Reilly 1995 hippocampus/neocortex account) that communicate via "pseudopatterns" (Robins 1995) rather than raw rehearsal of stored data; (3) convolution-correlation / holographic associative memories (CHARM, TODAM, Sparse Distributed Memory) that are isomorphic to Hopfield networks and degrade *gradually* rather than catastrophically — but only below a critical storage limit, beyond which retrieval collapses abruptly.
- **Box 4 — "catastrophic remembering" (Sharkey & Sharkey 1995):** the inverse failure mode — an auto-associative network trained on very many patterns eventually learns the identity function well enough that it "recognizes" (reconstructs) virtually any input, including ones never seen, because generalizing to the identity function necessarily destroys the network's ability to discriminate seen from unseen input.
- Box 3 (animal timing experiments) shows *sequential* learning of two facts (40s vs 8s peak duration) produces catastrophic loss of the earlier one, but *concurrent* learning of both produces no catastrophic loss when later required to shift back — order/interleaving of exposure changes whether interference is catastrophic or benign.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The stability-plasticity dilemma predicts that a fast-updating associative weight lacking any mechanism to protect prior state is structurally prone to either catastrophic overwriting or saturation, not smooth accumulation" (Discussion — explains flat/non-compounding carry) — French (1999).
- "Hinton & Plaut's dual fast/slow weight scheme is the earliest proposal that a rapidly-decaying fast weight combined with a slow weight can reduce interference, but its capacity limits under repeated sequential updates were never characterized" (Related Work — positions the proposal's hebbian fast-weight carry as testing exactly this open question) — French (1999), citing Hinton & Plaut (1987).
- "A purely Hebbian/auto-associative accumulator that is driven toward reproducing its own recent inputs can enter a regime of 'catastrophic remembering,' in which discrimination collapses even as reconstruction accuracy looks fine — a candidate explanation for a carry state that stops changing (plateaus at a fixed point) rather than compounding" (Discussion — explains flat carry) — French (1999), citing Sharkey & Sharkey (1995).
- "Distributed convolution-correlation memories (CHARM, TODAM, SDM) degrade gracefully rather than catastrophically only below a critical capacity, after which retrieval breaks down abruptly — an analogue for why carry stability must be checked before any gain is trusted" (Methodology — motivates stability/state-norm checks) — French (1999).

## Verification
- Verified directly from PDF text: the 100%→20%→1%→0% figures (p.2, "Measuring catastrophic interference" section), the Hinton & Plaut fast/slow weight description (p.6, "Other techniques..." section), the "weight cliffs" / Kolen & Pollack XOR claim (Box 1, p.11), and the catastrophic remembering account (Box 4, p.18).
- No numbers here are inferred — all bullets above map to explicit sentences in the extracted text. One caveat: French's paper does not use the term "hebbian" explicitly for the fast-weight mechanism it describes; the connection to this proposal's hebbian update rule is my inference/framing, not the paper's own language, and is flagged as such in the "What to cite" bullet.

# FILE: Continual Learning of Large Language Models: A Comprehensive Survey_importance.md

# Continual Learning of Large Language Models: A Comprehensive Survey

**Citation:** Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, Hao Wang, 2024, arXiv:2404.16789v3 (Rutgers University / Google Cloud AI Research)

## Important parts for my research
- Taxonomy: vertical continuity (general→specific: Continual Pre-Training CPT → Domain-Adaptive Pre-training DAP → Continual Fine-Tuning CFT) vs. horizontal continuity (continual adaptation across time/domains post-deployment, three shift types: temporal, content, language). Session-persistent TTT is neither — it is inference-time, per-session weight persistence with no gradient-based training loop, sitting outside this whole taxonomy; useful as an explicit contrast point.
- Standard CL evaluation metrics (§2.2.3), directly reusable: Overall Performance (OP, average accuracy across all tasks so far), Forgetting (F, largest performance drop per task averaged over all stages), Backward Transfer (BWT = −F; negative BWT means forgetting, positive means later tasks helped earlier ones), Forward Transfer (FWT, performance on unseen future tasks vs. a randomly initialized model). Also FUAR = Forgotten/(Updated+Acquired) ratio (§5), designed to fix OP's inability to reflect forgetting-vs-acquisition trade-off; FUAR=1 means an even trade-off, <1 means high learning efficacy relative to forgetting.
- Directly relevant finding (§6.1, "Intriguing Properties"): when LLMs are fine-tuned sequentially and *cyclically* over a series of documents, they exhibit "anticipatory recovering" — the ability to recover forgotten information on a document even *before* re-encountering it, suggesting a latent capacity for sequential memorization beyond simple forgetting curves (cites [275]). This is a citable precedent for non-monotonic, hard-to-predict memory dynamics under repeated/cyclic document exposure — directly relevant to explaining a flat (rather than growing) carry curve as a plausible, literature-consistent outcome rather than a bug.
- Catastrophic forgetting framing: LLMs show more resilience to forgetting than smaller downstream-tuned models, but this resilience is observed mainly at the representation level, not the decision/output level — i.e., near-zero forgetting on internal representations can still coexist with output-level performance degradation (§4.3.2, citing [144, 223, 260, 299]).
- Related-work catalog of parameter-efficient CL techniques relevant to methodology: LoRA-based continual learning (O-LoRA — orthogonal-subspace LoRA to minimize inter-task interference; CoLoR — trains separate LoRA modules per task and clusters at test time to infer task ID), EWC-style parameter regularization (penalizing ‖θ−θ_{t-1}‖ weighted by a Fisher-style importance matrix Σ), and episodic/external memory approaches (Larimar — Kanerva-machine-based episodic memory supporting one-shot writing/reading/forgetting without retraining; WISE — side memory system for lifelong model editing).
- §6.3 "Roles of Memory": argues the field's historical focus on storage-efficiency (small replay buffers) is becoming less relevant as storage becomes cheap; the real constraint shifting toward is compute/update efficiency — a framing point relevant to justifying inference-time fast-weight persistence (compute-bounded, not storage-bounded) as a distinct paradigm from replay-based CL.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Vertical/horizontal continuity taxonomy and CPT/DAP/CFT stages, to explicitly position session-persistent TTT as operating outside standard continual learning's training-time paradigm — no replay, no parameter regularization, updates confined to inference and (in our design) reset-free only within a session.
- (Related Work — explains flat carry) "Anticipatory recovering" phenomenon (§6.1) as a literature precedent for non-monotonic/counterintuitive memory dynamics in sequentially-exposed LLMs, useful when discussing why carry does not compound linearly.
- (Discussion) The representation-vs-output-level forgetting distinction (§4.3.2) as a citable framing for why perplexity-level flatness does not necessarily mean the fast weights are inert internally.
- (Methodology) OP/F/BWT/FWT and FUAR as standard, reusable continual-learning metrics if the paper wants to frame carry-reset comparisons in CL terminology.

## Verification
- Taxonomy, evaluation metrics (§2.2.3), FUAR definition, LoRA/EWC/Larimar/WISE descriptions, and the anticipatory-recovering finding (§6.1) were all read directly from PDF pages 2, 5, 21, 22, and 24. No inferred numbers; anticipatory recovering is attributed by this survey to a single cited primary source ([275], not independently verified here since that paper is outside the assigned set).

# FILE: Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions_importance.md

# Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions

**Citation:** Sarthak Kumar Maharana, Shambhavi Mishra, Yunbei Zhang, Shuaicheng Niu, Taki Hasan Rafi, Jihun Hamm, Marco Pedersoli, Jose Dolz, Yunhui Guo — Transactions on Machine Learning Research (TMLR), 07/2026, arXiv:2607.08164v3.

## Important parts for my research
- Formalizes CTTA as continually adapting a source model f_θS to a sequence of unlabeled, unknown-boundary target distributions {T_i}, no source data access, single forward-backward pass per batch, no model reset (θ0=θS, updated online to θt) — structurally close to our session-persistence setup (carry across bounded units without full resets), but in vision, not cross-document LM text.
- Names two central failure modes matching our null-result framing: (1) **catastrophic forgetting** of source/pretrained knowledge under continual updates, and (2) **error accumulation** from noisy self-generated pseudo-labels compounding over time. Explicitly distinguishes CTTA forgetting (losing *source* generalization) from standard continual-learning forgetting (losing performance on prior *training* tasks) — a useful terminological distinction for our own framing of "carry doesn't compound."
- Taxonomy of CTTA methods: optimization-based (entropy minimization, pseudo-labeling, topological consistency, parameter restoration), parameter-efficient (normalization-layer adaptation, adaptive layer selection), architecture-based (teacher-student, adapters, visual prompting, masked modeling).
- **Parameter restoration** (CoTTA, Wang et al. 2022): stochastically reverts a random subset of adapted weights back to source weights each step via a Bernoulli mask (θ̂^{t+1} = m⊙θ̂S + (1−m)⊙θ̂^{t+1}), explicitly to bound long-horizon drift/forgetting — directly analogous to our carry-reset baseline / a decay mechanism, and a citable prior instance of "controlled non-persistence as a stability safeguard."
- Reports that entropy-minimization gradients can form a "low-dimensional," near-degenerate structure over long adaptation horizons (LCoTTA, Duan et al. 2025) and that "reliable gradients lie in a stable low-rank subspace" while confident-but-wrong gradients leak into it (MGP, Xiong et al. 2026) — an explicit mechanism in the CV-TTA literature for why continual self-supervised adaptation can plateau/degenerate rather than keep improving, structurally analogous to a flat (non-compounding) carry signal.
- Domain-shift regime taxonomy (CSC, gradual/CCC/CGS, PTTA temporally-correlated batches, CDC recurring dynamic change, ReservoirTTA recurring/evolving patterns) is a reusable design vocabulary for describing our own document-ordering/session-boundary assumptions, though CTTA's "domains" (weather, corruption types) have no LM-text analogue in our setup.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Formal CTTA problem definition and forgetting/error-accumulation dichotomy — for positioning session-persistent TTT as CTTA's language-modeling, per-document analogue, extending it to cross-document text rather than cross-domain vision streams.
- (Discussion — explains flat/non-compounding carry) CoTTA's stochastic parameter restoration and the entropy-gradient degeneracy findings (LCoTTA, MGP) as literature precedent for adaptation signals plateauing or requiring active bounding over long horizons — supports framing a flat, non-growing carry as a known adjacent phenomenon rather than an isolated anomaly.
- (Methodology) CSC/CDC/PTTA taxonomy as a reference point when describing our dynamic-eval document-repeat protocol's assumptions (fixed order vs. random domain recurrence).

## Verification
- Formal setup (Eqs. and Table 1 comparison of standard CL vs CTTA), CoTTA restoration equations (Eqs. 6–7), and entropy-degeneracy claims (Sec 4.1.1, LCoTTA/MGP citations) checked directly against pages 1–14 of the PDF (Sections 1–4.1.4).
- Only the first ~14 of 48 pages were read (intro, problem definition, domain-shift taxonomy, and the optimization-based-methods subsection); benchmarks/experimental-results sections (later pages) were not reviewed, so no specific CTTA accuracy numbers are cited here — flagged as unread rather than inferred.

# FILE: Continual learning with deep generative replay_importance.md

# Continual Learning with Deep Generative Replay

**Citation:** Hanul Shin, Jung Kwon Lee, Jaehong Kim, Jiwon Kim — NIPS 2017 (31st Conference on Neural Information Processing Systems), arXiv:1705.08690v3.

## Important parts for my research
- Defines catastrophic forgetting per McCloskey & Cohen (1989): performance on old tasks "abruptly degrades" when a network is trained on a new task — the canonical citation for the phenomenon our flat/non-compounding carry result is adjacent to.
- Deep Generative Replay (DGR): a "scholar" model = generator (GAN) + solver. Instead of storing real past data, the generator synthesizes pseudo-inputs from past task distributions; a solver's response to a generated input becomes the pseudo-target. Loss for solver i: L = r·E_{(x,y)~Di}[L(S(x;θi),y)] + (1−r)·E_{x'~Gi-1}[L(S(x';θi), S(x';θi-1))] (Eq. 1) — training loss mixes real current-task data and replayed synthetic old-task data at ratio r.
- Key empirical result: on MNIST permutation tasks (independent, non-overlapping input distributions), a naively sequentially-trained solver "None" collapses to near-chance on old tasks, while generative replay (GR) tracks the "exact replay" (ER) upper bound almost exactly. Sequential scholar chain (Solver1→Solver5) loses only ~0.25% accuracy total (98.81%→98.56%) after 5 tasks.
- Random-noise replay ("Noise" condition — pseudo-inputs are Gaussian noise paired with old-solver responses) fails to prevent forgetting — meaningful reconstruction of the *input* distribution, not just target distribution, is necessary for retention (Sec 4.1, 4.3 disjoint-class experiment).
- Compares against EWC (elastic weight consolidation — protects important weights via Fisher-Information-weighted regularization) and LwF (Learning without Forgetting — regularizes on the network's own pre-finetuning response to new-task inputs). Both are alternatives to replay; the paper frames them as complementary, not mutually exclusive, contributions to retention.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Canonical definition/citation for catastrophic forgetting (McCloskey & Cohen, 1989, as summarized here) when introducing the concept that motivates our carry-reset baseline.
- (Discussion — explains flat/non-compounding carry) The Noise-replay negative result — that replay/retention mechanisms fail when the recreated signal doesn't capture the real underlying distribution — is a useful analogy for why a carried hebbian fast weight might plateau rather than compound: if the "memory trace" a fast weight accumulates does not faithfully reconstruct informative structure (as opposed to being a generic accumulation), stability can appear at a low, flat level instead of growing forgetting or growing benefit.
- (Related Work) EWC/LwF as classical parameter-protection alternatives to episodic/generative replay for continual learning — useful contrast when situating session-persistent TTT (a form of implicit, decay-free replay via weight carry) against the standard continual-learning toolkit.

## Verification
- All numbers (98.81%→98.56% across 5 scholars, Eq. 1 loss formulation, Noise vs GR vs ER vs None conditions) checked directly against Table 1, Eq. 1, and Sections 3.1, 4.1, 4.3 of the PDF text.
- This paper does not address TTT, fast weights, or LLMs at all — it is included purely for the catastrophic-forgetting/replay framing; no claims here should be read as language-model-specific.

# FILE: Dynamic Evaluation of Neural Sequence Models_importance.md

# Dynamic Evaluation of Neural Sequence Models

**Citation:** Ben Krause, Emmanuel Kahembwe, Iain Murray, Steve Renals (2017), arXiv:1709.07432v2 [cs.NE] (University of Edinburgh)

## Important parts for my research
- Core mechanism: test sequence x1:T is split into segments s1...sM of length n (n=5 for word-level, n=20 for char-level). Model evaluates P(si | θ_l^(i-1)), computes cross-entropy loss L(si), backprops via truncated BPTT (gradient only within si, so cost is linear in sequence length), and updates θ to θ_l^i before moving to s(i+1) — i.e., one SGD-like ML step per segment, adapting *all* model parameters (not just a fast-weight submodule) to the local distribution Pl(x) as distinct from the training-time global distribution Pg(x) = ∫P(l)Pl(x)dl.
- Final/best update rule ("RMS + RMS global prior"), eq. 8: θ_l^i ← θ_l^(i-1) − η·∇L(si)/√(MSg+ε) + λ(θg − θ_l^(i-1))⊙RMSnorm. Key pieces: (1) global decay prior λ(θg − θ_l^(i-1)) pulls adapted params back toward trained params θg every step — this is an explicit anti-drift/stability term, directly relevant to a carry-stability argument; (2) MSg = mean squared gradient computed on training batches (not test-time, unlike vanilla RMSprop) used to rescale the learning rate per-parameter; (3) RMSnorm = √MSg normalized to mean 1, clipped ≤1/λ, used to scale decay rate per-parameter (params with high natural gradient magnitude decay faster).
- Reducing update frequency (segment length 5 or 20 vs. n=1/fully truncated BPTT as in Mikolov et al. 2010, "traditional dynamic eval") measurably helps: PTB test ppl improves 76.2→75.6 just from n=1→n=5 (SGD, no other changes) (Table 1).
- Sparse dynamic evaluation (Section 6): instead of adapting all weights, introduce a learned adaptation matrix M (init to zero) applied to a subset H of hidden units: h'_t = h_t + M·h_t, d=H² adaptation params. On Hutter Prize, adapting only 500 of 2800 hidden units (250k params, ~0.5% of full dynamic eval's parameter count) still cut bits/char from 1.24 (static) to 1.13, vs. 1.08 for full dynamic eval (Table 3). This is a directly analogous precedent for a low-rank/small fast-weight module carrying adaptation state cheaply — relevant to justifying a small persistent carry vector/matrix rather than adapting full parameters.
- Section 7.3 timescale analysis (Fig. 2): on in-distribution Hutter data, the dynamic-vs-static advantage rises and *plateaus/saturates* after ~2-3k characters — advantage does not keep growing. On out-of-distribution Spanish data (Europarl), the advantage keeps *growing* through the full 10k-character window and is much larger in magnitude. This is a citable precedent for "adaptation benefit saturates within a session for in-distribution data" — relevant context for framing why a carry might plateau rather than compound indefinitely, though this paper's "plateau" is still a rising-then-flat curve, not flat from the start as in the current null result.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Dynamic evaluation adapts a full model's parameters to a local sequence distribution Pl(x) via per-segment gradient descent, distinguishing it from the global training distribution Pg(x); we adapt this local/global distributional framing to motivate why a persistent cross-document fast weight should help." (Methodology / Introduction)
- "Krause et al. (2018) show the benefit of dynamic evaluation saturates after a few thousand characters on in-distribution data but continues to grow under distribution shift, suggesting the *shape* of an adaptation-benefit curve (rising-then-plateau vs. sustained growth) is itself diagnostic of whether the adapted state is capturing genuine local structure — a lens we apply to characterize our carry's flat curve." (Discussion — explains flat/non-compounding carry, or contrasts it)
- "Krause et al.'s global decay prior — an explicit pull-back term λ(θg − θ_l) added to every adaptation step — is a precedent for regularizing fast-weight drift toward a stable anchor; the fact that dynamic evaluation requires this term to remain stable is suggestive of why an unregularized Hebbian carry could grow unboundedly or fail to compound coherently." (Discussion / Related Work — carry stability)
- "Sparse dynamic evaluation (Krause et al. 2018) demonstrates that adapting a small subset of hidden units via a low-rank matrix (0.5% of full adaptation parameters) recovers most of the benefit of full-parameter dynamic evaluation (1.13 vs 1.08 bits/char on Hutter Prize, vs 1.24 static), supporting the design choice of a small persistent fast-weight/carry vector rather than full fine-tuning." (Methodology / Related Work)
- "Dynamic evaluation (Krause et al. 2018) achieved SOTA perplexities of 51.1 (PTB) and 44.3 (WikiText-2) word-level, and 1.19 / 1.08 bits/char on text8/Hutter Prize, establishing that gradient-based test-time adaptation reliably improves perplexity within a continuous single sequence — the baseline our work extends by testing whether such gains persist *across* independently-bounded document/session boundaries rather than only within one continuous stream." (Related Work — positioning against continuous-context adaptation)

## Verification
- All numbers (51.1, 44.3, 1.19, 1.08 bits/char; Table 1-4 entries; segment lengths n=5/20; sparse dynamic eval 500 units/250k params/1.13 bits/char) were read directly from the PDF tables and text (Sections 5-7, Tables 1-4), not inferred.
- The equations (2), (4), (5), (7), (8) were transcribed directly from the paper's Section 5 text.
- The timescale claim (saturation ~2-3k chars for Hutter, continued growth for Spanish) is directly stated in Section 7.3 text, not inferred from the figure alone. Could not independently re-derive exact pixel values from Figure 2, but the qualitative characterization matches the caption and body text.

# FILE: EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS_importance.md

# Efficient Streaming Language Models with Attention Sinks

**Citation:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (2024), ICLR 2024 (MIT, Meta AI, CMU, NVIDIA)

## Important parts for my research
- Not a TTT/fast-weight paper — no parameter updates at all. StreamingLLM (KV-cache management: keep 4 initial "sink" tokens' KV + a rolling window of recent tokens' KV, evict everything else) achieves stable perplexity over 4M-token streams using a model trained with only a finite (e.g. 4K) attention window, with zero fine-tuning. Relevant as a contrasting "persistence via cached activations" solution to the same infinite-stream problem this proposal addresses via a persistent fast weight — worth explicit positioning.
- Mechanism: "attention sink" — softmax forces attention scores to sum to 1, so when a token has no strong match in context it dumps unneeded attention mass onto highly-visible (because autoregressively visible to all subsequent tokens), semantically-unimportant initial tokens. Ablation (Table 1) shows this is positional not semantic: replacing the first 4 tokens with linebreak "\n" tokens still recovers most of the effect (PPL 5.60 vs 5.40 with original tokens, vs 5158.07 with a plain 1024-window and no sink).
- Rolling KV cache reassigns positions *within the cache* rather than the original text's absolute positions when applying RoPE (e.g. cache holding tokens [0,1,2,3,6,7,8] decoding the 9th token gets positions [0..7], not the original [0,1,2,3,6,7,8,9]). This is a concrete precedent for a "carry" or persistent state that must be re-indexed/renormalized relative to its own accumulated frame rather than global document position.
- Direct methodological analog to this proposal's evaluation design: perplexity measured on the *concatenated* PG19 test set (100 books strung together into one long stream) — Figure 5 explicitly notes "perplexity fluctuations due to book transitions," i.e. they track PPL behavior specifically at document/session boundaries within a long eval stream, which is close to (though not identical to) this proposal's repeated-session replay-with-carry design.
- Table 6 ablation: increasing StreamingLLM's cache size does not consistently lower perplexity (Llama-2-7B: 9.73→9.32→9.08→9.59 as cache goes 508→1020→2044→4092) — more retained state does not monotonically help, evidence that "more persistent context/state" doesn't guarantee compounding benefit, a data point supportive of (not identical to) a flat/non-compounding carry finding.
- Pre-training with a dedicated learnable sink token (vs. vanilla) requires only that one token to stabilize streaming, and does not hurt normal zero-shot NLP benchmark accuracy (Table 4, ARC/HellaSwag/LAMBADA/OBQA/PIQA/WG all ~within noise of vanilla) — precedent that architectural persistence mechanisms can be "free" (no capability cost) even when they add little in other numbers.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Streaming approaches to unbounded context (Xiao et al. 2024) achieve stability by caching a small fixed set of activations (sink + recent-window KV states) rather than updating any parameters; our work instead tests whether persistence via an *updated fast weight* — not cached activations — transfers benefit across document boundaries, a mechanism Transformer-XL-style and StreamingLLM-style caching do not attempt." (Related Work — positioning against activation-carrying approaches)
- "Xiao et al. (2024) evaluate long-context stability on a concatenated multi-document stream (PG19, 100 books) and note perplexity fluctuations specifically at document transitions, a precedent for our protocol of tracking perplexity across document/session boundaries under a persistent carry." (Methodology)
- "Even with a well-motivated persistence mechanism, increasing the amount of retained state does not monotonically reduce perplexity (Xiao et al. 2024, Table 6): StreamingLLM's own cache-size ablation shows non-monotonic perplexity as cache size grows, cautioning against assuming more persistent state trivially compounds into better performance." (Discussion — explains flat/non-compounding carry)

## Verification
- All specific numbers (PPL 5641/5158/5.43/5.40 in Fig.1; Table 1: 5158.07/5.40/5.60; Table 4 zero-shot accuracies; Table 6 cache-size PPLs 9.73/9.32/9.08/9.59; 4 initial tokens as sufficient threshold; 22.2x speedup) were read directly from the figures/tables/text in the PDF (pages 2, 5, 7, 9).
- The rolling-cache repositioning example (cache [0,1,2,3,6,7,8] → positions [0..7] when decoding 9th token) is transcribed directly from Section 3.2 text.
- The "book transitions" perplexity-fluctuation claim is from the Figure 5 caption text directly, not inferred from the plot.

# FILE: EFFICIENTLY MODELING LONG SEQUENCES WITH STRUCTURED STATE SPACES_importance.md

# Efficiently Modeling Long Sequences with Structured State Spaces

**Citation:** Albert Gu, Karan Goel, Christopher Ré (2022), arXiv:2111.00396v3 [cs.LG], Stanford University (published version: S4)

## Important parts for my research
- Core recurrence (Section 2.3, eq. 3): discretized SSM state update x_k = Ā·x_{k-1} + B̄·u_k, y_k = C̄·x_k, where x_k ∈ R^N is a fixed-size "hidden state"/carry updated at every step by a linear transition matrix Ā, analogous structurally to a persistent carry vector updated across steps (here within one sequence, not across document boundaries) — a useful mechanical analogy for describing how a carry accumulates (or fails to).
- HiPPO matrix A (eq. 2, "HiPPO-LegS") is a specifically-derived, structured transition matrix proven to give provably-optimal continuous-time compression of input history into a fixed-size state — this is a rare example of a *theoretically justified* (not learned-from-scratch) recurrence structure for what to retain/decay in an accumulating state, contrasting with an unconstrained/pure-Hebbian outer-product accumulator that has no such guarantee.
- Direct ablation (Section 4.4, Fig. 3, Fig. 4a): random-matrix-initialized SSM recurrences perform far worse than HiPPO-initialized ones — over 15% validation accuracy gap on sequential CIFAR-10 persists even after training both from their respective initializations. Randomly-initialized NPLR matrices (Fig. 4a) "do not perform well," confirming gains come from the specific HiPPO structure, not just "having a recurrent/adaptable state." This is direct evidence that the *shape* of the update/decay rule governing a persistent state — not merely its presence — determines whether accumulated state is useful.
- Naive/basic SSMs "perform very poorly in practice" attributed partly to vanishing/exploding gradients in the linear recurrence (Section 2.2); the LSSL (a precursor with the same HiPPO matrix, unconstrained) is reported as numerically unstable and computationally infeasible because A is highly non-normal, requiring S4's specific low-rank-plus-normal decomposition to diagonalize stably. This documents that naively-parameterized persistent linear state easily becomes numerically unstable — relevant precedent for why a carry state needs explicit stability treatment/checks before its "improvement" is trusted.
- Table 8 (WikiText-103 language modeling, test perplexity): a Hebbian-associative baseline, "LSTM + Hebb.", achieves only 29.2 ppl, clearly worse than Transformer (20.51), S4 (20.95), TrellisNet (29.19, comparable), Dynamic Conv. (25.0), and TaLK Conv. (23.3) — i.e. among the weakest of the listed baselines on this benchmark.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Prior structured state-space work shows that whether a persistent recurrent state is useful depends critically on the specific structure of its update/decay rule, not merely on having an adaptable state: randomly-initialized state transition matrices underperform HiPPO-structured ones by over 15 points even after training on the same task (Gu et al. 2022) — consistent with our finding that an unconstrained Hebbian carry accumulator, lacking any such structured decay or error-correction, fails to compound useful information." (Discussion — explains flat/non-compounding carry)
- "A Hebbian-augmented LSTM baseline reported in Gu et al. (2022) achieves only 29.2 test perplexity on WikiText-103, underperforming both a vanilla Transformer (20.51) and structured state-space model (20.95), suggesting pure Hebbian association is a comparatively weak mechanism for capturing exploitable sequence structure in language modeling relative to gradient/attention-based alternatives." (Related Work / Discussion — Hebbian-specific ammunition)
- "Naively parameterized persistent linear state (e.g. the LSSL, which uses the same HiPPO matrix as S4 without the stabilizing reparameterization) is numerically unstable in practice, motivating our practice of checking carry stability/state norms before attributing any perplexity gain to persistence (cf. Gu et al. 2022, Section 3.1)." (Methodology — carry stability check justification)

## Verification
- All specific numbers (15% validation gap in Fig. 3/4a discussion, WikiText-103 Table 8 values: Transformer 20.51, S4 20.95/48K tok/s, LSTM+Hebb. 29.2, TrellisNet 29.19, Dynamic Conv. 25.0, TaLK Conv. 23.3, GLU CNN 37.2, AWD-QRNN 33.0) were read directly from Table 8 and Section 4.4 text on pages 9-11 of the PDF.
- Equations (1)-(3) and the HiPPO matrix (eq. 2) were transcribed directly from Sections 2.1-2.3.
- Did not read the appendices (pages 13-32, proofs and extended results) — the note above is based on the main text (Sections 1-5), which is sufficient for this paper's tangential relevance (it is not a TTT/fast-weight paper; relevance is via the recurrent-state/stability analogy and the Hebbian-baseline data point). No claim above required appendix content.

# FILE: Efficient Test-Time Adaptation of Vision-Language Models_importance.md

# Efficient Test-Time Adaptation of Vision-Language Models

**Citation:** Adilek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, Eric Xing — CVPR 2024, arXiv:2403.18293v1.

## Important parts for my research
- Proposes TDA (Training-free dynamic Adapter): a non-parametric, backprop-free test-time adaptation method for CLIP using two key-value caches (positive and negative) built online from a stream of test samples, instead of gradient-based prompt tuning (TPT/DiffTPT). Positive cache stores {test features as keys, pseudo-labels as values}; negative cache stores samples whose prediction entropy falls in a moderate uncertainty band [τ_l, τ_h], with negative pseudo-labels marking classes the model is confident the sample does *not* belong to (Eq. 4: L̂_n = −1[p_l < P(Q_n)]).
- Cache update is a bounded, priority-queue-like mechanism: each class has a fixed shot capacity k; new entries are only added/replace existing ones if they have lower prediction entropy than the current "most uncertain" cached entry for that class — an explicit anti-noise-accumulation design (Sec 3.2), directly analogous to needing bounded/decaying carry to avoid degenerate accumulation.
- Final prediction combines three additive terms: CLIP zero-shot logits + positive-cache adaptation function + negative-cache adaptation function (Eq. 7), each cache contribution computed via the Tip-Adapter-style adaptation function A(z)=α·exp(−β(1−z)).
- Results: TDA improves OOD average accuracy over CLIP-ResNet-50 baseline by +3.54pp (43.09→46.63) and cross-domain average by +4.4pp (56.63→61.03), while cutting test-time adaptation latency on ImageNet from 12h50min (TPT) / 34h45min (DiffTPT) to 16 minutes — near-parity accuracy gains at orders-of-magnitude lower compute than gradient-based TTT (TPT/DiffTPT), since no backpropagation occurs at test time.
- Ablation: shot capacity is a sensitive hyperparameter — larger negative-cache capacity (>2–3 shots) *degrades* accuracy because it accumulates higher-entropy, noisier pseudo-labels over time (Fig. 4) — a direct empirical instance of an adaptation cache getting worse, not better, with more accumulated "memory," from noise rather than distribution shift.
- Both positive and negative caches independently outperform the CLIP baseline, and combining them outperforms either alone (Sec 4.3 ablation) — supports a "multiple complementary signals beat a single one" methodological point, though this is a classification (not perplexity) task.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) TDA as a non-gradient, cache-based alternative to gradient-descent TTT (TPT/DiffTPT) for test-time adaptation — useful contrast point for positioning our gradient-based fast-weight carry against non-parametric adaptation mechanisms in the broader TTA literature.
- (Discussion — explains flat/non-compounding carry) The shot-capacity ablation showing accuracy degrades once cache capacity exceeds 2–3 due to noise accumulation (Fig. 4, "the performance decline with larger caches is mainly due to noise accumulation rather than class imbalance") — a directly citable adjacent-domain result that more accumulated test-time adaptation state is not monotonically better, supporting our framing of a flat/plateauing carry as a legitimate, literature-consistent finding rather than a bug.
- (Methodology) Entropy-thresholded, capacity-bounded update rule (only replace cache entries when new entropy is lower) as a reusable pattern for describing principled ways to bound accumulation in an online adaptation setting.

## Verification
- Numbers (43.09→46.63 OOD average, 56.63→61.03 cross-domain average, 12h50m/34h45m/16min testing times, shot capacity 3/2 optimal) checked directly against Tables 1–3, Table 2, and Figure 4 in the PDF.
- This is a vision-language classification (CLIP) paper with no perplexity/LM/fast-weight-persistence content; relevance here is strictly as an analogous-domain precedent for bounded-cache/noise-accumulation dynamics, not as an LM-adjacent method.

# FILE: End-to-End Test-Time Training for Long Context_importance.md

# End-to-End Test-Time Training for Long Context

**Citation:** Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun — arXiv:2512.23675v2 [cs.LG], 31 Dec 2025 (Astera Institute, NVIDIA, Stanford, UC Berkeley, UC San Diego).

## Important parts for my research
- TTT-E2E: fast weights (inner loop, W) updated via gradient descent on next-token prediction loss at test time; initialization W0 (outer loop, θ) meta-learned end-to-end so training loss matches test loss — resolves the TTT-naive/dynamic-eval mismatch where W0 is optimized for out-of-the-box loss, not post-TTT loss (Eq 1–6).
- Mini-batch TTT: gradient step every b=1K tokens, sliding-window attention k=8K (must have k≥b so local context bridges each batch before a TTT step). Only updates MLP layers, only last 1/4 of transformer blocks, and adds a second static MLP per updated block as "safe storage" to prevent forgetting pretrained knowledge — three implementation details load-bearing per ablations.
- **Critical for our gap framing**: fast weights RESET at every document boundary — the paper explicitly discards all training documents shorter than 8K "to avoid resetting the updated MLPs across document boundaries when different documents are packed into the same training sequence, since resetting slows down training" (footnote 4). Confirms no cross-document persistence is attempted or evaluated anywhere in this paper.
- TTT-KVB connection: when the fast-weight model g is linear and gradient is w.r.t. W_{t-1}, TTT-KVB reduces to DeltaNet; w.r.t. fixed W0 it reduces to linear attention; with a non-parametric kernel estimator it recovers self-attention (footnote 1, Sec 2.4.1) — a clean unifying equation-in-words for hebbian-vs-delta positioning.
- Number-of-layers-updated ablation: updating only 1 or 3 layers fails to scale with context length like full attention; updating 6 or 12 does, with 12 no better than 6 — larger fast-weight "state" is what enables context scaling, not more compute per se.
- On NIAH (RULER pass-key/UUID tasks), full attention dramatically outperforms all TTT/RNN methods including TTT-E2E — supports interpreting TTT as compression (lossy, loses irrelevant detail) rather than lossless retrieval memory.
- Ablation: TTT-E2E improves loss by 0.018 over full attention even when window k=8K equals full-attention context (no architecture difference), showing the gain is from learning-to-learn, not architecture.
- Scaling: TTT-E2E tracks full-attention's compute-scaling trend (regime boundary ~760M params / ~48B tokens) while Mamba2/Gated DeltaNet do not; 2.7× faster prefill latency than full attention at 128K context, but 3.4× slower training latency at 8K (1.2× faster at 128K) — training cost is a stated limitation.
- Pretraining: DCLM (filtered Common Crawl, 3.8T tokens, docs <8K discarded); long-context fine-tuning/eval: Books (not SlimPajama-6B, though SlimPajama and FineWebEdu are tested anecdotally for tokenizer/data-quality sensitivity).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Research Gap) "we discard all documents shorter than 8K to avoid resetting the updated MLPs across document boundaries... since resetting slows down training" — direct textual evidence that In-Place-TTT-style fast weights reset at document boundaries and cross-document persistence is unaddressed.
- (Related Work) TTT-E2E's meta-learned (E2E) initialization vs. dynamic evaluation (Krause 2018/2019, non-E2E) — useful contrast for framing our carry mechanism as building on non-meta-learned dynamic eval rather than E2E meta-learning.
- (Methodology) KV-Binding unifying equation: linear model → DeltaNet under W_{t-1} gradient, → linear attention under fixed-W0 gradient (footnote 1, Sec 2.4.1) — for explaining hebbian (fixed-init, no error correction) vs. delta-rule terminology precisely.
- (Discussion) NIAH: full attention "dramatically outperforms" TTT-E2E on pass-key/UUID retrieval — supports framing compression-based fast weights as fundamentally non-lossless, relevant to explaining why RULER shows no improvement in our own results.
- (Discussion) Two-MLP "safe storage" design and last-1/4-layers restriction as explicit anti-forgetting engineering choices — parallel to our need to explain stability/forgetting behavior in a persistent-carry setting.

## Verification
- All numbers (2.7×, 3.4×, 1.2×, 0.018 loss delta, 760M/48B regime boundary, k=8K/b=1K, last 1/4 layers, 88M vs 18M hidden state) checked directly against Figures 1/4/5/8, Table 1, and body text on pages 1–17.
- Footnote 4 (document-boundary reset) read directly, not inferred; this is the single most load-bearing fact for our Related Work section and was verified against the exact PDF wording.

# FILE: FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness_importance.md

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Citation:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, Stanford University / University at Buffalo, arXiv:2205.14135v2 [cs.LG], 23 Jun 2022.

## Important parts for my research
- Not about fast weights, TTT, or carry state — this is a systems/efficiency paper for exact softmax attention. Low direct relevance; useful only as infrastructure/methodology citation.
- Core idea: standard attention materializes the full N×N score matrix `S=QK^T` and `P=softmax(S)` in HBM (slow GPU memory), which is the actual bottleneck, not FLOPs. FlashAttention uses **tiling** (block-wise softmax via running max/sum statistics, "algebraic aggregation") and **recomputation** in the backward pass to avoid ever materializing the N×N matrix.
- IO complexity: standard attention requires Θ(Nd + N²) HBM accesses; FlashAttention requires Θ(N²d²M⁻¹) where M is SRAM size — proven asymptotically optimal (matching lower bound) across SRAM sizes.
- Memory: FlashAttention uses O(N) additional memory (linear in sequence length) vs O(N²) for standard attention — up to 20× more memory-efficient than exact-attention baselines.
- Reported speedups: 15% faster BERT-large training vs MLPerf 1.1 record; up to 3× faster GPT-2 training vs HuggingFace (1.8× vs Megatron-LM); 2.4× speedup on Long-Range Arena (seq len 1K–4K).
- Enables longer context cheaply: GPT-2 trained with 4K context via FlashAttention is still 30% faster than Megatron-LM's GPT-2 at 1K context, while achieving 0.7 better perplexity — i.e., longer context at lower wall-clock cost, not just parity.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "We use FlashAttention (Dao et al., 2022) for the transformer backbone's attention layers to make long-context / multi-document training and evaluation runs computationally tractable at the sequence lengths this study requires." (Methodology / Implementation)
- "Longer effective context enabled by IO-aware exact attention has been shown to directly improve language modeling perplexity (0.7 PPL improvement on GPT-2 with 4× context), independent of any fast-weight or recurrent-state mechanism — a confound worth controlling for when attributing perplexity gains to a persistent carry rather than to context length alone." (Methodology — potential confound to rule out)

## Verification
- All numbers (15% BERT speedup, 3×/1.8× GPT-2 speedup, 2.4× LRA speedup, 0.7 PPL gain at 4× context, 20× memory efficiency, IO complexity Θ(N²d²M⁻¹) vs Θ(Nd+N²)) were read directly from the abstract, Table 1, Table 2, Table 4, and Theorem 2 in the PDF text.
- Did not read remaining pages (backward-pass appendix, extended benchmarks) — not needed given this paper's low relevance to the proposal's core mechanism; nothing beyond page 10 was used in this note.

# FILE: Fine-Tuning Pre-trained Transformers into Decaying Fast Weights_importance.md

# Fine-Tuning Pre-trained Transformers into Decaying Fast Weights

**Citation:** Huanru Henry Mao (Jenni), arXiv:2210.04243v1 [cs.LG], 9 Oct 2022 (EMNLP-style formatting, cites as ACL workshop-style refs; venue not stated beyond arXiv).

## Important parts for my research
- Frames kernel-linear-attention recurrent state as a **fast weight**: `S_t ∈ R^{d×m}`, updated by outer product `v_t φ(k_t)^T`, matching Schmidhuber (1992) fast-weight framing — directly analogous to a Hebbian outer-product carry.
- **Additive rule (pure Hebbian, no forgetting)**: `S_t = S_t-1 + v_t φ(k_t)^T` — explicitly stated to be "unable to forget past memories," which "can overwhelm the fixed state capacity and lead to poor performance." This is the closest published analogue to the proposal's pure-Hebbian carry.
- **Gated rule** (Peng et al. 2021): scalar gate `g_t = σ(W_g x_t)` interpolates old/new state uniformly across all elements — paper calls this "problematic as it overwrites all state elements equally without fine-grained control."
- **Delta rule** (Schlag et al. 2021a, the same delta rule referenced in the proposal as "error-correcting"): `S_t = S_t-1 − g_t S_t-1 φ'(k_t)φ'(k_t)^T + g_t v_t φ'(k_t)^T` — removes the old value associated with the current key before writing the new one (true error-correction), requires sum-normalized features and a very small initial gate (`g_t ≈ 0.007`) or training diverges.
- **Their proposed decay rule**: `S_t = G_t ⊗ S_t-1 + v_t φ(k_t)^T` with a *learned low-rank matrix* gate `G_t = σ(W_z x_t+b_z)σ(W_f x_t+b_f)^T` (element-wise, per-state-cell decay, not a single scalar) — sigmoid bounds values for "stable linear recurrent dynamics," citing Miller & Hardt (2019).
- **Key empirical finding (Table 1, best configs, PPL on The Pile val, GPT-2 fine-tuning, state size m=4):** delta rule performs *worst* of all four rules tested (20.1 PPL); the pure additive/Hebbian rule (20.0) actually *outperforms* the delta rule under its best configuration — attributed to delta's "lack of stable recurrent dynamics… during the fine-tuning setting" (citing Chen 2017). Gated (17.1) and decay (17.1) both beat additive and delta.
- Scaling state capacity (Table 2, m=32, no normalization/feature maps): decay rule reaches 14.6 PPL vs GPT-2 dense-attention baseline of 14.5 — "recovers 99% of GPT-2's performance," clearly beating gated (16.4) at the same m.
- Delta rule is fragile: needs sum-normalization of features and near-zero gate init to avoid divergence; without these tricks "the model diverges." Additive/gated rules diverge under different ablation combinations too (marked `*` in Table 1), showing all these fast-weight update rules are sensitive to normalization/init choices.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A pure additive (Hebbian) outer-product fast-weight update is known to have unbounded/uncontrolled state growth and can be overwhelmed by fixed capacity, providing no mechanism to forget or reweight prior contributions (Mao, 2022)." (Discussion — explains flat/non-compounding carry)
- "Prior work fine-tuning attention into recurrent fast-weight states found the error-correcting delta rule (Schlag et al., 2021) to be *more* unstable and worse-performing than a plain additive/Hebbian update in a fine-tuning setting, contrary to the intuition that error-correction should always help (Mao, 2022)." (Related Work / Discussion — nuances the delta-vs-Hebbian comparison)
- "Gated and decay-based fast-weight update rules consistently outperform both plain additive and delta-rule updates for recurrent-state language modeling, suggesting that some form of controlled forgetting, not error-correction alone, is what stabilizes fast-weight carries over long sequences (Mao, 2022)." (Related Work — positions the proposal's hebbian-only, no-decay carry as a known-weaker configuration)

## Verification
- All PPL numbers (Table 1: Add 20.0/Delta 20.1/Gated 17.1/Decay 17.1 under φ✗ ablation; Table 2: Decay m=32 = 14.6 vs GPT-2 14.5) and equations (Eq. 3 additive, Eq. 5 gated, Eq. 6 delta, Eq. 7 decay) were read directly from the PDF text, not inferred.
- The claim that additive "outperforms" delta is a direct quote/paraphrase from Section 4.1.3 ("Surprisingly, under its best configuration, the simple additive rule outperforms the delta rule"), confirmed in text.
- Delta rule's fragility (sum-normalization + g_t≈0.007 init requirement) is from footnote 3 in Section 4, confirmed directly.

# FILE: GATED DELTA NETWORKS: IMPROVING MAMBA2 WITH DELTA RULE_importance.md

# Gated Delta Networks: Improving Mamba2 with Delta Rule

**Citation:** Songlin Yang (MIT CSAIL), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA), ICLR 2025, arXiv:2412.06464v3.

## Important parts for my research
- Central claim: gating (uniform state decay) and the delta rule (targeted, per-key erase-then-write) are **complementary**, not substitutes — gating enables rapid/global memory erasure, delta rule enables precise/local memory modification. Combining them (gated delta rule) beats either alone.
- **Mamba2 update (pure gating, Hebbian-like write)**: `S_t = α_t S_t-1 + v_t k_t^T`, scalar data-dependent decay `α_t ∈ (0,1)` uniformly decays *all* key-value associations regardless of relevance — "does not account for the varying importance of different key-value associations."
- **DeltaNet / delta rule (no gate)**: `S_t = S_t-1(I − β_t k_t k_t^T) + β_t v_t k_t^T`, writing strength `β_t ∈ (0,1)`. Explicitly decomposed as removing the old value bound to key `k_t` (`v_t^old = S_t-1 k_t`) before writing the new blended value — true error-correction, one key at a time.
- **Gated delta rule (this paper)**: `S_t = S_t-1(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T` (Eq. 10) — α_t provides global controlled forgetting, β_t/delta term provides targeted key-specific correction; setting `α_t→0` clears memory, `α_t→1` reduces to pure delta rule.
- **Delta rule = one-step test-time SGD**: paper explicitly shows the delta update is exactly `S_t+1 = S_t − β_t∇L(S_t)` for online regression loss `L(S_t) = ½||S_t k_t − v_t||²`, with β_t as the adaptive learning rate — directly frames delta rule as TTT (cites Sun et al. 2024a "Learning to Learn at Test Time" and Titans/Behrouz et al. 2024) and contrasts with Hebbian/linear-attention updates which have **no such loss-minimization interpretation** (LA/Hebbian just uses `-⟨S_t k_t, v_t⟩`, an unregularized objective with no error term, per their Table 1).
- **DeltaNet's known weakness (their own citation of Yang et al. 2024b)**: "DeltaNet received limited attention... lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches" and "has been found to perform moderately on real-world tasks... likely due to the absence of a robust memory-clearing mechanism" — i.e., pure delta/error-correction alone is NOT sufficient without a forgetting/gating mechanism either.
- **S-NIAH case study (RULER suite, Hsieh et al. 2024, Table 2, 1.3B models)** — three clean findings directly relevant to explaining update-rule behavior:
  - "Decay hurts memory retention": DeltaNet near-perfect on S-NIAH-1 across all lengths (97.4–99.0% at 1K–8K); Mamba2 (pure gating, no delta) degrades sharply beyond 2K (65.4%→30.4% at 4K/8K); Gated DeltaNet degrades less severely than Mamba2 (94.1%/91.8% at 4K/8K) "thanks to the use of delta rule."
  - "Gating facilitates filtering": on real-world-essay S-NIAH-2/3, DeltaNet's accuracy drops significantly at longer sequences from "poor memory clearance" (memory collision — info becomes superimposed/indistinguishable); Mamba2 and Gated DeltaNet hold up better via gating.
  - "Delta rule helps memorization": on S-NIAH-3 (UUID-valued needles, harder pattern), Mamba2 drops quickly while Gated DeltaNet performs better, "verifying that the delta rule indeed has better memorization ability."
- Related Work states plainly: "The delta learning rule demonstrates superior memory capacity compared to Hebbian learning (Gardner, 1988; Prados & Kak, 1989)... this memory capacity advantage is evident in synthetic in-context learning tasks and extends to language modeling" — but immediately qualifies: "Despite these significant advantages, the delta rule faces theoretical limitations (Irie et al., 2023) and shows only moderate performance on real-world datasets (Yang et al., 2024b)."
- Benchmarks (1.3B params, 100B tokens, FineWeb-Edu, Table 3): Wiki PPL — DeltaNet 17.71, Mamba2 16.56, Gated DeltaNet 16.42 (best pure-recurrent model); LAMBADA PPL — DeltaNet 16.88, Gated DeltaNet 12.17.
- Real-world retrieval (Table 4) shows a *much smaller* margin between DeltaNet/Mamba2/Gated DeltaNet than the synthetic S-NIAH gap — attributed to repetition errors from instruction-unaligned small LMs dominating error, a confound "largely independent of the update rule choice."

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Gating (global, uniform state decay) and the delta rule (targeted, per-key error-correcting update) are complementary rather than interchangeable memory-control mechanisms; a pure delta-rule (error-correcting) carry with no decay/gating has been shown to suffer from an inability to rapidly clear outdated information across context switches, performing only moderately on real-world tasks despite strong synthetic-recall performance (Yang et al., 2025)." (Related Work / Discussion — motivates why our pure-Hebbian, no-gate, no-delta carry is even further from the state of the art, and clarifies delta rule alone is not a silver bullet either)
- "The delta update rule can be derived exactly as one step of test-time stochastic gradient descent on an online regression objective, making it a direct error-correcting analogue of a Hebbian outer-product update, which instead optimizes an unregularized similarity objective with no error term (Yang et al., 2025)." (Methodology/Discussion — grounds why the proposal frames its hebbian carry as lacking error-correction, citing this paper's explicit SGD derivation)
- "A pure gated (decay-only) update rule with no delta term degrades sharply at longer effective context and retrieval loads (S-NIAH accuracy 65.4%→30.4% from 4K to 8K for Mamba2), while the corresponding gate-free delta rule (DeltaNet) instead suffers from memory collision at longer real-world contexts absent any forgetting mechanism — suggesting neither decay alone nor error-correction alone is sufficient for stable long-horizon associative memory (Yang et al., 2025)." (Discussion — supports characterizing why simple carry mechanisms plateau or fail to compound)

## Verification
- All equations (Eq. 1–10), the S-NIAH Table 2 numbers (DeltaNet 97.4/96.8/99.0/98.8; Mamba2 99.2/98.8/65.4/30.4; Gated DeltaNet 98.4/88.4/91.4/91.8 for S-NIAH-1), Table 3 PPL numbers (DeltaNet 17.71/16.88, Mamba2 16.56, Gated DeltaNet 16.42/12.17), and the online-learning-objective framing (Table 1, one-step SGD derivation) were read directly from the PDF text on pages 1–8.
- The Related Work quotes on delta rule's "superior memory capacity" vs Hebbian and its "theoretical limitations... moderate performance on real-world datasets" are direct paraphrases/quotes from Section 5 (page 9–10), confirmed in text.
- Did not verify pages 12–22 (appendix/proofs/full reference list) — not needed for this note's scope.

# FILE: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints_importance.md

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Citation:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai — Google Research, arXiv:2305.13245v3, 23 Dec 2023.

## Important parts for my research
- Grouped-query attention (GQA): divides H query heads into G groups, each group sharing one key/value head; GQA-1 = multi-query attention (MQA), GQA-H = standard multi-head attention (MHA). GQA-8 is chosen as the favorable middle ground in their experiments.
- Uptraining recipe: convert an existing MHA checkpoint to MQA/GQA by mean-pooling the K/V projection matrices of each group (works better than selecting one head or random re-init, Fig. 4), then continue pretraining for a small fraction α (5%) of original pretraining compute to let the model adapt.
- Results (T5-XXL, summarization/QA/translation avg): MHA-XXL 47.2, uptrained MQA-XXL 46.6 at 6.3× lower inference latency (1.51s→0.24s/sample), uptrained GQA-8-XXL 47.1 (near-MHA quality) at 0.28s/sample — GQA closes most of the MQA quality gap at nearly MQA speed.
- MQA trained from scratch is unstable (frequent pretraining loss spikes, divergence on fine-tuning with long inputs); uptrained MQA is more stable but still high-variance; uptrained GQA is stable (Appendix A) — a concrete instability finding tied to reducing KV-head capacity too aggressively.
- Motivation is purely inference-time KV-cache memory bandwidth reduction (autoregressive decoding bottleneck), independent of TTT/fast-weights/continual learning.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology / architecture description) GQA-8 or GQA specifics if our experiments use a Qwen3 model variant with grouped-query attention — needed only to accurately describe the base architecture's attention mechanism, not for any TTT-related claim.
- (Related Work, minor) Uptraining-from-checkpoint technique (5% compute to adapt an existing checkpoint to a new attention structure) as a general precedent for cheap architectural adaptation via continued pretraining, if relevant to describing how a base Qwen3 checkpoint's attention differs from full multi-head.

## Verification
- Table 1 numbers (47.2/46.6/47.1 average, 1.51s/0.24s/0.28s latency) and the uptraining/mean-pooling procedure checked directly against Table 1, Figure 3, Figure 4, and Section 2.1 of the PDF.

## Note on relevance
This paper is an architecture-efficiency paper (attention head grouping for KV-cache reduction) with no direct connection to TTT, fast weights, hebbian/delta rules, carry, catastrophic forgetting, or dynamic evaluation. Its only relevance to this research is if the base model used (Qwen3) employs GQA, in which case a citation may be needed purely for architectural accuracy in a Methods section. No mechanism, number, or finding here bears on the research questions (Q1/Q2) or the null-result framing.

# FILE: Gated Linear Attention Transformers with Hardware-Efficient Training_importance.md

# Gated Linear Attention Transformers with Hardware-Efficient Training

**Citation:** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim (MIT / MIT-IBM Watson AI Lab), ICML 2024 (PMLR 235), arXiv:2312.06635v6.

## Important parts for my research
- Explicitly frames linear-attention recurrent states as **fast weights**: "This type of model with matrix-valued hidden states that change over time is also known as fast weights (Hinton & Plaut, 1987; Schmidhuber, 1992; Ba et al., 2016)," citing the connection to Transformers via Schlag et al. 2021, Irie et al. 2021, Mao 2022 — same lineage the proposal's carry sits in.
- **Baseline linear attention recurrence (pure additive/Hebbian, unnormalized, no gate)**: `S_t = S_t-1 + k_t^T v_t`, `o_t = q_t S_t` (Eq. 1) — structurally identical to the proposal's Hebbian outer-product carry, with no decay or error-correction term.
- **Gated Linear Attention (GLA) recurrence**: 2D forget gate `G_t ∈ (0,1)^{d_k×d_v}` varying over time: `S_t = G_t ⊙ S_t-1 + k_t^T v_t` (Hadamard/elementwise product, not scalar). Practical parameterization uses a low-rank outer-product gate `G_t = α_t^T 1`, giving `S_t = Diag(α_t) ⊙ S_t-1 + k_t^T v_t` (Eq. 3) — balances parameter-efficiency, state size, and hardware efficiency (avoids the full d·d_k·d_v gate matrix, which would need a materialized-per-step update, blocking tensor-core matmuls).
- **Table 1 taxonomy** of gating parameterizations across Mamba, Mamba-2, mLSTM, Gated Retention, DFW, GateLoop, HGRN-2, RWKV-6, and GLA itself — a ready-made related-work map of exactly how each model instantiates its forget gate `G_t`, useful for positioning the proposal's update rule within the landscape.
- **Direct, quantified evidence that removing gating badly hurts a Hebbian-style linear-attention carry (Table 4, 340M model, 7B tokens, training PPL, ablation)**: "No gate (i.e. Linear Attention)" = 23.21 PPL — dramatically worse than: data-independent scalar decay (RetNet-style) = 16.55, data-dependent scalar gate = 15.56, and full fine-grained GLA (4 heads) = 14.77. Pure ungated Hebbian accumulation is ~57% worse in PPL than the fully gated variant, and clearly worse than even a simple scalar decay.
- Ablation also shows granularity matters: increasing to 8 heads (finer per-head, smaller head dim) *degrades* PPL; reducing to 1 head (largest head dim) gives the best PPL (14.61) but at much higher GPU memory — "a finer-grained gating mechanism is still necessary" beyond scalar gating, but there's a granularity/memory tradeoff.
- States explicitly (citing Hochreiter & Schmidhuber 1997; Cho et al. 2014; van der Westhuizen & Lasenby 2018): "The lack of a decay term makes it difficult for a model to 'forget' information, and has been hypothesized to be partially responsible for the instability of linear attention in long-context tasks (Buckman & Gelada, 2024)" — direct citable mechanism for why an un-decayed Hebbian carry could misbehave over long/repeated exposure.
- Main LM results (Table 2, 1.3B params/100B tokens, SlimPajama, Mistral tokenizer): GLA Wiki PPL 17.22, LMB PPL 14.47 — comparable to Transformer++ (16.85/13.44) and Mamba (17.06/13.89), all clearly beating a hypothetical ungated linear-attention baseline (implied ~23 PPL range per the ablation).
- Length extrapolation (Fig. 5, PG19 test set, 1.3B models trained on 2K context): GLA extrapolates better than Mamba/RetNet in most position buckets beyond training length; Mamba "struggles to extrapolate beyond 4K."
- Recall-intensive real tasks (Table 3: FDA, SWDE, SQUAD): GLA outperforms RetNet and Mamba (e.g., SWDE 66.6 vs Mamba 41.4, RetNet 42.8 at 1.3B scale) attributed to "larger recurrent state (compared to Mamba) and selection mechanism (compared to RetNet)" — but all subquadratic models still underperform Transformer++ substantially (SWDE 66.6 vs 74.9 for Transformer++... verify), showing gating narrows but doesn't close the recall gap vs full attention.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A purely additive (Hebbian, ungated) linear-attention recurrence — structurally the same outer-product accumulation used in this proposal's carry — has been shown to underperform even a simple scalar data-independent decay by a wide margin (23.21 vs 16.55 training perplexity at matched scale), and underperforms full data-dependent fine-grained gating by an even larger margin (14.77 PPL) (Yang et al., 2024)." (Discussion — quantifies, with a controlled ablation, why an undecayed Hebbian fast-weight carry is expected to be a weak configuration relative to gated alternatives)
- "The absence of a forget/decay term in a linear-attention-style recurrent state has been hypothesized in prior work to underlie instability of such fast-weight states over long contexts, since the model has no mechanism to remove stale associations (Yang et al., 2024, citing Buckman & Gelada, 2024)." (Discussion — candidate explanation for a carry that plateaus rather than compounds: capacity/interference saturation rather than active growth)
- "Matrix-valued recurrent states updated by an outer-product rule are a direct instance of the fast-weight framework, connecting linear attention, this proposal's carry, and classic associative memory formulations (Yang et al., 2024, citing Schmidhuber 1992; Ba et al. 2016)." (Related Work — lineage citation)

## Verification
- The no-gate/RetNet/data-dependent-scalar/full-GLA PPL ablation numbers (23.21, 16.55, 15.56, 14.77, and the 8-head/1-head variants 15.29/14.61) were read directly from Table 4 on page 9 of the PDF.
- Eq. 1 (ungated linear attention recurrence), Eq. 3 (GLA recurrence with `Diag(α_t)`), and the `G_t = α_t^T 1` low-rank gate parameterization were read directly from Section 2.1/4.1 (pages 2, 5).
- Table 2 main LM PPL numbers (GLA 17.22/14.47, Transformer++ 16.85/13.44, Mamba 17.06/13.89 at 1.3B/100B tokens) confirmed directly from Table 2, page 7.
- The SWDE comparison figure I nearly cited from Table 3 (Transformer++ value) was NOT clearly re-verified against the exact printed number in the extracted text — I have GLA's SWDE=66.6 and Mamba's=41.4 confirmed directly, but I did not confirm the exact Transformer++ SWDE value with confidence, so I omitted a specific Transformer++ number in the final bullet rather than risk misstating it.
- Did not read pages 11–23 (extended related work appendix, full backward-pass derivations, references) — not needed for this note's scope.

# FILE: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models_importance.md

# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Citation:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen, NeurIPS 2023 (arXiv:2306.14048v3).

## Important parts for my research
- Not a fast-weight/TTT paper — this is a KV-cache eviction policy for inference efficiency. Relevance here is as a contrasting approach to the same underlying problem (making long-context/multi-document inference tractable) via activation caching rather than parametric fast weights.
- Core mechanism: attention score matrices are >95% sparse even in densely-trained LLMs (measured on OPT family via Wiki-Text-103); a small set of tokens ("Heavy Hitters", H2) accumulate most of the attention mass and follow a power-law distribution. Removing H2 tokens causes severe accuracy collapse, confirming their functional importance.
- H2O eviction policy: dynamically retains a budget-k KV cache split between H2 tokens (by cumulative attention score, computed greedily from local/past statistics only — no future lookahead needed) and the most recent tokens. Formalized as a "dynamic submodular" maximization problem with a near-optimality guarantee (Theorem 4.4, informal: greedy top-k selection achieves (1-α)(1-1/e) of optimal minus β).
- With only 20% KV cache budget, H2O matches full-cache accuracy across OPT/LLaMA/GPT-NeoX (6.7B–66B) on COPA, OpenBookQA, PiQA, Winogrande, MathQA, RTE, XSUM, CNN/DailyMail — i.e., 5–10x memory reduction with no accuracy loss on these tasks.
- Throughput: up to 29x over DeepSpeed Zero-Inference, 29x over HuggingFace Accelerate, 3x over FlexGen (OPT-6.7B/30B, T4/A100 GPUs); latency reduced up to 1.9x at matched batch size vs FlexGen.
- Only-H2 or only-recent-tokens ablations both underperform full KV by 2.85–22.75%; combining both is necessary — recency alone (the "Local" strategy) collapses badly at high eviction rates (e.g., LLaMA-13B/XSUM, LLaMA-7B/CNN-DailyMail at 60% budget).
- Streaming result: H2O handles inputs up to 4 million tokens with lower perplexity than StreamingLLM's attention-sink method at matched cache size (tested on PG-19).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Efficient long-context inference has largely pursued activation-level compression — evicting or approximating which past key/value pairs to retain (Zhang et al. 2023's H2O finds >95% attention sparsity and a power-law 'heavy hitter' structure) — a fundamentally different strategy from compressing context into an updated parametric fast weight, which is the approach this work and In-Place TTT take." (Related Work)
- "H2O demonstrates that a small, dynamically-identified subset of tokens dominates attention allocation and that discarding the rest costs little accuracy, illustrating that most of a document's per-token information is not equally load-bearing for downstream prediction — a consideration relevant to interpreting why a compressed fast-weight carry might plateau rather than keep absorbing new documents." (Discussion)

## Verification
- Verified directly from PDF text: >95% attention sparsity claim (Sec 3.1), power-law heavy-hitter distribution and submodularity framing (Sec 3.2, Def 4.1, Thm 4.4), throughput multipliers (29x/29x/3x) and 1.9x latency figure (Abstract, Sec 5.2, Table 3/5), 20% budget matching full-cache accuracy across listed tasks (Table 1/2, Sec 5.1), 2.85–22.75% degradation range for single-component ablations (Sec 5.3 Q4), 4-million-token streaming claim and PG-19 comparison (Sec 5.3 Q1, Fig. 5).
- Could not independently verify exact numeric values in Tables 3–5 beyond what's shown in the extracted pages (e.g., precise per-cell throughput numbers were read directly and are accurate to the table as rendered).

# FILE: HOPFIELD NETWORKS IS ALL YOU NEED_importance.md

# Hopfield Networks is All You Need

**Citation:** Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter. arXiv:2008.02217v3 [cs.NE], 28 Apr 2021 (ELLIS Unit Linz / JKU Linz / IARAI).

## Important parts for my research

- Central result: transformer self-attention **is** one update step of a modern continuous-state Hopfield network. Derivation (Eq. 10): with stored patterns Y mapped by W_K to keys X^T=K, query patterns R mapped by W_Q to Q, and V = X^T W_V, the Hopfield update rule ξ^new = X·softmax(βX^Tξ) becomes softmax(1/√d_k · QK^T)V exactly when β=1/√d_k — i.e. **attention = one Hopfield retrieval step**, and QKV attention is literally associative-memory retrieval, not an ad hoc mechanism.
- New continuous energy function (Eq. 2): E = −lse(β, X^Tξ) + ½ξ^Tξ + β⁻¹log N + ½M². Update rule (Eq. 3): ξ^new = f(ξ) = X·softmax(βX^Tξ) = Xp. Proven (Thm 1, 2) to converge globally via CCCP (concave-convex procedure); limit points are stationary points (minima/saddle points) of E.
- Storage capacity is **exponential** in pattern dimension d (Thm 3: N ≥ √p·c^((d−1)/4) patterns storable with failure probability p), a major improvement over classical binary Hopfield's linear/near-linear capacity (Cd/log d, ~0.138d). Retrieval typically converges in **one update step** (Thm 4/5) with exponentially small error in pattern separation Δ_i — this one-step-retrieval property is exactly what makes it differentiable and layer-compatible (only one softmax application needed, matching standard NN layers being applied once).
- Metastable states: when patterns are not well-separated (similar to each other), iteration converges near the *average* of the similar cluster rather than a single pattern — a soft/graded retrieval mode; controlled by inverse-temperature β. Empirically, pretrained BERT attention heads are classified into 4 regimes from "large metastable state" (broad averaging, early layers) to fixed points (single-pattern retrieval, later layers) by measuring effective softmax entropy (their k-statistic).
- Introduces 3 new differentiable layer types built on this energy (`Hopfield`, `HopfieldPooling`, `HopfieldLayer`) that generalize attention/pooling/fully-connected layers as memory-lookup layers with learnable or externally-filled stored patterns; state-of-the-art results on 3/4 multiple-instance-learning datasets, immune repertoire classification, 10/75 UCI small-classification datasets, and 2 drug-design benchmarks (BACE, SIDER).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / Methodology framing) — attention-as-Hopfield-retrieval equivalence (Eq. 10) as the theoretical grounding for describing In-Place-TTT-style fast weights and KV-memory update rules as associative-memory operations rather than ad hoc heuristics.
- (Related Work) — exponential storage capacity and provable single-step convergence (Thm 3, 4) as the classical theoretical ceiling for associative/outer-product memories, useful contrast when discussing why a Hebbian-only fast-weight update (no error correction) may saturate/plateau rather than keep compounding — capacity bounds exist even for well-behaved associative memories.
- (Discussion — explains flat carry, speculative) — metastable-state averaging behavior: when stored patterns are insufficiently separated, updates converge to a *shared average* rather than accumulating new distinct information; offers a mechanistic analogy for why repeated-session carry might plateau instead of compounding once accumulated fast-weight state resembles a metastable average of prior documents. Flag: this is an analogy from a different architecture (feedforward Hopfield lookup, not a decayed/accumulated fast-weight state across sessions), not a directly transferable proof.

## Verification
- Read full main text pages 1–10 (through Conclusion) directly; did not need the appendix (proofs, additional BERT head analysis) for the claims cited above. Eq. 10 derivation and β=1/√d_k equivalence read directly from Section 2, not inferred. UCI/MIL/drug-design numbers transcribed directly from Tables 1–2 and text. The "explains flat carry" connection is my own analogy, explicitly flagged as such above, not a claim made by the paper itself (this paper never discusses fast-weight persistence or TTT).

# FILE: Hybrid computing using a neural network with dynamic external memory_importance.md

# Hybrid computing using a neural network with dynamic external memory

**Citation:** Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, Demis Hassabis (DeepMind). *Nature*, Vol. 538, pp. 471–476, 27 October 2016. doi:10.1038/nature20101. Introduces the Differentiable Neural Computer (DNC).

## Important parts for my research

- DNC = a "controller" recurrent/feedforward network coupled to an external N×W memory matrix M it reads/writes via differentiable attention, generalizing the earlier Neural Turing Machine (NTM). Memory is external to network weights — a different persistence mechanism than fast-weight TTT (memory is a stored matrix accessed by content/location, not a set of model parameters updated by gradient steps).
- Three read/write attention mechanisms, combinable: (1) **content-based lookup** — cosine similarity between a key k and memory rows, C(M,k,β)[i] = exp(D(k,M[i,·])β) / Σ_j exp(D(k,M[j,·])β), sharpened by key strength β (this is functionally identical in form to key-value dot-product/softmax retrieval, i.e. a Hopfield/attention-style associative recall over an *external* matrix rather than internal weights); (2) **dynamic memory allocation** — a differentiable "free list" analogue tracking a usage vector u_t ∈ [0,1]^N per location, freed via per-read-head "free gates," so the controller can allocate to unused slots without external supervision; (3) **temporal link matrix** L_t ∈ [0,1]^{N×N} tracking write order, enabling forward/backward sequential replay of memory (near-adjacent-write recall) — approximable at O(N log N) cost via a sparse-link variant (only K nonzero entries per row/column) for scalability.
- Write operation: M_t = M_{t-1} ∘ (E − w_t^w e_t^⊤) + w_t^w v_t^⊤ (erase-then-write with erase vector e_t and write vector v_t, weighted by write weighting w_t^w) — an explicit erase term, unlike the purely additive/Hebbian outer-product accumulation of TTT fast weights; this erase-write gating is one concrete precedent for how a persistence mechanism could implement *forgetting*, relevant to any discussion of carry not being purely monotonic-additive.
- Empirical results: on bAbI (20 synthetic QA task types), single DNC reaches 3.8% mean per-task error (task failure = >5% error) on 2/20 task types, vs. 7.5% mean/6 failed tasks for the best previous jointly-trained LSTM; on graph tasks (London Underground traversal 98.8% avg accuracy, shortest-path 55.3%, family-tree inference 81.8%), DNC generalizes from random-graph training to specific real graphs never seen in that exact form; on Mini-SHRDLU block-puzzle (RL setting) DNC solves goals LSTM cannot learn at all within the training budget (best LSTM run reached only 37% accuracy on the easiest curriculum level after ~2M episodes vs. DNC's 98.8% after ~1M).
- Discussion explicitly frames the memory-allocation/temporal-link machinery as motivated by hippocampal analogues (CA3/CA1 fast synaptic potentiation, dentate gyrus sparse coding for capacity, "free recall" temporal-context effects) — a citable parallel for framing persistent fast-weight memory as loosely hippocampus-like.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) — DNC as a landmark external-memory/read-write architecture predating and architecturally distinct from TTT: memory is a separate addressable matrix manipulated by learned attention heads, not a subset of the network's own weights updated by gradient descent — useful contrast to sharpen what makes fast-weight TTT (weights-as-memory) a different persistence paradigm.
- (Related Work) — dynamic memory allocation (free gates + usage vector) and content+temporal addressing as prior art for *managing* persistent state across long/streaming input, relevant background when motivating why cross-document carry needs explicit design choices (reset vs. decay vs. accumulate) rather than being automatic.
- (Discussion, optional) — the explicit erase-then-write update rule (M_t = M_{t-1}∘(E − w e^⊤) + w v^⊤) as a concrete precedent for forgetting/erase mechanisms in persistent memory, if discussing why a pure-accumulation (Hebbian, no erase) fast-weight carry might behave differently (e.g., saturate or not compound) than an erase-capable memory.

## Verification
- Read full main article (pages 1–7, through Discussion/references) plus the first page of Methods (controller equations, content-based addressing formula, dynamic allocation, temporal link matrix, sparse link matrix complexity) directly from the PDF. All numeric results (3.8% bAbI error, 98.8%/55.3%/81.8% graph task accuracies, 37% vs. 98.8% Mini-SHRDLU comparison) transcribed directly from the text, not estimated. Did not read remaining Methods/Supplementary pages (task descriptions, further equations) beyond what's summarized here — not needed for this note's claims.

# FILE: In-Place Test-Time Training_importance.md

# In-Place Test-Time Training

**Citation:** Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Di He, Wenhao Huang, Tianle Cai (ByteDance Seed / Peking University), 2026. arXiv:2604.06169v1 [cs.LG], 7 Apr 2026. Code: github.com/ByteDance-Seed/In-Place-TTT.

## Important parts for my research

- **Fast weight choice:** repurposes only the MLP's final projection matrix **W_down** (in a gated MLP O = (φ(HW_gate^T) ⊙ (HW_up^T)) W_down^T) as the fast weight; W_up and W_gate stay frozen "slow weights." No new layer added — "drop-in," no retraining from scratch needed.
- **Update rule (chunk-wise, apply-then-update):** for chunk i, Apply: O_[i] = Z_[i](W_down^(i))^T. Update: W_down^(i+1) = W_down^(i) − η∇_W L(Z_[i](W_down^(i))^T, V_[i]). With their similarity loss L(·,·) = −⟨·,·⟩_F, this reduces in closed form to **W_down^(i) = W_down^(i−1) + η V̂_[i]^T Z_[i]** (Eq. 1) — a pure outer-product (Hebbian-style) accumulation, no error-correction/delta term.
- **LM-aligned target (their key contribution over prior TTT):** target V̂ = Conv1D(X_0)·W_target, where Conv1D is depthwise (kernel size 5, causal padding, no bias) and W_target ∈ R^{d×d} is learned. At init, Conv1D is zero-initialized and W_target is a near-identity sparse-diagonal matrix — so effective ΔW_down ≈ 0 at start, guaranteeing the model starts identical to the pretrained checkpoint and TTT "smoothly emerges." Setting Conv1D kernel to select "1 for next token, 0 else" plus identity W_target recovers pure next-token-prediction target; this replaces prior work's reconstruction target v = current token embedding.
- **Theorem 1 (proved, App. A):** under induction-head-style assumptions (near-orthogonal embeddings, key-query alignment), the NTP-aligned target guarantees E[Δlogit for correct token] ≥ λ_lr·c_norm²·c_align (>0) while other logits shift by ≤ λ_lr·ε·c_align (≈0); the reconstruction target's effect on the correct logit is bounded by the same negligible ≤ λ_lr·ε·c_align — i.e., reconstruction targets provably do NOT help next-token prediction in expectation, only the NTP-aligned target does.
- **Document-boundary handling — directly answers the assignment's key question:** "at document boundaries, the fast weights are reset to their pre-trained state to prevent context leakage across independent sequences" (§3.4, "Causality and Boundary Handling"). Algorithm 1 (App. B), line 14, explicitly: "At document boundaries: Reset fast weights to W_down^(0)." **This is unconditional and total reset — no partial carry, no decay, no cross-document persistence tested anywhere in the paper.** This is the exact gap our proposal targets.
- **Context-parallel implementation:** because the update is a pure sum of per-chunk outer products (associative), ΔW_down deltas can be computed for all chunks in parallel, prefix-summed (cumsum) to get aggregated state per chunk, then applied in parallel — fully CP-native despite being "causal." This is only possible because the update has no nonlinearity/error term (pure Hebbian accumulation) — an error-correcting (delta-rule) update would break this associativity.
- **Inference-time stability trick:** for the continually-pretrained Qwen3-4B setting, they clip each chunk's update delta by Frobenius norm to a threshold τ = 1e-5 (ΔW ← τ·ΔW/‖ΔW‖_F if ‖ΔW‖_F > τ) "to ensure stable fast-weight updates" and prevent unbounded growth over long sequences — directly analogous to `clip_tau` in our own implementation.
- **RULER results (Table 1, Qwen3-4B-Base, drop-in):** Baseline vs In-Place TTT accuracy(%): 4k 96.6→96.1 (TTT slightly *worse*), 8k 94.1→95.6, 16k 92.1→92.7, 32k 88.7→89.3, 64k 74.3→**78.7**, 128k 74.8→**77.0**, 256k extrapolation 41.7→**43.9**. Gains widen with context length; at short context (4k) baseline wins.
- **Table 2 (extension to LLaMA-3.1-8B, Qwen3-14B-Base):** consistent gains at all lengths; +2.1 at 64k (LLaMA-3.1-8B), +2.7 at 64k (Qwen3-14B); gains hold when combined with YaRN RoPE extension (64k+YaRN: 81.3→82.5 for Qwen3-14B).
- **Table 3 (4B models, pretrained from scratch, 8k context):** RULER-16k Full-Attn 6.58→19.99; RULER-8k SWA 9.91→26.80; common-sense reasoning (HellaSwag/ARC/MMLU/PIQA) mostly flat-to-slightly-improved.
- **Ablations (§4.3, Fig. 3):** state size (number of TTT-enabled layers) scales monotonically with performance; chunk size C=512 or 1024 best (1024 more efficient — larger chunks than typical TTT literature because attention already handles fine-grained token mixing, only MLP is chunk-updated); LM-aligned objective ablation shows Conv1D component essential for long context, W_target essential for short context, both needed together.
- **Baselines used, from-scratch comparison (§4.2):** standard Transformer+SWA, Gated Linear Attention (GLA), DeltaNet, and LaCT (Large Chunk Test-Time Training) — **LoRA is never mentioned anywhere in the paper, and no LoRA citation appears in the 67-item reference list.** Confirms the FuturePlan.md claim that In-Place TTT never compares to LoRA — verified directly, no correction needed.
- **Efficiency (Fig. 4):** negligible prefill throughput/memory overhead vs. baseline SWA/Full-Attn at 8k–128k context, because only one matrix (W_down) is touched and updates are chunk-parallel.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / Methodology) — direct architectural basis: In-Place TTT repurposes MLP W_down as fast weight, chunk-wise apply-then-update, LM-aligned Conv1D target rather than reconstruction target (Eq. 1, §3.1–3.2).
- (Related Work — gap statement) — "at document boundaries, the fast weights are reset to their pre-trained state to prevent context leakage across independent sequences" (§3.4) + Algorithm 1 line 14: quote directly to establish that In-Place TTT never carries fast weights across document boundaries — the precise gap this proposal fills.
- (Related Work — no LoRA baseline) — cite the from-scratch baseline set (SWA, GLA, DeltaNet, LaCT, Table 3/Fig. 2) and the complete absence of any LoRA comparison or citation, to justify adding LoRA and LoRA+TTT as baselines ourselves.
- (Methodology) — Frobenius-norm clipping of the update delta (τ = 1e-5) as precedent for our own `clip_tau` stabilization mechanism.
- (Discussion) — Theorem 1's proof that a reconstruction-style/pure-Hebbian target has *provably negligible* expected effect on the correct next-token logit; useful for framing why a Hebbian-only update rule (ours) might plateau/carry-flat unless the target is explicitly NTP-aligned like theirs.
- (Discussion) — pure outer-product update (Eq. 1) has no error-correction term, same "hebbian, no delta rule" character as our active update rule; the paper's own associativity/parallel-scan argument (§3.4) only holds because of this Hebbian-only form, i.e., adding delta-rule correction would break their CP algorithm — relevant if discussing why delta rule is deprioritized.

## Verification
- Re-read all 20 pages including full Appendix (proof of Theorem 1, Algorithm 1 pseudocode, all hyperparameter tables, references 1–67). Boundary-reset claim confirmed verbatim in both §3.4 prose and Algorithm 1 line 14 — not inferred. RULER numbers in Tables 1–3 transcribed directly from the tables, not estimated. LoRA-absence claim confirmed by scanning the full reference list (no LoRA/Hu et al. 2021 entry present) and searching main text — no inference involved.

# FILE: LONG RANGE ARENA: A BENCHMARK FOR EFFICIENT TRANSFORMERS_importance.md

# Long Range Arena: A Benchmark for Efficient Transformers

**Citation:** Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler (Google Research, Google DeepMind). arXiv:2011.04006v1 [cs.LG], 8 Nov 2020 (Preprint).

## Important parts for my research
- Six synthetic/real-data tasks spanning 1K–16K tokens, purpose-built to probe long-range reasoning independent of pretraining: Long ListOps (2K, hierarchical 10-way classification), byte-level IMDb text classification (4K), byte-level ACL Anthology document retrieval/matching (4K per doc, two-tower, no cross-attention), CIFAR-10 pixel-sequence image classification (1024), Pathfinder (1024, binary path-connectivity), and Pathfinder-X (16K, same task at extreme length).
- Defines "required attention span" (mean query–attended-token distance weighted by attention) as a task-difficulty proxy computed from a trained vanilla Transformer; all LRA tasks score high, confirming they need genuinely long-range dependency use, not just local context.
- Striking failure mode: every one of 10 evaluated efficient-Transformer variants (plus vanilla Transformer) fails to learn Pathfinder-X (~50%, chance) despite succeeding at the identical task at 1024 length — pure sequence-length scaling, no architecture/task change, can cause total training failure rather than graceful degradation.
- No one-size-fits-all result: BigBird has the best average LRA score (55.01) but is not best on any individual task; kernel-based models (Performer, Linear Transformer) trade quality for large speed/memory wins (Performer 5.7× faster, Linformer ~10× less memory than vanilla Transformer at 4K).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology) The Pathfinder-X total-failure-at-length finding is a useful cautionary parallel for our "stability checked before trusting any gain" step — evidence that extreme-length/repeat-count regimes can produce qualitative training breakdowns rather than smooth trends, motivating our own stability checks before trusting carry effects.
- (Related Work / Methodology) "Required attention span" as a difficulty proxy and LRA's broader philosophy (isolate long-range dependency use, avoid conflating inductive bias with pretraining benefits) is a reusable framing for justifying our repeated-session, carry-isolating evaluation design.

## Verification
- Task descriptions, Table 1 results (Transformer 54.39, BigBird 55.01, all models FAIL on Path-X), and efficiency numbers (Table 2) were read directly from the PDF (pp. 3–8).

# FILE: LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS_importance.md

# LoRA: Low-Rank Adaptation of Large Language Models

**Citation:** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft Corporation), arXiv:2106.09685v2, 2021 (ICLR 2022)

## Important parts for my research
- Core mechanism: freezes pretrained weight `W0 ∈ R^(d×k)`; injects a trainable low-rank update `ΔW = BA` with `B ∈ R^(d×r)`, `A ∈ R^(r×k)`, `r ≪ min(d,k)`. Forward pass: `h = W0·x + ΔW·x = W0·x + BA·x` (Eq. 3). `A` is random-Gaussian-initialized, `B` initialized to zero, so `ΔW=0` at training start. Output is scaled by `α/r` (α is a constant, tuned like a learning rate).
- Applied *only to attention weight matrices* in most experiments (`Wq, Wk, Wv, Wo`); MLP modules, LayerNorm, and biases are explicitly frozen/untouched — "we leave the empirical investigation of adapting the MLP layers... to a future work" (Sec 4.2). This is a direct contrast point since In-Place TTT/this proposal instead updates the MLP down-projection.
- Most reported results use LoRA on `{Wq, Wv}` only, for simplicity — not all four attention matrices.
- Rank ablation (Table 6, GPT-3 175B): `r=1` on `{Wq,Wv}` already gets WikiSQL 73.4 / MultiNLI 91.3, barely below `r=64` (73.7 / 91.4). Full model rank `d=12,288`; a rank of 1–2 suffices. Table 5: given a fixed 18M-parameter budget, spreading rank across both `Wq` and `Wv` beats putting all rank into just `Wq` or `Wk` (73.7/91.3 vs ~70–71/91.0).
- No additional inference latency: `BA` can be explicitly merged into `W0` at deployment (`W = W0 + BA`), unlike adapter layers, which add sequential compute and show a measured 20–30% latency increase at batch size 1 (Table 1).
- Efficiency numbers (GPT-3 175B): 10,000× fewer trainable params vs full fine-tuning, 3× less GPU memory (Adam VRAM 1.2TB → 350GB), checkpoint size 350GB → 35MB (r=4, only Wq/Wv adapted), 25% training speedup.
- Subspace analysis (Sec 7.1–7.3): singular-vector directions of `ΔW` learned at `r=8` and `r=64` overlap strongly only in the top 1–2 directions (normalized subspace similarity >0.5 for dim-1 overlap) — explains why very small r works. `ΔW` correlates with `W` but *amplifies directions that are underrepresented in W*, with an amplification factor of ~21.5 for r=4 (Table 7) — i.e., LoRA doesn't just repeat what pretraining already emphasized, it boosts task-relevant directions W had learned only weakly.
- Numeric benchmark results directly usable for comparison: RoBERTa-base LoRA avg 87.2 vs full FT 86.4 (0.3M vs 125M trainable params); RoBERTa-large LoRA avg 89.0 ≈ FT 89.0 (0.8M vs 355M); DeBERTa-XXL LoRA 91.3 ≈ FT 91.3 (4.7M vs 1500M); GPT-3 175B (Table 4) LoRA (4.7M params) WikiSQL 73.4 / MNLI-m 91.7 / SAMSum R1/R2/RL 53.8/29.8/45.9 vs full FT (175,255.8M params) WikiSQL 73.8 / MNLI-m 89.5 / SAMSum 52.0/28.0/44.5 — LoRA matches or exceeds full FT with ~5 orders of magnitude fewer trainable params.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "LoRA freezes the base model and learns a rank-`r` additive update `BA` to a weight matrix, composing as `h = W0x + BAx`, with zero-initialized `B` guaranteeing the adapted model equals the base model at the start of training" (Methodology — describing the `lora_only` ablation precisely).
- "In the original formulation, LoRA is applied only to attention projection matrices (typically `Wq, Wv`), leaving the MLP block untouched; this proposal's TTT mechanism instead targets the MLP down-projection, making the LoRA-only baseline and the TTT mechanism structurally non-overlapping in which weights they adapt" (Related Work / Methodology — motivates why LoRA-only and LoRA+TTT are not redundant a priori, which sharpens the surprise of the null result).
- "LoRA at very low rank (r=1–4) already recovers ~99% of full fine-tuning performance on GPT-3 175B, at 10,000× fewer trainable parameters, establishing LoRA as a strong, cheap general-adaptation baseline that any additional adaptation mechanism (e.g., TTT) must beat to be worth its cost" (Related Work — sets up why "LoRA alone outperforms LoRA+TTT" is a meaningful, not trivial, negative result).
- "Hu et al. (2021) show LoRA's learned update is low-rank and concentrated in directions the pretrained weights underweight, rather than in directions already dominant in the base model" (Discussion — potential contrast: unlike LoRA's directed, gradient-corrected low-rank update, this work's Hebbian fast-weight carry has no comparable error-correction signal, which may explain why LoRA's targeted low-rank update outcompetes an uncorrected additive carry).

## Verification
- Directly verified from PDF text: the `h = W0x + BAx` formula (p.4, Eq. 3), zero-init of B / Gaussian init of A, the `α/r` scaling, attention-only targeting statement in Sec 4.2 ("We limit our study to only adapting the attention weights... and freeze the MLP modules"), Table 1 latency numbers, Table 2/3/4/5/6/7 numeric values, and the 10,000×/3×/350GB-to-35MB/25%-speedup claims (Sec 4.2, p.5–6).
- Not independently re-derived: the amplification factor "21.5 ≈ 6.91/0.32" is quoted directly from the paper's own stated ratio (p.12), not recomputed by me.

# FILE: Language model evaluation beyond perplexity_importance.md

# Language Model Evaluation Beyond Perplexity

**Citation:** Clara Meister, Ryan Cotterell (ETH Zürich, University of Cambridge), 2021. arXiv:2106.00085v3 [cs.CL] (30 Aug 2021).

## Important parts for my research
- Core proposal: instead of (or alongside) perplexity, test whether text *generated* by a model reproduces the *statistical tendencies* of natural language (Zipf's rank-frequency law, Heaps' type-token law, length/stopword/symbol distributions, unigram distribution) — a hands-off framework that doesn't require assuming a priori which linguistic phenomena matter.
- Uses statistical-distance "primary metrics" (difference in means) plus "simple metrics": Kolmogorov-Smirnov (KS) statistic D = sup_y |P1(y)-P2(y)| (one-sample: empirical vs. theoretical cdf; two-sample: two empirical cdfs) and Total Variation Distance (TVD) for categorical/unigram distributions.
- Nonparametric significance testing via **permutation tests**: build the null sampling distribution of a test statistic φ(S1,S2) by exhaustively (or via Monte Carlo) recombining pooled samples into new S1',S2' splits and comparing observed stat to that empirical distribution — this avoids needing to know the true underlying distribution. Pseudocode given (Algorithm 1, App. B).
- Explicit methodological caution: with very large sample sizes (they use 1M samples per set), almost every null hypothesis test will be statistically significant even when the true effect size is trivial — they flag this directly and argue effect-size/statistic magnitude must be reported alongside significance, not significance alone.
- Empirical findings: neural LMs (Transformer, Transformer w/ adaptive softmax, CNN, LSTM, trigram) capture only a *subset* of natural-language statistical tendencies, and which subset depends heavily on **both architecture and decoding/generation strategy** (ancestral random sampling vs. nucleus sampling vs. beam sampling) — no single model/decoding configuration matches all trends.
- Nucleus sampling text adheres most closely to the natural type-token (Heaps') relationship; beam-sampled text diverges strongly from natural-language distributions across essentially every metric tested (length, stopwords, symbols, type-token).
- On Zipf's law: even the real test-set text does not perfectly fit a theoretical Zipfian distribution (D_p̂ = 0.148, i.e., natural language itself deviates from the idealized power law) — so "model doesn't perfectly match theoretical law" is not itself evidence of model failure; comparison should be model-vs-empirical-natural-language, not model-vs-theoretical-law.
- Corpus: English Wikipedia Dumps, preprocessed as in XLM (Conneau & Lample 2019), 44.7M train / 1e4 valid / 1e6 test tokens; all comparisons draw 1M generated samples per model/decoding setting.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Model evaluation should not rely on perplexity alone: text generated under inference-time adaptation can be probed for whether its statistical/distributional properties (not just token-level likelihood) shift toward or away from the target domain" (Methodology / Discussion — motivates supplementing perplexity with distributional diagnostics).
- "Following Meister & Cotterell (2021), we note that with large held-out sample counts, standard significance tests will reject the null for arbitrarily small effect sizes; we therefore report effect magnitudes (e.g., perplexity deltas, carry-norm growth) alongside bootstrap CIs rather than significance alone" (Methodology — statistical rigor framing, directly supports the proposal's bootstrap-CI-across-many-documents design).
- "Prior work shows that a generative model's fit to natural-language statistics depends jointly on architecture and decoding/sampling strategy, with no single configuration matching all empirical trends (Meister and Cotterell, 2021)" (Related Work — general caution against overclaiming uniform model improvement from any single intervention, applicable to framing a mixed/null TTT result).
- Permutation testing as a nonparametric significance-test technique that requires no assumption about the underlying data distribution is directly reusable for comparing carry-on vs. carry-off perplexity distributions across held-out documents (Methodology).

## Verification
- Directly verified from PDF text: the KS/TVD formal definitions (Eqs. 10-11), permutation test procedure (§4.1, App. B pseudocode), Zipf's-law D_p̂=0.148 finding (§5.1), the "almost always significant due to large sample sizes" caveat (footnote 8, §5), nucleus-sampling closest fit to type-token (§5.2), beam sampling divergence (§5.4/5.5), and corpus/sample-size details (§5, footnotes 8-9).
- Not independently re-derived: exact numeric KS values in Table 2/3 (e.g., Transformer D_p=0.150) were read directly from the table, not recomputed — treated as accurately transcribed rather than verified for internal consistency.

# FILE: Language models are few-shot learners_importance.md

# Language Models are Few-Shot Learners

**Citation:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (OpenAI). arXiv:2005.14165v4 [cs.CL], 22 Jul 2020. Introduces GPT-3.

## Important parts for my research

- Introduces **in-context learning (ICL)** as the paradigm of adapting a frozen model purely via its input context — explicitly framed as a meta-learning **inner loop** (task adaptation happening within a single forward pass) nested inside an **outer loop** of ordinary SGD during pretraining (Fig. 1.1). This is the canonical no-weight-update contrast case for TTT: ICL adapts via activations/context only, bounded by context window; TTT adapts via parameters (fast weights), persisting beyond the context window and (in principle) beyond a single document.
- Three evaluated settings, all with **zero gradient updates or fine-tuning**: zero-shot (task instruction only), one-shot (instruction + 1 demonstration), few-shot (instruction + K≈10–100 demonstrations, limited by context window n_ctx=2048 tokens). Traditional fine-tuning (repeated gradient updates) is explicitly excluded from GPT-3's evaluation and left to future work (§2, Fig. 2.1) — GPT-3 itself never updates weights at inference time, making it a pure-context baseline against which any weight-updating method (TTT, LoRA) can be positioned.
- Model family: 8 sizes from 125M to 175B params (Table 2.1); GPT-3 175B has 96 layers, d_model=12288, 96 heads, trained on 300B tokens with context window n_ctx=2048. Architecture is GPT-2-style with alternating dense/locally-banded sparse attention (Sparse Transformer-style).
- **Scaling result (Fig. 1.3):** few-shot performance improves faster with model scale than zero-shot across 42 benchmarks — larger models are more efficient at exploiting in-context information (steeper "in-context learning curves," Fig. 1.2), suggesting in-context adaptation ability itself is an emergent/scaling property, not just raw capability.
- Selected numbers: CoQA F1 zero-shot 81.5 / one-shot 84.0 / few-shot 85.0; TriviaQA (closed-book) zero-shot 64.3% / one-shot 68.0% / few-shot 71.2% (few-shot result state-of-the-art relative to fine-tuned closed-book models at the time). Useful as a scale reference for how much few-shot context-conditioning alone can close the gap to fine-tuning, without any weight update.
- **Limitations directly relevant to motivating TTT/persistence (§5):** (1) GPT-3 has "poor sample efficiency during pre-training" — sees far more text than a human ever would, unlike test-time adaptation which could in principle use far less data per update; (2) explicit open question — it is *ambiguous whether few-shot ICL learns new tasks "from scratch" at inference or merely recognizes/identifies a task pattern already learned during pretraining* — directly relevant framing for asking the analogous question about our own TTT carry (is compounding gain from genuinely new information, or just recognition/pattern-matching effects); (3) the objective "weights every token equally and lacks a notion of what is most important to predict" — a direct antecedent motivation for target-shaping approaches like In-Place TTT's NTP-aligned (future-token) objective vs. plain reconstruction; (4) structural limit acknowledged: purely autoregressive, no bidirectionality, and — most relevant to context-length work — the paper flags that scaling will "eventually run into (or could already be running into) the limits of the pretraining objective," i.e. context-bound in-context learning has an inherent ceiling that motivates architectures/methods (like ours) that persist state beyond a fixed context window.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / Introduction — contrast case) — in-context learning as the dominant zero-weight-update paradigm for test-time task adaptation (meta-learning inner-loop framing, Fig. 1.1), positioned against TTT/fast-weight approaches that instead update actual parameters at inference time; useful single citation to establish "what TTT is not."
- (Related Work) — GPT-3's context window (n_ctx=2048) and the explicit exclusion of fine-tuning from its evaluation protocol, to motivate why context-only adaptation cannot by itself provide unbounded/cross-document persistence — the gap that both long-context architectures and TTT (including cross-document carry) try to close from different directions.
- (Discussion) — the paper's own open question of whether ICL is "learning from scratch" vs. "recognizing a pretrained task" (§5) as a direct methodological precedent for interrogating whether our carry-attributable gains reflect genuine new learning vs. degenerate pattern recognition, especially given the null/flat-carry result.
- (Discussion) — the "every token weighted equally, no notion of importance" pretraining-objective limitation as backdrop for why NTP-aligned/future-token-weighted TTT objectives (In-Place TTT) were proposed as an improvement over naive reconstruction targets.

## Verification
- Read pages 1–8 (title through start of §2.2 Training Dataset, including Table 2.1 model sizes and Fig. 1.1–1.3) and pages 33–34 (§5 Limitations through start of §6 Broader Impacts) directly from the PDF. CoQA/TriviaQA numbers and model architecture table transcribed directly, not estimated. Did not read the full 75-page paper (Results §3 subsections 3.1–3.9, contamination study §4, Broader Impacts §6, Related Work §7, appendices) in detail — only the Introduction/Approach framing and Limitations were needed for this note's scope (general in-context-learning contrast, not benchmark-by-benchmark comparison), and this scoping choice is deliberate rather than an oversight.

# FILE: Learning What to Remember: Test-Time Training via Context Distillation_importance.md

# Learning What to Remember: Test-Time Training via Context Distillation

**Citation:** Zixuan Wang, Xingyu Dang, Rui-Jie Zhu, Zixin Wen, Hengyu Fu, Wenhao Chai, Jason D. Lee (Princeton, UC Berkeley, UC Santa Cruz, Carnegie Mellon University). arXiv:2608.01672v1 [cs.CL], preprint dated 3 Aug 2026.

## Important parts for my research
- Proposes TTCD (Test-Time Context Distillation): a shared frozen backbone runs a long-window "teacher" and short-window "student" over the same chunk; the fast weight is updated to make the student's MLP output match the teacher's (stop-gradient) output, rather than reconstructing raw tokens.
- IP-TTCD is the in-place instantiation: it updates the MLP down-projection matrix W_down exactly as In-Place TTT does, but with the distillation loss instead of reconstruction — same fast-weight site as our proposed extension of In-Place TTT.
- Update rule (chunk i, size C): W_down^(i) = W_down^(i-1) − (η_i/|C_i|)[Y_i^(S) − stopgrad(Y_i^(T))][Z_i^(S)]ᵀ — this is an error-correcting (delta-rule-like) update using the teacher−student residual as the "error" signal, NOT pure Hebbian outer-product.
- Theoretical result (Proposition 1): under stated assumptions, the fast-weight update induced by an earlier in-context episode provably shifts the student's hidden state at a later related query toward the teacher's (in-context) solution — a formal argument for why fast weights can carry useful signal to later positions within a sequence.
- Practical add-ons: causal short convolutions on teacher/student MLP activations plus a learned projection before forming the write key/value (ΔW_down = β·R·Kᵀ), off-policy chunk parallelism (all chunk updates computed from the base W_down^(0) then accumulated causally, avoiding sequential dependency).
- Results (760M, RULER NIAH avg): IP-TTCD 65.79/46.92/41.75/21.96 at 4k/8k/16k/32k vs IP-TTT 59.83/33.88/23.92/9.29 — more than doubles IP-TTT at 32k, gap widens with context length. Sliding-window perplexity on Books (Fig. 2) is consistently lowest for IP-TTCD across 1k–32k at both 340M and 760M, beating DeltaNet, Gated DeltaNet, SWA, and IP-TTT.
- Continual pretraining: applying IP-TTCD to already-pretrained SmolLM2-360M/1.7B and LLaMA-3.1-8B (lightweight augmentation, no from-scratch retrain) improves RULER average accuracy over IP-TTT and plain continual-pretraining baselines at most context lengths, though IP-TTT edges out IP-TTCD at 64K for the 360M model specifically.
- All experiments are within continuous long-context chunking (teacher/student windows inside one sequence) — no cross-document/session persistence is proposed or tested.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) IP-TTCD directly extends In-Place TTT (Feng et al. 2026), reusing the MLP down-projection as the fast-weight site but replacing reconstruction with context distillation; explicitly frames In-Place TTT as "a fundamental step towards connecting temporary memory and long term memory" — good citation for motivating persistent fast weights.
- (Methodology — contrast) Their update is error-correcting/delta-style (teacher−student residual), directly contrastable with our hebbian-only (pure outer-product, no error correction) active update rule — a citable methodological fork.
- (Discussion — explains flat carry) Proposition 1's mechanism (residual-driven write) suggests carry strength depends on a nonzero teacher−student discrepancy; under hebbian-only updates without such an error signal, weaker/flatter carry growth is plausible — cite as a candidate explanation.
- (Research Gap) Confirms nobody in this line (including this Aug-2026 paper) tests persistence across independently-bounded documents/sessions — all evaluation is chunk-wise within one long sequence.

## Verification
- Update equations, Proposition 1 statement, and RULER/sliding-window numbers were read directly from the PDF (pp. 4–9). The 360M-vs-64K anomaly (IP-TTCD 9.79 < IP-TTT 10.35) was cross-checked against Table 2 and is a genuine reading, not an inference — flagged for nuance rather than omitted, since the abstract's "gains widen with context length" claim doesn't hold uniformly at every model scale/length combination.

# FILE: Learning to (Learn at Test Time): RNNs with Expressive Hidden States_importance.md

# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

**Citation:** Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin (Stanford, UC San Diego, UC Berkeley, Meta AI). arXiv:2407.04620v4 [cs.LG], last revised 31 Aug 2025.

## Important parts for my research
- Core formulation: hidden state is itself model weights W_t; output rule z_t = f(x_t; W_t); update rule W_t = W_{t-1} - η∇ℓ(W_{t-1}; x_t) — a gradient step per token is TTT.
- Self-supervised loss is multi-view reconstruction: corrupted "training view" θ_K x_t, "label view" θ_V x_t, loss ℓ(W;x_t)=‖f(θ_K x_t;W) − θ_V x_t‖², output uses a separate "test view" θ_Q x_t — θ_K,θ_Q,θ_V are outer-loop learned, analogous to attention's K/Q/V.
- Two instantiations: TTT-Linear (f linear) and TTT-MLP (2-layer MLP, hidden dim 4× input, GELU); f always wraps LN + residual: f(x)=x+LN(f_res(x)) — critical for stability.
- Mini-batch TTT (chunk size b=16) is the single biggest ablation win (Table 1: 15.23→12.35 ppl), far more than learnable W_0 or learnable η; dual form gives >5× wall-clock speedup on TPUs by avoiding materializing per-token gradients.
- Theorem 1: TTT-Linear + batch GD + η=1/2 + W_0=0 is exactly equivalent to linear attention. Theorem 2: TTT with Nadaraya-Watson (nonparametric) learner is exactly equivalent to self-attention.
- DeltaNet is stated as equivalent to TTT-Linear with inner-loop batch size 1, without LN/residual — direct link between our hebbian/delta-rule design space and this framework's ablations.
- Learnable η(x)=η_base·σ(θ_lr·x) acts as a per-token gate on the gradient step — a mechanism for input-dependent update magnitude, relevant to controlling carry strength.
- Empirical: at 8k context, TTT-Linear/MLP beat Mamba; Mamba's per-token-index perplexity plateaus after 16k while TTT and Transformer keep improving to 32k — motivates why fixed-size RNN states saturate but TTT's learned compression does not (as much).
- W_0 (θ_init) is shared/learned across ALL sequences but hidden state is reset to W_0 at the start of every sequence — the paper's TTT is single-continuous-sequence only; no cross-document/session carry is discussed or tested.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Introduction/Related Work) Foundational TTT-layer framing: hidden state as a model, update as an SGD step, unifying RNNs/attention/TTT under one abstraction (Sun et al. 2025).
- (Methodology) Multi-view reconstruction loss and K/Q/V-style outer-loop parameterization as the template our self-supervised task follows.
- (Related Work — contrast) Theorem 1 equivalence of TTT-Linear+batch-GD to linear attention, and the paper's own statement that DeltaNet = TTT-Linear with batch size 1 minus LN/residual — useful for positioning hebbian (pure outer-product, no correction) vs. delta-rule variants in our design space.
- (Research Gap) This paper's TTT resets W_0 at every new test sequence; it never proposes or evaluates carrying fast weights across independently-bounded documents — supports our gap claim.
- (Discussion — mini-batch/chunk size effect) Chunk size b materially changes perplexity (Fig. 7); relevant if we discuss how chunking/frequency of updates could interact with flat carry.

## Verification
- Equations, Theorem 1/2 statements, Table 1 ablation numbers (15.91→11.09), and the DeltaNet-equivalence claim were read directly from the PDF text (pp. 4–17).
- The claim that W_0 is shared but resets per-sequence (no cross-document persistence) is a direct reading of Section 2.7 ("Learnable W_0") plus the absence of any cross-sequence carry discussion in the paper — not explicitly stated as a "limitation" by the authors, so this is an inference from what is and isn't described.

# FILE: Let's (not) just put things in Context: Test-time Training for Long-context LLMs_importance.md

# Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs

**Citation:** Rachit Bansal, Aston Zhang, Rishabh Tiwari, Lovish Madaan, Sai Surya Duvvuri, Devvrit Khatri, David Brandfonbrener, David Alvarez-Melis, Prajjwal Bhargava, Mihir Sanjay Kale, Samy Jelassi (Meta, Harvard University, Kempner Institute at Harvard, OpenAI, UC Berkeley, UT Austin). arXiv:2512.13898v1 [cs.LG], 15 Dec 2025.

## Important parts for my research
- Diagnoses "score dilution": in static softmax attention, if ≥m distractor keys have logits within Δ of the true target, target attention mass α ≤ 1/(1+m·e^{-Δ}) (Lemma 2.2) → as context length T grows with a constant fraction of near-tie distractors, α→0. Guaranteeing fixed target mass against worst-case distractors requires the logit margin to grow as Ω(log T) (Lemma 2.3).
- Proves generated "thinking" tokens cannot fix this: a thinking token's needle-signal is bounded by its own (diluted) attention mass on the needle (Prop. 2.4/Cor. 2.5) — more inference-time tokens under frozen weights cannot recover a buried signal, only re-shaping query-key similarity can.
- Proposes query-only TTT (qTTT): single full forward pass to cache K,V per layer (frozen, reused), then N_qTTT cheap gradient steps that update ONLY the query projection matrices W_Q (all else frozen) using standard next-token CE loss on randomly sampled short spans (length k≪T) from the same context.
- Proposition 3.1: gradient of the per-position retrieval loss w.r.t. query q_i is (1/√d_k)(μ_i − k_{j*}), where μ_i is the attention-weighted mean key — a descent step provably moves q_i toward the true target key and away from the distractor centroid, directly counteracting dilution; Lemma 3.2 shows the logit margin strictly increases each step.
- Full-parameter TTT (updating W_Q,W_K,W_V,FFN) is shown FLOP-infeasible for long context: one such step over T≈10^5 tokens costs ≈1.2×T FLOPs, equivalent to ≈120K decoding tokens, because every KV update invalidates the cache and forces full recompute — motivates restricting updates to queries only.
- FLOP-equivalence rule of thumb for comparing to "thinking": T_think ≈ 2·N_qTTT·k for long T, span k≪T — used to compute-match qTTT against chain-of-thought baselines.
- Results (Qwen3 1.7B/4B/8B, LongBench-v2 + ZeroScrolls): qTTT gives +12.6 and +14.1 avg points over in-context-only, beating FLOP-matched thinking-token baselines; e.g. Qwen3-4B Long Dialogue History 30.8→43.6, Multi-Document QA 40.0→46.0; Qwen3-8B Code Repositories 30.0(ICL)→44.0(thinking)→52.0(qTTT). Gains are largest on retrieval/multi-hop tasks; smaller and comparable-to-thinking on pure summarization (generation-bottlenecked, not retrieval-bottlenecked).
- Optimal LR range for qTTT is fairly wide (1e-6 to 1e-5), relatively insensitive within that band (Table 1, appendix).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Introduction/Research Gap) This is the "Bansal et al. 2025 qTTT" the proposal already cites — confirm: qTTT updates only attention query weights per-prompt via cheap gradient steps reusing a frozen KV cache; it is single-context, non-persistent (no fast weight survives across prompts/documents), and targets attention score dilution specifically — distinct mechanism from our MLP-down-projection fast-weight carry.
- (Related Work — contrast) Score dilution / Ω(log T) margin requirement is a formal argument for why static attention degrades in long context; useful to contrast against RNN/fast-weight compression approaches (like TTT-Linear/MLP, In-Place TTT) that don't rely on softmax attention margins at all.
- (Discussion) Their finding that thinking-token inference-time scaling has "rapid diminishing returns" and cannot repair retrieval failure is a useful comparator when discussing why other cheap inference-time interventions (not just carry-based TTT) sometimes underperform expectations.
- (Methodology) FLOP-matching methodology (T_think ≈ 2·N_qTTT·k) is a reusable pattern for compute-matched ablations, relevant to our "matched-budget ablations" remaining-bar item.

## Verification
- Lemma/Proposition statements, update-rule gradient derivation, and headline percentage figures (12.6%, 14.1%, per-task numbers like 30.8→43.6) were read directly from the PDF text and figures (pp. 2, 4–9). Confirmed this matches the proposal's "Bansal et al. 2025 qTTT" citation via author list and topic; note the arXiv date read from the PDF is Dec 2025, consistent with that citation year.

# FILE: Lifelong In-Context Learning with Transformers Requires Parametric Forms of Attention_importance.md

# Lifelong In-Context Learning with Transformers Requires Parametric Forms of Attention

**Citation:** Luke McDermott, Robert W. Heath Jr., Rahul Parhi (2026), Preprint, arXiv:2606.25342v1 [cs.LG], UC San Diego

## Important parts for my research
- Frames all attention as **test-time regression**: attention builds an online estimator m_t: R^dk → R^dv from a self-supervised training set {(k_j,v_j)}_{j≤t} generated by the transformer itself, then predicts y_t = m_t(q_t). Theorem 2.1 (citing Wang, Shi & Fox 2025): softmax attention is exactly Nadaraya–Watson kernel regression with RBF kernel K_h(q−k)=exp(qᵀk/√d_k), bandwidth h=d_k^(1/4), when ‖q‖=‖k‖=1.
- **Nonparametric vs. parametric** split: nonparametric (softmax) stores the growing KV cache itself as the estimator → cost grows with context length. Parametric attention (linear attention, SSMs, fast-weight programmers, TTT layers) bounds the estimator to a fixed-size state θ_t ∈ R^p updated online — this is the formal category the proposal's "carry" belongs to.
- Gives the exact **update-rule taxonomy** (Eq. 9) relevant to the hebbian/delta distinction: Hebbian rule = −⟨m(k_t|θ_{t−1}), v_t⟩ (pure correlational, no error term); Delta rule = ‖m_t(k_j|θ_{t−1})−v_j‖₂ (least-squares/error-correcting); Omega rule = decayed sum over a sliding window of c past pairs (Behrouz et al., used in Atlas). Linear attention's canonical Hebbian update (Eq. 10): θ_t = θ_{t−1} + v_t k_tᵀ — literally the outer-product accumulator this proposal calls "hebbian."
- **Direct explanation for a flat/non-compounding carry**: the paper states plainly that "instantaneous updates can unintentionally overwrite past information, leading to catastrophic forgetting" (citing McDermott et al.'s own LoLA work) — single-step Hebbian/instantaneous objectives only enforce m(k_t)≈v_t at the current step and can only *hope* earlier associations m(k_j)≈v_j, j<t survive; there is no mechanism enforcing consistency across the full history, unlike the Delta or Omega rule.
- States that linear maps under the Hebbian rule "fall short in memory capacity" — the number of orthogonal key–value pairs storable is bounded by the **rank of θ_t**, causing degraded long-term memory. This gives a concrete mechanistic reason a Hebbian carry could saturate/plateau rather than grow: once rank capacity is reached, new writes interfere with old ones instead of accumulating.
- Single-head, full-rank linear attention has a **closed-form degenerate solution**: if W_k is invertible, the "perfect" estimator is m*(q) = (W_v W_k^{-1})q — a fixed linear map that does **not depend on the context at all**. Multi-head linear attention avoids this by shrinking per-head parameters (d_kd_vh = d²/h), acting as implicit regularization.
- Names batched-update and auxiliary-state approaches as the two current fixes for short-horizon-only updates: sliding-window Omega rule (Titans/Atlas, Behrouz et al.), and second-order auxiliary state with a quadratic regularizer Λ (MesaNet, von Oswald et al. 2025) or momentum+orthogonalization via Muon (LaCT, Atlas) — orthogonalized updates are noted to reduce catastrophic forgetting "similar to how low-rank updates (LoRA) forget less" (citing Biderman et al. 2024).
- Explicitly frames gating/weight-decay as **online regularization** and argues current parametric methods lack principled regularization design — relevant to explaining why a pure Hebbian carry (no gating, no decay) either drifts unboundedly or saturates rather than showing controlled, useful growth.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Attention itself can be formalized as test-time regression over an online key-to-value estimator, with linear attention's canonical update being the Hebbian outer-product rule θ_t = θ_{t−1} + v_tk_tᵀ" (Methodology/Background — grounds the hebbian update rule used in this proposal).
- "Instantaneous, non-error-correcting update rules such as the Hebbian rule provide no guarantee that new writes preserve previously encoded associations, and are known to induce catastrophic forgetting of context — a candidate explanation for why a purely Hebbian carry fails to compound across sessions in our results" (Discussion — explains flat carry).
- "The memory capacity of a linear fast-weight state under Hebbian updates is bounded by the rank of the state matrix, meaning the number of independently storable key–value associations is inherently limited" (Discussion — explains a flat/saturating carry, not one that grows then plateaus).
- "McDermott et al. (2026) argue that current test-time-training/parametric-attention methods remain limited by short-horizon-only objectives and a lack of principled online regularization (e.g., gating, decay), problems our cross-document carry setting inherits and empirically characterizes" (Related Work — positions this proposal within an identified open gap).
- "This proposal's cross-document carry experiment operationalizes McDermott et al.'s open question of whether parametric attention/TTT states can retain information over long horizons, extending it to an explicit inter-session (cross-document) setting not tested in their survey of methods (Titans, Atlas, MesaNet, LaCT)" (Related Work — positioning ammunition).

## Verification
- All equations (Eq. 1–19), the Hebbian/Delta/Omega rule definitions, the rank-capacity claim, the closed-form single-head degenerate solution, and the "instantaneous updates...catastrophic forgetting" sentence were read directly from pages 4–8 of the PDF and are quoted/paraphrased accurately.
- The claim "McDermott et al. observe..." refers to a *different, self-cited* prior paper (likely ref [27], "Lola: Low-rank linear attention with sparse caching," McDermott, Heath Jr., Parhi 2025) — I did not independently verify that separate paper's content, only that this paper cites it for the catastrophic-forgetting claim. Flagged as inference from citation context, not independently confirmed.

# FILE: Linear Transformers Are Secretly Fast Weight Programmers_importance.md

# Linear Transformers Are Secretly Fast Weight Programmers

**Citation:** Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber (2021), Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139 — The Swiss AI Lab IDSIA, USI & SUPSI (arXiv:2102.11174v3)

## Important parts for my research
- Establishes the **formal equivalence** underlying "fast weights": removing softmax from causal self-attention (Eq. 7→8) gives y^(i)=W^(i)q^(i) with W^(i) = Σ_{j≤i} v^(j)⊗k^(j) — a purely additive outer-product fast-weight matrix. Incrementally, **W^(i) = W^(i−1) + v^(i)⊗k^(i)** (Eq. 10). This is precisely the pure Hebbian/outer-product accumulator this proposal's "hebbian" update rule implements — this paper is the theoretical origin of that mechanism (tracing to Schmidhuber 1991/92/93 Fast Weight Programmers, and to Hebb 1949 and Hopfield/Kohonen-style associative memory).
- **Capacity limitation (Sec. 4.1)**: because associations are stored additively in a matrix and retrieved via matrix-vector product, keys must be near-orthogonal to avoid interference; with keys in a d_dot-dimensional space, no more than d_dot associations can be stored without retrieval error. Sequence length > d_dot puts the model in an "overcapacity regime." Formalized via Tensor Product Representation theory (Smolensky 1990): crosstalk/retrieval error grows once capacity is exceeded.
- **Empirical demonstration the pure Hebbian ("sum") rule fails at exactly this capacity boundary**: with d_key=d_dot=64, linear attention retrieval error starts appearing once the number of stored associations exceeds ~60 (Fig. 2, Sec. 6.1.1).
- **Critical finding for explaining a flat/non-compounding carry**: in a *rebinding* setting where the same key is later paired with a new value (Sec. 6.1.2), the purely additive sum/Hebbian rule **cannot overwrite** a stale association — it can only add interference — so it fails to retrieve the updated value; only their proposed delta-rule-like update succeeds (Fig. 3). This is direct, quantified evidence that a pure Hebbian accumulator has no mechanism to update/replace old associations with new, more relevant ones — a structural reason repeated exposure would show flat, not compounding, improvement.
- Proposes the **delta-rule programming instruction** (Eq. 20–24), the "delta rule" this proposal keeps in code but leaves unused: retrieve current value v̄^(i)=W^(i−1)φ(k^(i)); compute a learned, input-dependent write-strength β^(i)=σ(W_βx^(i)); write a convex combination v_new=β^(i)v^(i)+(1−β^(i))v̄^(i); update W^(i)=W^(i−1)+β^(i)(v^(i)−v̄^(i))⊗φ(k^(i)) — i.e., write minus remove, an error-correcting update akin to Widrow–Hoff (1960).
- **WikiText-103 perplexity, sum (Hebbian) vs. delta rule** (Table 2): small config, Linear Transformer (sum) test PPL 38.3 vs. Delta Network (delta) 35.5; medium config, sum 33.0 vs. delta 31.5. Consistent ~2–3 point gain from error-correction over pure Hebbian accumulation, holding parameter count roughly fixed.
- **Carrying fast weights across segment boundaries without truncation** (Table 4, Sec. "Without truncating context"): they explicitly test carrying the fast-weight state across training segments (not resetting). For the sum/Hebbian rule this **"resulted in an even worse perplexity of over 1600" and "breaks"** without careful (removed) normalization; the delta-rule Delta Net, by contrast, carries successfully and reaches valid/test PPL 27.8/29.4 at a state size of only 0.13M — competitive with Transformer-XL at 2.10M state size (26.4/27.4). This is precedent for exactly the proposal's cross-boundary carry experiment, and shows carrying a naive Hebbian accumulator is prone to instability/blow-up absent explicit normalization or gating.
- Discusses two normalization schemes for the accumulator (attention normalization via an accumulator z that "always grows... may result in instability," vs. their preferred "sum normalization" dividing φ(k), φ(q) by the sum of their components) — a concrete, reusable technique for checking/controlling carry-state norm growth.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The Hebbian outer-product update central to this proposal's carry mechanism is formally identical to the fast-weight update of linear attention (Schlag et al., 2021), W_t = W_{t−1} + v_tk_tᵀ, tracing to Schmidhuber's original Fast Weight Programmers" (Methodology/Background).
- "Schlag et al. (2021) show analytically and empirically that purely additive (Hebbian) fast-weight updates have a hard capacity ceiling set by key dimensionality, and — more importantly for a non-error-correcting rule — cannot overwrite a stale key–value association once written, only add interference to it" (Discussion — explains flat carry / lack of compounding).
- "Prior work directly observed that carrying a purely Hebbian/summed fast-weight state across sequence-segment boundaries without careful normalization destabilizes the model (perplexity >1600 on WikiText-103), whereas an error-correcting delta-rule update carries stably and matches a much larger Transformer-XL memory at a fraction of the state size (Schlag et al., 2021)" (Methodology/Discussion — motivates this proposal's carry-stability check before trusting any gain, and is a citable precedent for why persistence is nontrivial under Hebbian updates specifically).
- "On WikiText-103, replacing a sum/Hebbian update with an error-correcting delta rule improved test perplexity by roughly 2–3 points at matched parameter count (Schlag et al., 2021), suggesting error-correction, not mere persistence, may be the more important ingredient for inference-time gains — a candidate explanation if this proposal's Hebbian-only carry shows no benefit" (Related Work/Discussion).
- "This proposal's carry is deliberately restricted to the pure Hebbian rule (no error correction), isolating the persistence question from the delta-rule's known capacity/overwriting advantages documented by Schlag et al. (2021)" (Methodology — justifies scope).

## Verification
- All equations (Eq. 1–29), the capacity-limit argument, the Setting 1/2 synthetic retrieval results (Fig. 2/3), Table 2 perplexity numbers (37.1/38.3/31.1... wait: verified as Linear Transformer sum = 37.1/38.3 small, 31.1/33.0 medium; Delta Network = 34.1/35.5 small, 29.7/31.5 medium), and Table 4 (Delta Net state size 0.13M, PPL 27.8/29.4; Transformer-XL at 2.10M state, PPL 26.4/27.4) were read directly from the PDF tables and text.
- The ">1600" perplexity figure for the sum rule without normalization when carried across segments is a direct quote read from the "Without truncating context" paragraph on the page before the Conclusion.
- Not independently re-derived: the full algebraic derivation of Eq. 23–25 from Eq. 20–22 (paper defers detail to Appendix A.1, which was not in the extracted page range) — I report the stated final update rule as given in the main text, which is unambiguous, but did not check the appendix derivation itself.

# FILE: LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding_importance.md

# LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding

**Citation:** Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li (Tsinghua University, Zhipu.AI, Institute of Automation, Chinese Academy of Sciences). arXiv:2308.14508v2 [cs.CL], last revised 19 Jun 2024.

## Important parts for my research
- 21 datasets across 6 task categories (single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic retrieval, code completion), bilingual English/Chinese, 4,750 test instances, avg length 6,711 words (English) / 13,386 characters (Chinese). Fully automated evaluation (F1, ROUGE-L, classification accuracy, exact-match accuracy, Edit Sim) — no manual annotation needed per model.
- LongBench-E: a length-balanced subset (0–4k, 4k–8k, 8k+ bins) constructed specifically to separate a model's long-context ability from its raw task-solving ability — a model with strong task skill but weak length-robustness shows a steep score decline across these bins even though it may score fine overall.
- Memorization-vs-understanding diagnostic (Table 6): rerun QA datasets with context withheld; the Δ score (with-context minus without-context) isolates genuine context use from pretraining memorization — flagged as important because Wikipedia-derived datasets (HotpotQA, 2WikiMultihopQA, MuSiQue) show surprisingly high no-context accuracy.
- Context-compression findings: retrieval-based (top-N chunk retrieval via embeddings, text-embedding-ada-002 best) and summarization-based compression only measurably help models with weak native long-context ability; they don't close the gap to models with strong long-context training — compression is "performance compensation," not a real substitute.
- Truncation protocol for inputs exceeding a model's max length: keep the front and end of the sequence, drop the middle (S_{1:L} → [S_{1:⌊M/2⌋}; S_{L−⌊M/2⌋−1:L}]) since instructions/questions are often at the edges.
- Key result: GPT-3.5-Turbo-16k leads overall but still degrades at longer lengths; fine-tuning/position-scaling for long context (ChatGLM2-6B-32k, LongChat-v1.5-7B-32k) gives large relative gains (62%, 19%) over their un-adapted counterparts.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology) The withhold-context Δ-score diagnostic is directly reusable: apply it to rule out that any carry-attributable perplexity gain on our held-out replayed document is base-model memorization rather than genuine persistence.
- (Methodology) LongBench-E's length-binned evaluation design is a template for checking whether carry gains specifically strengthen at later repeats/longer accumulated context rather than being a length-independent artifact.
- (Discussion) Their finding that context compression helps only weak-baseline models and never closes the gap to strong native long-context models is a useful parallel for framing why carry might help less than expected relative to a strong LoRA baseline.
- (Related Work) Establishes LongBench as a standard bilingual, multi-task long-context benchmark family, predating and complementing RULER/ZeroScrolls as an alternative to pure perplexity evaluation.

## Verification
- Dataset counts, task categories, truncation formula, and the retrieval/summarization compression findings were read directly from the PDF (pp. 1–8, Tables 1–6). No inferred claims beyond directly stated text.

# FILE: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens_importance.md

# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**Citation:** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang — Microsoft Research, arXiv:2402.13753v1 [cs.CL], 21 Feb 2024.

## Important parts for my research
- Extends LLaMA2-7B / Mistral-7B context from 4k to 2048k tokens purely via non-uniform RoPE rescaling — no architecture change, no fast-weight/memory mechanism. Two non-uniformities exploited: (1) different RoPE dimensions need different interpolation strength; (2) initial token positions (first n̂) should get less/no interpolation (echoes attention-sink findings).
- Evolutionary search over per-dimension rescale factors λ_i and threshold n̂ (Table 4 search space), with a monotonicity constraint (λ_i non-decreasing in dimension i, motivated by NTK theory) to prune the search space.
- Progressive extension: search+fine-tune to 256k, then a second non-fine-tuned search extends 8× further to 2048k without added training — "search generalizes across window sizes" is the key trick enabling million-token windows cheaply (1k fine-tuning steps total).
- Passkey retrieval ≥90% accuracy from 4k–2048k; standard 4k benchmarks (ARC, HellaSwag, MMLU, TruthfulQA) preserved after a final "short-context recovery" re-search.
- No claim of any parametric/weight update at inference time — this is a pure positional-embedding technique, entirely orthogonal to weight-carrying TTT approaches.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) "Positional-interpolation methods such as LongRoPE (Ding et al., 2024) extend effective context length without touching model weights, in contrast to our approach which carries information via weight updates (fast weights) rather than position encoding — the two are complementary, not competing."
- (Related Work) Non-uniform treatment of "recent"/boundary tokens (their n̂ initial-token exemption) parallels our within-document vs. cross-document boundary distinction — both literatures find that boundary positions/tokens need special handling.

## Verification
- All numbers (2048k window, 1k fine-tuning steps, ≥90% passkey accuracy, PG19/Proof-pile perplexities in Tables 1/5/6) directly read from text and tables, not inferred.
- No fast-weight, TTT, or carry mechanism appears anywhere in this paper — confirmed by full read; it is purely a RoPE-embedding search method.

# FILE: MEMORIZING TRANSFORMERS_importance.md

# Memorizing Transformers

**Citation:** Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy — Google, Published as a conference paper at ICLR 2022, arXiv:2203.08913v1 [cs.LG], 16 Mar 2022.

## Important parts for my research
- Mechanism: one attention layer near the top of the stack gets kNN-augmented attention into a large non-differentiable external memory of (key, value) pairs from past tokens; gate g=σ(b_g) combines memory-attention output V_m and local-context output V_c: V_a = V_m⊙g + V_c⊙(1-g) (Eq. 1–2). Keys/values are never backpropagated into — memory is filled from prior forward passes only, making it cheap to scale to 262K tokens.
- **Directly relevant to cross-document persistence gap:** "each subsequence in the batch comes from a different document, and thus requires a separate external memory, which is cleared at the start of each new document" (Sec 3.1, Batching). This is the same reset-at-document-boundary pattern as In-Place TTT — confirms that both retrieval-augmented and TTT/fast-weight long-context methods conventionally treat documents as memory-isolated units; nobody in this lineage carries state across independently-bounded documents either.
- Memory helps most on rare words: proper names, citations, references, function/variable names (Sec 4.6) — i.e., recall of *specific* prior tokens, not generalization. Case study on Isabelle: 8/10 sampled predictions of lemma names successfully retrieved the lemma's own definition body from memory.
- Distributional shift / staleness (Sec 3.2): because model parameters producing queries drift during training while stored keys/values don't, older memory entries become "stale"; normalizing keys/queries mitigates but doesn't eliminate this. Training instability appears with very large memories unless first pretrained small then fine-tuned to larger memory — relevant analogue for why accumulated fast-weight carry might degrade or plateau: staleness of a persisted update relative to a drifting readout.
- Perplexity improves monotonically with memory size up to 262K tokens (near-longest documents in arXiv); diminishing/no further gain expected past document length (Table 4, 5) — a clean numeric example of a compounding external representation, useful as *contrast* to our flat carry.
- Small Memorizing Transformer (8K memory) matches a vanilla Transformer with 5× more parameters (Fig. 1) — striking comparable-numbers claim for Related Work.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) "Prior long-context memory mechanisms, whether retrieval-based (Memorizing Transformers, Wu et al. 2022) or fast-weight-based (In-Place TTT), uniformly clear their memory/fast-weights at document boundaries; our carry ablation is, to our knowledge, the first to test what happens if this reset is removed."
- (Discussion — explains flat/non-compounding carry) Staleness argument: a persisted representation whose producing parameters keep drifting can decay in usefulness over repeats, a candidate explanation alongside "no error-correction" for flat carry.
- (Comparable numbers) 8K-memory model ≈ 5× larger vanilla Transformer in perplexity (Fig. 1); diminishing returns past ~262K memory ≈ document length (Table 5).

## Verification
- Gate equation, batching/clearing-per-document quote, staleness discussion, and the 8K-memory-vs-5×-params claim all directly read from Sections 3.1–3.2, 4.3–4.6 and Figure 1 captions. No inference.

# FILE: Mamba: Linear-Time Sequence Modeling with Selective State Spaces_importance.md

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Citation:** Albert Gu, Tri Dao — CMU / Princeton, arXiv:2312.00752v2 [cs.LG], 31 May 2024.

## Important parts for my research
- Core mechanism: makes SSM parameters (Δ, B, C) input-dependent ("selective"), breaking linear time-invariance (LTI) so the hidden state h_t can be written/erased content-adaptively rather than by fixed dynamics — direct architectural analogue to what a carried fast weight would need in order to be useful across boundaries.
- Theorem 1: with N=1, A=-1, B=1, selective SSM recurrence reduces to an RNN gate h_t=(1-g_t)h_{t-1}+g_t x_t, g_t=σ(Linear(x_t)) — Δ plays the role of a generalized gate; large Δ resets state and focuses on current input, small Δ persists state and ignores input. This is a clean formal vocabulary for "carry vs. reset" that maps onto our carried_decay / clip_tau knobs.
- **Boundary Resetting (Sec 3.5.2, directly relevant to explaining flat/non-compounding carry):** "Selective SSMs can reset their state at boundaries... these settings may occur artificially (e.g. packing documents together)... LTI models will bleed information between sequences" while selective models can filter it out. This is an explicit mechanistic account of state persistence vs. reset across document boundaries — the same problem our carry-reset ablation targets, but Mamba's selection mechanism can *choose* to reset while a Hebbian (non-selective, pure outer-product) fast-weight update has no such gating and thus cannot distinguish useful cross-document signal from noise.
- Ablations (Table 7): making Δ selective alone is the single largest driver of improvement (10.93→9.81 perplexity); adding selective B, C on top gives further but smaller gains (→8.71) — evidence that selectivity/gating, not mere presence of a recurrent update, is what makes an accumulating state useful. This is suggestive: an update rule without error-correction or selectivity (our Hebbian, no delta rule) plausibly explains flat carry.
- "Filtering Context": models without selection cannot ignore irrelevant history, so performance does not improve monotonically with context — cited explanation for why many sequence models don't benefit from more context, relevant to why naive carry may not compound.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains flat carry) Gu & Dao (2024)'s selection-mechanism argument: state updates lacking input-dependent selectivity/gating cannot filter relevant from irrelevant signal, offering a mechanistic account for why our pure-Hebbian (non-selective, non-error-correcting) carry stays flat rather than compounding.
- (Related Work) Boundary-resetting discussion directly parallels our document-boundary carry-vs-reset framing; Mamba treats it as an architectural choice (Δ→∞), we treat it as an empirical ablation on fast weights.
- (Methodology) Δ-as-gate formalization (Theorem 1) is a citable reference point for describing carry/decay hyperparameters in fast-weight terms.

## Verification
- Theorem 1 equation, boundary-resetting quote, and Table 7 ablation numbers checked directly against PDF text (pages 8–9, 16-17 of paper). No inference beyond direct reading.

# FILE: Meta-Learning in Neural Networks: A Survey_importance.md

# Meta-Learning in Neural Networks: A Survey

**Citation:** Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey — University of Edinburgh / Samsung AI Centre, arXiv:2004.05439v2 [cs.LG], 7 Nov 2020.

## Important parts for my research
- Formalizes meta-learning as bilevel optimization: outer loop learns meta-knowledge ω (e.g. initialization, optimizer, architecture), inner loop solves task-specific θ*(ω) given ω fixed (Eq. 5–6). TTT/fast-weight adaptation at inference time is a degenerate single-task instance of this inner loop, with the base pretrained weights playing the role of ω.
- Proposes a three-axis taxonomy: meta-representation ("what" is learned — initialization, optimizer, embedding, architecture, hyperparameters), meta-optimizer ("how" — gradient, RL, evolution), meta-objective ("why" — few/many-shot, fast adaptation vs asymptotic performance, online vs offline). Useful vocabulary for positioning session-persistent TTT: our ω is the base LM weights, our meta-representation is the fast-weight update itself, meta-objective is "fast per-document adaptation with a persistence-vs-reset choice."
- Distinguishes optimization-based (MAML-style, explicit inner-loop gradient steps) from feed-forward/black-box (amortized, single forward pass) from metric-based methods — TTT-Linear/MLP and our Hebbian carry are optimization-based in this taxonomy, since the "inner loop" is literal gradient/Hebbian weight updates on new data, but with none of it backpropagated through an outer meta-objective (unlike MAML).
- Continual Learning (CL) section (5.7): explicitly notes CL "seeks to accelerate learning of new tasks... without forgetting old tasks" and that meta-objectives for CL are an active but still nascent research area — situates our carry-across-documents evaluation as adjacent to continual learning but distinct (single model, no explicit task boundaries or replay).
- Meta-overfitting / meta-generalization (Sec 6) discussion: models can fail to generalize from meta-train to meta-test tasks, especially when the number of "tasks" is small — a cautionary framing applicable to our small number of held-out replay documents.
- "Task families" and narrow vs diverse task distributions (Sec 6) are flagged as a major open challenge — relevant to our domain-comparison research question (Q2), since meta-learning literature shows performance is highly sensitive to how diverse/narrow the task/document distribution is.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / framing) General bilevel meta-learning formalization (Hospedales et al. 2020) as the broader category TTT/fast-weight methods belong to, distinguishing our approach as optimization-based with no explicit outer meta-objective (unlike MAML-based approaches).
- (Discussion) Their meta-generalization / narrow-task-family caution supports treating a null/flat carry result as a plausible, well-precedented outcome rather than an implementation bug, given known difficulties of generalizing across diverse task/document distributions.

## Verification
- Taxonomy, bilevel formalization (Eq. 5–6), and continual-learning/meta-generalization sections read directly from Sections 2–3 and 5.7/6 of the PDF. No numeric results are claimed from this paper (survey, not empirical), so no numbers to fact-check.

# FILE: Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation_importance.md

# Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation

**Citation:** Zejiang Hou (Princeton), Julian Salazar, George Polovets (Amazon AWS AI) — arXiv:2207.03509v1 [cs.CL], 7 Jul 2022.

## Important parts for my research
- Proposes MLtD: explicit meta-learning (MAML-style bilevel, but initialized from a pretrained LM rather than from scratch) of a dynamic low-rank reparameterization (TARP) of weight matrices, W^l = Φ_1^l⊙W_0^l + Φ_2^l, plus a learned architecture controller (TAMS) that generates task-specific sublayer structure.
- **Dynamic TARP: Φ_j^l(x) = U_j^l Σ_j^l(x) V_j^{lT}**, where Σ is produced by a small input-conditioned MLP — i.e., the low-rank adaptation matrices themselves are functions of the current input, not static. This is a strictly more expressive alternative to LoRA's static low-rank decomposition (LoRA = special case with Φ_1=I).
- Key comparable-numbers result (Table 6, GLUE): dynamic TARP at 1% of parameters matches or beats full fine-tuning (86.7 vs 86.4 avg) and beats LoRA (85.3 avg) and MAM Adapter (85.7 avg) at similar budgets — direct evidence that a well-designed low-rank/meta-learned reparameterization can match full fine-tuning, useful comparator for our LoRA vs LoRA+TTT null result.
- "MAML after pretraining" finding (Sec 2, Fig. 4): MAML trained from a random init overfits and underperforms plain fine-tuning, but MAML initialized from a pretrained/multitask model reliably outperforms both fine-tuning alone and pretraining alone — meta-learning only pays off on top of a good pretrained initialization, not from scratch. Relevant caution: any TTT/fast-weight gain we hope to see is contingent on interacting well with pretrained weights, not automatic.
- Table 5 ablation: their "dynamic" TARP beats "bilinear" (≈static LoRA-like) and "Kronecker" decompositions on every generative metric — evidence that input-dependence/dynamism in the adaptation itself materially helps, congruent with our Hebbian (input-dependent but error-uncorrected) approach sitting somewhere between static LoRA and full dynamic reparameterization.
- Reports that meta-learning explicitly requires "the same labeled data ... for both Pretrain and MAML" to see gains (no extra data) — meta-learning here is a training-procedure change, not a data-scale change, which matches our own approach of holding pretraining data fixed and varying only the adaptation mechanism.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) MLtD's dynamic low-rank TARP (Hou et al., 2022) as the closest prior work showing a meta-learned, input-conditioned weight reparameterization can match/exceed full fine-tuning and static LoRA — positions LoRA as a degenerate/static case of a broader family our carried fast weights also belong to.
- (Discussion — framing a null result) Their finding that MAML underperforms plain fine-tuning without a good pretrained init is a precedent for adaptation mechanisms not helping, or even hurting, under certain initialization/training regimes — supports treating "LoRA alone beats LoRA+TTT" as a legitimate, precedented outcome rather than an anomaly.
- (Methodology) TARP's dynamic low-rank formulation (Eq. 3) is a citable technical precedent for input-conditioned adaptation weights, relevant if describing our carry mechanism's relationship to LoRA-style decompositions.

## Verification
- TARP equations, GLUE Table 6 numbers, and the "MAML after pretraining" claim (Fig. 4, Sec 4.1) checked directly against PDF text and tables. No inference beyond direct reading.

# FILE: Mind the Gap: Assessing Temporal Generalization in Neural Language Models_importance.md

# Mind the Gap: Assessing Temporal Generalization in Neural Language Models

**Citation:** Angeliki Lazaridou, Adhiguna Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liška, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d'Autume, Tomas Kocisky, Sebastian Ruder, Dani Yogatama, Kris Cao, Susannah Young, Phil Blunsom. DeepMind, London. NeurIPS 2021 (35th Conference on Neural Information Processing Systems).

## Important parts for my research
- Core setup: trains 18-layer, 287M-param Transformer-XL (context 1024, memory cache 384 train/1600 test) on WMT News, CustomNews, ArXiv, comparing a TIME-STRATIFIED model (trained only on pre-2018 data) vs a CONTROL model (same size, trained on data overlapping the 2018-2019 test period). TIME-STRATIFIED underperforms CONTROL by up to 16% relative perplexity (WMT +6.34%, CustomNews +16.04%, ArXiv +7.90%; Table 2), and degradation *increases* monotonically the further test data is from the training cutoff (Fig. 1, Appendix A.2: perplexity rises from 24.65 at 0.5yr gap to 32.79 at 5.5yr gap).
- Scaling up model size (287M→448M, +60% params) does NOT flatten the degradation slope (t-test on slopes, p>0.05) — a smaller, more up-to-date 287M CONTROL model beats a larger 448M stale model by the end of the test period. Directly useful: model capacity alone does not substitute for adaptation/persistence.
- Degradation is concentrated in proper nouns and common nouns (named entities), and in politics/sports topics, far more than function words or "lifestyle" topics (Fig. 2 POS/topic breakdown).
- EMERGING NEW WORDS analysis (words absent or 5x rarer in training vs test, min. 50 test occurrences; 287 words, 87,636 mentions): overall perplexity on these words is 109.73 vs 22.45 baseline — ~5x worse (Table 3). First-occurrence ppl is extremely high (694.95); second-occurrence ppl drops sharply to 75.95 **only if** the first occurrence is still inside the Transformer-XL memory window — if it fell outside the memory cache, second-occurrence ppl stays catastrophic (2719.25). This isolates in-context copying (attention over recent tokens) from true parametric knowledge update, and shows attention alone cannot substitute for a persistent weight update once information exits the context window.
- Dynamic evaluation (defined formally, Appendix B, Eq. 1-2): online per-document gradient descent, θ_{t+1} = θ_t − α∇_θ ℓ(D^(t); θ_t), with K gradient steps per document (K tuned on validation) and updates computed on contiguous token batches (e.g., 512 tokens). Tuned learning rates: WMT 5e-5, CustomNews 5e-4, ArXiv 1e-3.
- Dynamic eval significantly flattens (but does not eliminate) the upward degradation slope for ArXiv and WMT (p<0.05); for ArXiv only ~1 week of streamed data is needed for the dynamically-updated model to overtake the static CONTROL model. Emerging-word perplexity falls 39.62% (109.73→66.26) under dynamic eval, vs only 1.25% (22.45→22.17) overall — i.e., gains concentrate exactly where staleness is worst, not uniformly.
- **Catastrophic forgetting result (Appendix B.1, Fig. 9), directly relevant to explaining a flat/non-compounding carry**: after dynamic evaluation updates a model forward through 2018-2019, its perplexity on held-out data from progressively earlier years (2017 back to 2007) rises with a gentle but clear upward slope — the model that adapts to "now" quietly degrades on "then." This is presented as a real plasticity/stability trade-off from *unconstrained* full-parameter gradient updates, and the paper cites McCloskey & Cohen (1989) and Kirkpatrick et al. (2017, EWC) directly in this context.
- Lightweight update ablation (Table 4): updating only bias terms (198K params, BitFit-style, cf. Ben-Zaken et al. 2021) achieves perplexity nearly identical to updating all parameters (e.g., ArXiv 21.24 vs 20.98 all-params, both far better than no-dynamic-eval 23.07); embeddings-only updating is a weaker middle ground (21.21-22.32). Suggests a small, targeted parameter subset can capture most of the adaptation benefit — relevant precedent for a small hebbian fast-weight module doing a comparable job to full fine-tuning.
- Even with dynamic evaluation, the paper explicitly states it "does not fully solve temporal generalization," as evidenced by the "prevailing, albeit gentler" upward slopes that remain (§6) — inference-time adaptation is reported as a partial, not complete, fix.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Static train/test splits systematically overestimate language model performance relative to realistic future-data evaluation, by up to 16% relative perplexity, and this gap widens monotonically with the training-test time gap" (Related Work / Motivation — establishes that temporal/session staleness is a real, measured problem, not a contrived one).
- "Increasing model size alone does not mitigate temporal performance degradation; a 60%-larger model shows no significant change in degradation slope" (Related Work / Discussion — supports framing session-persistence as addressing something scale doesn't fix).
- "Full-parameter dynamic evaluation (per-document online gradient descent) narrows but does not close the temporal generalization gap, and produces measurable catastrophic forgetting of older data even as it improves fit to recent data" (Related Work — direct precedent for a partial/null-leaning persistence result being a legitimate, citable finding; also Discussion — motivates why this proposal's carry-stability check exists).
- "Dynamic evaluation improves perplexity most on out-of-distribution/emerging content and least on already-well-modeled content, showing inference-time adaptation gains are unevenly distributed rather than uniform" (Discussion — relevant framing for why a proposal's Q2 domain-dependence question is worth asking).
- "Lightweight parameter updates (bias-only, ~0.07% of model parameters) recover nearly all the benefit of updating all parameters during online adaptation" (Methodology — precedent that small/targeted fast-weight-style updates can match full updates).

## Verification
- Directly re-checked in PDF text: perplexity table (Table 2: WMT 22.45/21.11, CustomNews 21.33/18.38, ArXiv 23.07/21.38, with 6.34%/16.04%/7.90% relative deltas); Table 3 emerging-word numbers (109.73→66.26, 694.95, 75.95→44.21, 2719.25→1430.34, 39.62% and 1.25% reduction figures stated explicitly in body text); Table 4 bias/embedding/all-parameter perplexities; model size 287M/448M and "no significant effect" p>0.05 claim (§4); Appendix B.1 catastrophic-forgetting description and its explicit citation of McCloskey & Cohen 1989 and Kirkpatrick et al. 2017.
- Not independently verified beyond what's stated: the exact TPU-hour compute figure (186,000 TPU hours) and the "~1 week of data" ArXiv dynamic-eval claim are taken directly from stated text, not recomputed — no reason to doubt them but flagging as author-reported rather than cross-checked against a plot.

# FILE: Model-agnostic meta-learning for fast adaptation of deep networks_importance.md

# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

**Citation:** Chelsea Finn, Pieter Abbeel, Sergey Levine — Proceedings of the 34th ICML, PMLR 70, 2017 (arXiv:1703.03400v3, 18 Jul 2017).

## Important parts for my research
- MAML meta-learns an initialization θ such that a small number of gradient steps on a new task's small dataset produces good generalization: inner update θ'_i = θ − α∇_θ L_{T_i}(f_θ) (one or more steps); outer/meta update θ ← θ − β∇_θ Σ_i L_{T_i}(f_{θ'_i}) — evaluated at post-adaptation parameters (Eq. 1, Algorithm 1).
- Framed as maximizing sensitivity of new-task loss to θ: MAML finds a point in parameter space where small, task-specific gradient steps produce large improvements (dynamical-systems view).
- Requires backprop through the gradient step (second derivatives / Hessian-vector products). A first-order approximation (drop second-order term) gives nearly identical MiniImageNet accuracy (48.07±1.75% vs 48.70±1.84% 1-shot) with ~33% less compute — most of the benefit comes from gradients evaluated at post-update θ', not from the second-order correction.
- Continues to improve with additional gradient steps at meta-test time without overfitting to the small K-shot set (sinusoid regression, Fig. 3) — contrasts with a pretrained-then-fine-tuned baseline, which does not extrapolate.
- Multi-task parameter averaging (500 separately trained models) underperforms output-space pretraining-on-all-tasks, which itself underperforms MAML (5-shot sinusoid MSE: multi-task 2.19–3.85 vs pretrain-on-all 2.19–2.41 vs MAML 0.35–0.67 across gradient-step counts) — parameter averaging destroys per-task solutions.
- Results: Omniglot 5-way 1-shot 98.7±0.4%, 5-shot 99.9±0.1%; MiniImageNet 5-way 1-shot 48.70±1.84%, 5-shot 63.11±0.92%. RL (half-cheetah/ant, TRPO): MAML adapts velocity/direction in 1–2 gradient steps; pretraining is sometimes worse than random initialization.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) MAML as the canonical gradient-based meta-learning approach to fast within-task adaptation — contrast against our fast-weight/carry approach: MAML meta-learns an initialization exploited by ordinary SGD steps at test time (no persistent parametric memory carried forward across independent tasks/documents), whereas our carry persists an actual fast-weight state across independently-bounded documents.
- (Methodology) MAML's "continues to improve with more gradient steps, no overfitting" diagnostic (Fig. 3) is a useful contrast case for our finding that carry does *not* grow over repeats — MAML's compounding improvement under repeated gradient steps is the positive control our null result lacks.
- (Discussion — framing null result) The first-order-approximation finding — most of MAML's benefit comes from gradient direction at the adapted point, not the second-order meta-gradient term — is a precedent for isolating which mechanism actually drives an adaptation method's gains, relevant to isolating carry's contribution via the carry-reset baseline.

## Verification
- Equations 1–4, Algorithms 1–3, and all numerical results (Table 1, Tables 2–5, Figs. 2–7) were read directly from the PDF. No inferred/uncited figures used.

# FILE: On the Dynamic Adaptation of Stochastic Language Models_importance.md

# On the Dynamic Adaptation of Stochastic Language Models

**Citation:** Reinhard Kneser, Volker Steinbiss, Philips GmbH Forschungslaboratorien, Aachen, Germany. IEEE ICASSP 1993 (published as pp. II-586–II-589, 0-7803-0946-4/93).

## Important parts for my research
- Mechanism: not a gradient-based fast weight at all — it's a mixture-of-experts style linear interpolation. K=15+1 pre-trained bigram language models (one per text category: newspapers, scientific writing, fiction, etc., plus one general model on all data) are combined as P(w_n|w_{n-1}) = Σ_k λ_k P_k(w_n|w_{n-1}) (Eq. 1), with Σλ_k=1, 0≤λ_k≤1.
- The "adaptation" is entirely in re-estimating the mixture weight vector λ online: λ is re-estimated at each new word position n via a few iterations of forward-backward (EM), maximizing likelihood of the preceding L words (window length), with an explicit update formula (Eq. 2): λ̂_k = (1/L) Σ_{m=n-L}^{n-1} [λ_k P_k(w_m|w_{m-1}) / Σ_i λ_i P_i(w_m|w_{m-1})]. This is a soft, EM-style responsibility-weighted running average — not an outer-product/Hebbian accumulator and not gradient descent on model weights. No error-correction term beyond the EM re-normalization at each step.
- Best result: L=400 word window gives 480.7 test perplexity vs 532.1 baseline (no adaptation) — a 10% relative improvement (stated explicitly in abstract and §3.2). Cache-augmented baseline (410.9) is also improved by the same adaptation method to 384.0, i.e., adaptation and caching are complementary, roughly additive gains.
- λ does NOT need to be reset to uniform between categories to work — it is continuously re-estimated word-by-word over running text spanning multiple category shifts (Fig. 1, ~70,000 words across 15 shifting categories), and the paper shows λ_k spikes toward the correct category's component within each shift, then decays back down — i.e., the state naturally tracks local domain but doesn't accumulate/compound over the long run; it's tracking-only, not memory-building.
- Window length L is a bias-variance knob: too short (L=25, ppl 533.8) barely beats baseline; too long is worse than L≈400 for standard specific models (L=50000 → 492.8, worse than baseline in cache-off case is not true but is clearly worse than the L=400 optimum) — sharp optimum, not monotonic. With additional smoothing of specific models the optimal L is shorter (peaks earlier, plateaus).
- Fixed-but-optimal-per-category λ (Table 1b, oracle knowing the true category) reaches only 487.6 — worse than the dynamically varying λ (480.7), meaning within-category variation also matters, not just between-category switching — the adaptive scheme captures finer structure than category identity alone.
- Hard-decision variant (choosing the single best specific model rather than soft mixture) performs slightly worse (494.2) than the soft EM mixture (480.7), evidence that soft/graded mixing outperforms winner-take-all adaptation in this setting.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Even a simple, gradient-free adaptation mechanism — dynamically re-weighting a fixed set of domain-specific sub-models via online EM — recovers a 10% relative perplexity improvement over a static baseline, establishing that inference-time domain-tracking has been a viable idea since at least 1993" (Related Work — historical precedent/positioning for dynamic/session adaptation broadly).
- "This early dynamic-adaptation scheme tracks the current local domain via a running weighted average that decays back down after a domain shift rather than accumulating persistent structure across shifts" (Discussion — precedent for adaptation mechanisms that are inherently tracking/local rather than compounding, relevant to explaining a flat, non-growing carry).
- "A soft mixture-based adaptation outperforms a hard winner-take-all variant of the same mechanism, suggesting graded/soft state updates are preferable to discrete ones for inference-time adaptation" (Methodology/Discussion).

## Verification
- All numbers (532.1, 410.9, 487.6/387.7, 480.7/384.0, 482.0/384.2, 494.2/394.0, 10% improvement claim, L=400 with ppl 480.7 vs 482.4, Table 2 full L-sweep 533.8→511.2, Table 1e ~482.0) checked directly against Table 1 and Table 2 text on pages 2–3.
- Eq. 1 and Eq. 2 read directly from page 1–2; description above paraphrases them faithfully (soft EM/forward-backward re-estimation, not gradient descent).
- Fig. 1 λ-spike-and-decay behavior across the 15 categories is a visual/qualitative read of the plot on page 4, not a numeric claim — flagged as a qualitative interpretation rather than a stated quantitative result in the text.

# FILE: Optimization as a model for few-shot learning_importance.md

# Optimization as a Model for Few-Shot Learning

**Citation:** Sachin Ravi, Hugo Larochelle — ICLR 2017 (Princeton University & Twitter; Google DeepMind collaboration per slide credits). Note: the source PDF is a conference-talk slide deck presented by Sachin Ravi, not the typeset paper; all content below is as shown on the slides, which match the published Ravi & Larochelle ICLR 2017 paper of this title.

## Important parts for my research
- Casts learning-to-learn as learning a training algorithm: a meta-learner LSTM produces the parameter trajectory of a learner model M. Gradient descent θ_t = θ_{t-1} − α_t∇_{θ_{t-1}}L_t is reparametrized as an LSTM cell update: c_t = f_t⊙c_{t-1} + i_t⊙c̃_t, where cell state c_t = learner parameters θ_t, candidate c̃_t = the negative gradient, and i_t, f_t are learned sigmoid gates (each a function of ∇L_t, L_t, θ_{t-1}, and the previous gate value).
- The **forget gate f_t is a learned, per-step, per-parameter decay on the carried state** — a direct generalization of a fixed weight-decay/learning-rate schedule into something the network adapts online. The **input gate i_t** learns an adaptive step size on the new gradient. Standard SGD is the special case i_t=α (constant), f_t=1.
- Meta-learner also learns c_0, i.e., learns M's initialization jointly with the update rule.
- Training: meta-learner LSTM parameters are shared across all of M's parameter dimensions (treated as one large "batch"); trained end-to-end via backprop through the unrolled adaptation trajectory to minimize test-set loss after T inner steps (Algorithm 1), while gradients are not backpropagated through the LSTM's own gradient/loss inputs.
- MiniImageNet 5-way results: 1-shot 43.44±0.77%, 5-shot 60.60±0.71% — beats Matching Networks FCE (43.56/55.31) on 5-shot, competitive on 1-shot; outperforms plain fine-tuning and nearest-neighbor baselines substantially.
- Learned input-gate values (Fig., slide 15) differ qualitatively between 1-shot and 5-shot regimes — with more data per class, gates saturate near 1 across nearly all steps/layers (aggressive, uniform updates), while 1-shot gates show more per-layer, per-step variation (more conservative, selective updates).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains flat/non-compounding carry) The explicit contrast between a *learned, adaptive* forget/decay gate here and a *fixed, hand-set* decay coefficient in our carry mechanism: this paper's core finding is that naive fixed-rate parameter carrying (plain SGD) is a special, unstable case, and that good sequential parameter accumulation requires per-step learned gating — a fixed carried_decay (as in our Hebbian carry) lacks this adaptivity, which may explain why carry neither compounds nor blows up but instead sits flat (over-conservative decay) rather than accumulating like a tuned forget gate would.
- (Related Work) An early framing of "the update rule as something to be learned" applicable when contrasting our fixed Hebbian outer-product update rule (no error correction, no learned gating) against learned alternatives.
- (Methodology) The gate-visualization diagnostic (differing gate behavior 1-shot vs 5-shot) as a template for interpreting when/why a persistence mechanism engages differently across data regimes — analogous to how carry behavior might differ by domain (Q2).

## Verification
- All equations, the algorithm box, and the MiniImageNet table were read directly off the slide images. Because the source is a slide deck (not the typeset paper), exact wording/phrasing is my paraphrase of slide bullets, not the paper's original prose — numbers and equations themselves were read directly and match slide content exactly, not inferred.

# FILE: Overcoming catastrophic forgetting in neural networks_importance.md

# Overcoming catastrophic forgetting in neural networks

**Citation:** Kirkpatrick, Pascanu, Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis, Clopath, Kumaran, Hadsell — arXiv:1612.00796v2, 25 Jan 2017 (DeepMind). [PDF header shows arXiv only; the paper was later published in PNAS 2017, not stated on this PDF's title page.]

## Important parts for my research
- Elastic Weight Consolidation (EWC): loss L(θ) = L_B(θ) + Σ_i (λ/2)·F_i·(θ_i − θ*_{A,i})², a quadratic penalty anchoring each parameter to its old (task-A) value, scaled by its importance F_i (Eq. 3).
- F is the diagonal of the Fisher information matrix, used as a Laplace approximation to the posterior over θ given past data: log p(θ|D) = log p(D_B|θ) + log p(θ|D_A) − log p(D_B) — i.e., the old task's posterior becomes the new task's prior. F is cheap (first-order derivatives only), guaranteed PSD, and ≈ second derivative of the loss surface near the optimum.
- Sum of two quadratic penalties is itself quadratic, so EWC composes across >2 sequential tasks without added terms.
- Permuted-MNIST: plain SGD catastrophically forgets task A upon training B; L2 (uniform) regularization protects A but blocks learning B (protects all weights equally); EWC learns B while retaining A, scaling to many more tasks than dropout-regularized SGD (~2 permutations).
- Perturbation validation (Fig. 3C): perturbing weights along the inverse-Fisher direction harms performance far less than uniform perturbation, confirming diagonal-Fisher importance estimates are meaningful — but perturbing in the Fisher's "nullspace" (nominally unimportant weights) hurts just as much, indicating the point-estimate diagonal Fisher underestimates parameter uncertainty. Authors flag this as the method's chief limitation.
- Atari/DQN continual-RL experiment: EWC penalty (Fisher multiplier 400) applied only after 20M frames per task; task inferred via an HMM-based generative "forget-me-not" module. EWC lifts total human-normalized score well above plain SGD (which collapses toward 0 after task switches) but stays below separately trained single-task DQNs.
- Runtime is linear in #parameters and #training examples, unlike prior curvature-approximation approaches (French & Chater 2002 recomputes curvature per sample; ELLA needs matrix inversion of size #params).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains carry-stability rationale) EWC's core diagnosis: catastrophic forgetting is unconstrained parameter drift with no importance-weighted anchor to the previously good solution; the quadratic penalty exists precisely to bound this drift.
- (Discussion — framing flat/non-compounding carry) Contrast our Hebbian-only, pure-outer-product fast-weight carry (no error correction, no per-weight importance weighting, no explicit anchor to a prior solution) against EWC's explicitly importance-weighted constraint — motivates why an unconstrained carry might either drift destructively or plateau/self-cancel rather than compound.
- (Related Work) EWC as the canonical solution to catastrophic forgetting via synaptic consolidation, contrasted with our carry-stability check as an empirical (not penalty-based) guard against the same failure mode.
- (Methodology) Fisher-based importance weighting and perturbation-sensitivity testing (Fig. 3C) as a template for validating whether a proposed "important"/protected direction in parameter space actually matters.

## Verification
- Equation 3, the Bayesian posterior derivation (Eq. 1–2), MNIST/Atari results, and the Fisher-overlap/perturbation findings were all read directly from the PDF text (pages 2–7). The PNAS venue is common knowledge but not stated on this PDF's title page, so citation above reports only what's shown (arXiv, DeepMind, 2017).

# FILE: Parallelizing Linear Transformers with the Delta Rule over Sequence Length_importance.md

# Parallelizing Linear Transformers with the Delta Rule over Sequence Length

**Citation:** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim (MIT, Soochow University, MIT-IBM Watson AI Lab), NeurIPS 2024 (38th Conference on Neural Information Processing Systems); arXiv:2406.06484v6, 15 Jan 2025.

## Important parts for my research
- Vanilla linear attention recurrence is purely additive/outer-product: **S_t = S_{t-1} + v_t k_t^T**, o_t = S_t q_t. This is explicitly called a "Hebbian-like update rule" with "limited memory capacity" (citing Hopfield-network literature) — this is structurally the same accumulator as our active `hebbian` update rule.
- Failure mode named explicitly: purely additive updates cause "key collisions" once sequence length L > state dimension d, because the rule has no mechanism to deallocate/remove old key–value associations — new writes just pile on top of old ones instead of correcting them.
- DeltaNet's delta rule: **S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T**, β_t = σ(W_β x_t) ∈ (0,1) a data-dependent "writing strength." Equivalent retrieve/interpolate view: v_t^old = S_{t-1}k_t (retrieve with current key); v_t^new = β_t v_t + (1−β_t) v_t^old (interpolate old vs. new value); then S_t = S_{t-1} − v_t^old k_t^T + v_t^new k_t^T (remove old association, write new one).
- **This is one step of online SGD on a regression loss**: L_t(S) = ½‖Sk_t − v_t‖², so S_t = S_{t-1} − β_t∇L(S_{t-1}) = S_{t-1} − β_t(S_{t-1}k_t − v_t)k_t^T. This is the Widrow-Hoff/LMS rule. The update magnitude is proportional to the **prediction error** (S_{t-1}k_t − v_t) — i.e., it is error-correcting.
- Direct contrast (their footnote 3): vanilla linear attention's implicit loss is linear, L_t = −⟨Sk_t, v_t⟩, whose **gradient is constant regardless of error magnitude** — "this gradient behavior provides an intuitive explanation for why linear transformers underperform DeltaNet in in-context retrieval tasks." This is the precise mechanistic reason a pure-Hebbian accumulator can't self-correct or prune stale state, unlike the delta rule.
- Eigenvalue/stability analysis of the delta-rule transition matrix (I − β_t k_t k_t^T): eigenvalues are 1 (multiplicity d−1) and 1−β_t‖k_t‖² (multiplicity 1). With L2-normalized keys (‖k_t‖₂ ≤ 1) this guarantees 0 ≤ 1−β_t‖k_t‖₂ ≤ 1 — the transition matrix's eigenvalues stay bounded in [0,1], so the state provably cannot blow up. When β_t=1 the transition becomes an exact projection matrix: it erases information in one subspace while exactly preserving the other d−1 subspaces ("targeted forgetting"), not accumulation.
- DeltaNet has **worse length generalization** than GLA/Mamba (which extrapolate beyond training length); authors speculate this is because DeltaNet lacks an explicit decay factor in its recurrence — Gated DeltaNet (cited, Yang et al. 2025) adds a gating/decay term α_t: S_t = S_{t-1}(α_t(I − β_t k_tk_t^T)) + β_t v_t k_t^T to fix this.
- DeltaNet also shows **poorer state-size scalability** at larger scale (1.3B): underperforms GLA on recall-intensive tasks there, though at matched state size (340M scale) it beats GLA on SWDE/SQuAD/FDA recall tasks. Suggests error-correction alone isn't sufficient if state capacity per size is limited.
- Language modeling numbers (1.3B params / 100B tokens, SlimPajama subset, Mistral tokenizer): DeltaNet Wiki ppl 16.87 vs. Mamba 17.22 vs. GLA(w/ conv) 17.15 vs. Transformer++ 16.85; DeltaNet average zero-shot accuracy 51.6% (w/ conv) vs. GLA 51.6%, Mamba 50.4%, Transformer++ 50.9%.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A purely additive/Hebbian-style fast-weight update (S_t = S_{t-1} + v_tk_t^T) has no mechanism to correct or remove prior associations, unlike the delta rule's error-driven update S_t = S_{t-1}(I−β_tk_tk_t^T)+β_tv_tk_t^T, which is a single step of online least-squares regression (Yang et al., 2024)." (Discussion — explains flat carry / why hebbian shows no compounding gain).
- "Prior work frames the delta rule's advantage over additive linear-attention updates precisely as error-correction: the additive rule's implicit loss has constant gradient regardless of prediction error, while the delta rule's gradient scales with the error (Yang et al., 2024)." (Discussion — mechanistic explanation for null Q1 result).
- "Yang et al. (2024) show the delta-rule transition matrix (I−β_tk_tk_t^T) has eigenvalues bounded in [0,1] under L2-normalized keys, guaranteeing state stability; this contrasts with a pure Hebbian accumulator, which has no analogous built-in bound and can only grow or saturate." (Methodology — motivates our carry-stability/state-norm check before trusting any gain).
- "DeltaNet itself was found to generalize poorly beyond training sequence length, an effect the authors attribute to the rule's lack of an explicit decay term (Yang et al., 2024), later addressed by Gated DeltaNet's added decay factor." (Related Work — positions our hebbian-only, no-decay carry as inheriting a known limitation class).
- "Yang et al. (2024) report the delta rule outperforms the purely additive linear-attention rule on recall-intensive real-world tasks (SWDE, SQuAD, FDA) at matched state size, evidence that error-correction — absent in our active hebbian update — materially affects what a fast-weight state retains." (Related Work / Discussion).

## Verification
- All equations (linear attention recurrence, DeltaNet update, SGD-loss derivation, retrieve/interpolate form, eigenvalue claim, Gated DeltaNet form) verified directly against the PDF text (§2.1–2.2, §5.1, Table 2, footnote 3, and the "feature map and normalization" paragraph in §3.3).
- Numbers in Table 1 (Wiki ppl, avg accuracy for 1.3B models) verified directly from the table on p.8.
- The claim that DeltaNet's poor length generalization is due to lack of decay is explicitly stated as the authors' own speculation ("We speculate that this is because...") in §5.3 — flagged as their hypothesis, not a proven fact, and reported as such above.

# FILE: REFORMER: THE EFFICIENT TRANSFORMER_importance.md

# Reformer: The Efficient Transformer

**Citation:** Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya, 2020, ICLR (Published as a conference paper at ICLR 2020; arXiv:2001.04451)

## Important parts for my research
- Two independent efficiency tricks, not related to fast weights: (1) LSH attention replaces O(L²) dot-product attention with O(L log L) via angular locality-sensitive hashing — queries only attend to keys in the same hash bucket; hash h(x)=argmax([xR; -xR]) for random rotation matrix R. Multi-round hashing (n_rounds hashes) trades compute for accuracy: 8 hashes ≈ full-attention accuracy on a synthetic duplication task (Table 2: LSH-8 reaches 94.8%/99.9%/99.9%/99.9% across train/eval settings vs full attention).
- (2) Reversible residual layers (from RevNets, Gomez et al. 2017): paired streams y1=x1+F(x2), y2=x2+G(y1), with F=attention, G=feed-forward; activations recomputed on backward pass instead of stored, removing the N-layer memory multiplier. Combined with chunked feed-forward computation, memory becomes independent of both depth and d_ff.
- Shared-QK attention (k_j = q_j/||q_j||) needed to make LSH bucketing meaningful; empirically shown not to hurt performance (enwik8/imagenet64 loss curves nearly identical to separate Q/K).
- Results: 12-layer Reformer reaches 1.19 bits/dim on enwik8 in 20K steps (1.05 with more tuning); processes 64K-token sequences on enwik8 and 12K on imagenet64 on a single accelerator; LSH attention speed stays flat with sequence length while full attention grows linearly (Figure 5).
- Purely an inference/training-efficiency architecture — no test-time parameter updates, no persistence across inputs, no memory carried between documents. All efficiency comes from approximating a fixed, static attention computation.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Reformer represents the efficient-long-context-via-approximation family (LSH attention, reversible layers) — orthogonal to and non-competing with TTT/carry approaches: it reduces the cost of processing a long context but does not persist any adapted state across documents or sessions, unlike In-Place TTT or our session-persistent carry.
- (Related Work) Useful contrast point when framing "how prior work achieves long-context efficiency without weight updates" — Reformer's O(L log L) attention is a pure architectural/compute answer to the same length problem that TTT/fast-weight methods answer via parameter adaptation.

## Verification
- All complexity claims (O(L²)→O(L log L)), the shared-QK result, reversible-layer equations (Eq. 7–9), and the enwik8/imagenet64 bits/dim numbers were read directly from the PDF (Tables 1–4, Figures 3–5, Section 2–3). No inferred figures.

# FILE: RULER: What's the Real Context Size of Your Long-Context Language Models?_importance.md

# RULER: What's the Real Context Size of Your Long-Context Language Models?

**Citation:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (2024), NVIDIA — Published as a conference paper at COLM 2024 (arXiv:2404.06654v3 [cs.CL])

## Important parts for my research
- RULER has four task categories, 13 tasks total (Table 2), each auto-generated/synthetic with controllable length and complexity:
  - **Retrieval (NIAH variants)**: Single NIAH (S-NIAH, vanilla needle-in-haystack: word/number/UUID key-value pair in Paul Graham essay haystack or repeated noise sentences), Multi-keys NIAH (MK-NIAH, multiple needles inserted, only one queried — others are hard distractors), Multi-values NIAH (MV-NIAH, multiple needles share one key, all values must be retrieved), Multi-queries NIAH (MQ-NIAH, multiple needles with distinct keys, all must be retrieved; same setup as Arora et al. 2024's multi-query associative recall).
  - **Multi-hop Tracing**: Variable Tracking (VT) — a variable X1 is bound to a value, then a linear chain of assignments (X2=X1, X3=X2, ...) is inserted at various positions; model must return all variable names bound to the original value. Emulates minimal coreference chain resolution. Complexity scales via number of hops or number of chains.
  - **Aggregation**: Common Words Extraction (CWE) — words sampled from discrete uniform distributions, task is to find the K most common words (K = number of common words); Frequent Words Extraction (FWE) — words sampled from a Zeta distribution (Zipf's Law), task is to return top-K frequent words (K fixed at 3, since higher K degrades performance even at short context).
  - **Question Answering (QA)**: existing short-context QA datasets (SQuAD, HotpotQA) with distractor paragraphs inserted around the golden paragraph to simulate long-context — a "real-world adaptation of NIAH" where the question is the query, golden paragraphs are the needles, distractor paragraphs are the haystack.
- Evaluated 17 long-context LLMs (15 open-source + Gemini-1.5-Pro + GPT-4), context lengths 4K–128K, 500 examples per task per length, vLLM inference, greedy decoding, recall-based accuracy against target output (with answer-prefix prompting to prevent refusals).
- "Effective context length" defined via threshold grading: the max length at which performance still exceeds Llama2-7B's own performance at 4K (85.6%) — most models fall well below their claimed context length by this measure (e.g., only half of models handle 32K satisfactorily despite claiming ≥32K).
- Key failure modes at scale (Yi-34B-200K deep-dive, Section 5): non-robustness to needle type (near-perfect on word-number pairs, degrades sharply on UUID retrieval), failure to ignore distractors (~40-point drop at 256K when haystack is full of distractor needles, #K=FULL), return of incomplete information (accuracy drops ~15 points going from 1 to 8 queries in MQ-NIAH), tendency to copy from context verbatim (>80% of CWE output at 128K is copied from the one-shot example, near 0% at short context), unreliable multi-hop tracking (degrades with both more hops and more chains), and hallucination/reliance on parametric knowledge in QA (performance converges toward the no-context baseline as context grows, sometimes ignoring correct context entirely).
- Model-size ablation: Yi-34B-200k >> Yi-9B-200k > Yi-6B-200k trained on identical data/context length — larger model size helps at both short (4K) length and rate of degradation.
- **Architecture finding directly relevant to this proposal**: RWKV-v5-7B and Mamba-2.8B-slimpj (non-Transformer / linear-recurrent architectures) both "lag behind Transformer [Llama2-7B baseline] by large margins" on RULER, and both "demonstrate significant degradation when extending context size to 8K" (Fig. 4, right panel) — direct evidence that non-Transformer recurrent/state-based architectures underperform on RULER-style long-context tasks, relevant to positioning a fast-weight/TTT approach (which is also a recurrent-state mechanism) against this baseline gap.
- Explicit limitation admitted by the authors: RULER's variable tracking and frequent/common words tasks are proxies for real long-context tasks (coreference, summarization) but their correlation with realistic task performance is *not verified* — the paper says RULER should be used as a "convenient behavioral check," not preferred over more realistic benchmarks.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "RULER (Hsieh et al., 2024) extends the vanilla needle-in-a-haystack (NIAH) test with four task categories — retrieval, multi-hop tracing, aggregation, and question answering — and we draw supplementary held-out probes from its NIAH (single/multi-key/multi-value/multi-query), variable tracking, and common/frequent words extraction tasks." (Methodology)
- "Despite near-perfect accuracy on the vanilla NIAH test, RULER shows that almost all evaluated long-context LMs exhibit large performance drops as context length and task complexity increase, and only about half of models claiming ≥32K context handle RULER's 32K setting satisfactorily by a fixed threshold — motivating RULER as a more discriminating probe than raw NIAH for our carry mechanism." (Methodology / Motivation)
- "RULER's own evaluation found that non-Transformer recurrent architectures (RWKV-v5, Mamba) substantially underperform a same-size Transformer baseline (Llama2-7B) and degrade sharply by 8K context, indicating that fast/recurrent-state mechanisms have historically struggled on RULER-style long-context probes — a precedent for tempering expectations of large RULER gains from a persistent fast-weight carry, and for explaining a null/flat RULER result." (Discussion — explains flat/non-improving RULER)
- "The RULER authors caution that its multi-hop tracing and aggregation tasks are unverified proxies for realistic long-context tasks and recommend treating RULER as a 'convenient behavioral check' rather than a primary benchmark — consistent with our use of RULER as a supplementary probe rather than a benchmark to optimize against." (Methodology — justifies scope limitation)

## Verification
- All task names/definitions (S/MK/MV/MQ-NIAH, VT, CWE, FWE, QA) and their construction details were read directly from Section 3 and Table 2 of the PDF.
- The Llama2-7B 85.6% threshold, effective-length methodology, and Table 3 model rankings were read directly from Section 4 and Table 3.
- The RWKV/Mamba underperformance claim was read directly from Section 6 ("Effect of architecture") and Figure 4 caption/text — confirmed as "significant degradation... 8K" and "underperform the Transformer baseline Llama2-7B by large margins," not inferred.
- The CWE ">80% copied at 128K" figure and MQ-NIAH "~15 points" drop are both quoted/paraphrased directly from Section 5 body text, not inferred from figures alone.
- Could not access page beyond 14 (appendices, additional model details) — all cited claims above come from the main text (pages 1–14), which is sufficient for what this note needs.

# FILE: RWKV: Reinventing RNNs for the Transformer Era_importance.md

# RWKV: Reinventing RNNs for the Transformer Era

**Citation:** Bo Peng, Eric Alcaide, Quentin Anthony, et al. (2023), arXiv:2305.13048v2 [cs.CL], 11 Dec 2023 (Generative AI Commons / EleutherAI et al.)

## Important parts for my research
- Core recurrence (Eq. 16): the WKV operator is a running weighted sum of value vectors, `wkv_t = [Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} ⊙ v_i + e^{u+k_t} ⊙ v_t] / [Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}]` — a fixed-size per-channel state updated additively at every step, with `w ∈ (R≥0)^d` a learned per-channel time-decay (Eq. 10: `w_{t,i} = -(t-i)w`) and `u` a separate learned bonus vector for the current token (added specifically to prevent degradation of the decay term `w`). This is a single-vector, decay-weighted accumulator — structurally the same family as a Hebbian/associative fast-weight carry, but with an explicit, learned exponential forgetting term rather than pure unbounded accumulation.
- Four core elements per block: R (Receptance, gates how much past info the output uses), W (positional/time decay weight), K (key, like attention K), V (value). Time-mixing and channel-mixing sub-blocks both use token shift — linear interpolation between current and previous timestep inputs (Eqs. 11–15) — before projecting to R/K/V.
- Two equivalent formulations: "time-parallel mode" (transformer-like, parallelizable for training, O(BTd²)) and "time-sequential mode" (RNN-like recurrent decoding, O(d) memory per step, no KV cache growth) — explicit demonstration that a linear-attention/fast-weight formulation can be trained in parallel yet deployed as a constant-memory recurrent state.
- Complexity: RWKV inference is O(T·d) time / O(d) space (Table 1), vs Transformer O(T²d)/O(T²+Td) — the state itself never grows with sequence length, it only decays and accumulates.
- Scaled to 14B params (169M–14B, 6 models, Table 2), trained one epoch (330B tokens) on the Pile; found RWKV *does* follow the same log-log scaling law form as Transformers (Pareto fit r²=0.994, extrapolated r²=0.875) — first RNN-family architecture shown to scale competitively to tens of billions of parameters.
- **Explicitly stated architectural limitation (Section 9, directly relevant to explaining a flat/non-compounding carry):** "the linear attention of RWKV leads to significant efficiency gains but still, it may limit the model's performance on tasks that require recalling minutiae information over very long contexts. This is due to the funneling of information through a single vector representation over many time steps, compared with the full information maintained by the quadratic attention of standard Transformers... the model's recurrent architecture inherently limits its ability to 'look back' at previous tokens." The authors also note prompt engineering matters more for RWKV because "the linear attention mechanism...limits the information from the prompt that will be carried over to the model's continuation."
- On RULER specifically (per the RULER paper's own note, cross-referenced): RWKV-v5-7B underperforms a same-size Transformer baseline by large margins and degrades sharply by 8K context — independent evidence a single-vector recurrent state compresses/loses information rather than accumulating it usefully at long range.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "RWKV's WKV state update (Peng et al., 2023) is a decay-weighted running sum over value vectors with a learned per-channel decay rate and a separate bonus term for the current token — structurally close to a Hebbian fast-weight accumulator, but unlike a pure Hebbian rule it includes an explicit forgetting mechanism rather than unbounded accumulation." (Related Work / Discussion — contrasts with hebbian-only carry used here)
- "RWKV's authors acknowledge that funneling all past context through a single fixed-size vector state inherently limits a recurrent model's ability to recall fine-grained (minutiae) information over long contexts, compared to full quadratic attention — a documented mechanism by which single-vector recurrent/associative state degrades rather than compounds useful information over time." (Discussion — explains flat/non-compounding carry)
- "RWKV demonstrates that a linear-attention recurrent architecture can match Transformer-style log-log compute scaling up to 14B parameters, establishing that fast-weight/linear-recurrent mechanisms are not inherently compute-inefficient at scale, even though (per RULER) their long-context task performance still lags same-size Transformers." (Related Work — positions carry mechanism against a proven-scalable recurrent baseline)

## Verification
- Eq. 16 (WKV operator), Eq. 10 (time decay), and the u-bonus explanation were read directly from Section 3.1.2 of the PDF.
- The O(Td)/O(d) complexity claim and Table 1 comparison were read directly from the PDF's Table 1 and surrounding text.
- The Section 9 "funneling of information through a single vector representation" limitation and the prompt-engineering sensitivity claim are direct quotes/close paraphrases from the Limitations section (page with Section 9) of the PDF.
- The scaling-law r² values (0.994, 0.875) were read directly from Section 4.2 and Figure 4 caption.
- The RULER-vs-RWKV comparison is carried over from the separately-verified RULER note (not re-verified against RWKV's own text, since RWKV's own PDF does not benchmark against RULER) — flagged as cross-paper, not internal to this PDF.

# FILE: Recurrent Memory Transformer_importance.md

# Recurrent Memory Transformer

**Citation:** Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev — 36th Conference on Neural Information Processing Systems (NeurIPS 2022) (arXiv:2207.06881v2, 8 Dec 2022).

## Important parts for my research
- RMT augments a Transformer's input/output sequence with special [mem] tokens — m read-memory tokens prepended, m write-memory tokens appended per segment — with no change to the backbone Transformer itself. Segment τ+1's read memory is initialized from segment τ's write-memory output: H^mem_{τ+1} := H^write_τ (segment-level recurrence via token passing, not internal state).
- Trained with BPTT; the number of previous segments to backpropagate through ("BPTT unroll," 0–4 in experiments) is a key hyperparameter — deeper unrolling substantially improves language-modeling perplexity, and is the mechanism that makes the recurrent memory tokens actually useful (not just a fixed-point compressor).
- On algorithmic Copy/Reverse tasks, RMT solves the task near-perfectly up to 9 segments (360 tokens total) with fixed memory size, while Transformer-XL degrades sharply past ~2–3 segments; RMT with memory size 60 tokens still copies correctly out to 720-token sequences.
- Language modeling: RMT matches Transformer-XL perplexity using far less memory (e.g., RMT memory 5–25 ≈ Tr-XL cache 50–75 tokens on WikiText-103 and enwik8) — RMT compresses more effectively per stored token. Combining Tr-XL cache (short-term) with RMT memory (long-term) gives the best PPL (23.99 vs. ~24.0–24.68 for either alone).
- Attention-map analysis shows interpretable write/read patterns: write-memory attention compresses the whole segment (diagonal pattern); for Reverse, the model learns to write tokens into memory in reversed order — i.e., the write operation is itself task-adapted, not a generic average.
- Real-task test: adding just 10 recurrent memory tokens to pretrained encoder LMs (RoBERTa-base, BERT, DeBERTa, T5) and fine-tuning for Hyperpartisan news classification achieves SOTA F1 (up to 98.11), beating models with 4× longer native context — memory augmentation transfers to already-pretrained models without pretraining from scratch.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) RMT as the clearest prior example of state passed *across segment/document boundaries* — but the carried state is a compact learned token-based representation trained end-to-end via BPTT-through-segments, not raw fast weights updated by an unsupervised/self-supervised rule; and RMT's segments are still slices of one continuous training corpus, not independently-bounded documents the way our carry crosses.
- (Discussion — explains flat/non-compounding carry) RMT's finding that deeper BPTT unrolling is *necessary* for the recurrent memory to be useful is a candidate explanation for our flat carry: our carry has no analogous supervised signal teaching the write step what to preserve for a future document — it accumulates via pure Hebbian outer-product updates with no gradient feedback from later performance, so nothing trains it to be predictively useful the way RMT's memory tokens are trained to be.
- (Methodology) The "memory can be retrofitted onto an already-pretrained model without retraining from scratch" result (Hyperpartisan experiment) supports methodological choices that add a persistence mechanism to a pretrained base model rather than training from scratch.

## Verification
- Segment-recurrence equations, BPTT-unroll ablations, and all quantitative results (Tables 1–5, Figs. 3–6) were read directly from the PDF text and tables (pages 3–9). Nothing here is inferred beyond direct reading.

# FILE: RedPajama: an Open Dataset for Training Large Language Models_importance.md

# RedPajama: an Open Dataset for Training Large Language Models

**Citation:** Maurice Weber, Daniel Y. Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, Ce Zhang — 38th Conference on Neural Information Processing Systems (NeurIPS 2024), Track on Datasets and Benchmarks (arXiv:2411.12372v1, 19 Nov 2024).

## Important parts for my research
- Two datasets: **RedPajama-V1**, an open best-effort reproduction of the original LLaMA (Touvron et al.) training corpus (~1.2T tokens: CommonCrawl 878B, C4 175B, GitHub 59B, Books 26B, ArXiv 28B, Wikipedia 24B, StackExchange 20B — Table 2); and **RedPajama-V2**, a much larger (>100T token, 113B documents) raw/unfiltered web-only corpus spanning 84 CommonCrawl snapshots (2014–2023) across English, German, French, Italian, Spanish, annotated with 40+ quality signals but not pre-filtered.
- RedPajama-INCITE model family (3B/7B) trained on RPv1 on the Summit supercomputer (V100 GPUs, no bf16 support, had to use fp16 + loss scaling and a lower learning rate than LLaMA — 1.6e-4 vs. presumably higher — which may explain a residual ~1–4 point HELM/LM-harness gap vs. LLaMA-7B).
- RPv2 quality signals fall into four categories: natural-language heuristics (all-caps fraction, unique-word fraction, terminal punctuation), repetitiveness (fraction of characters in repeated n-grams, n=2–10), content-based (LDNOOBW blocklist counts, UT1 blocked-domain flag), and ML heuristics (fastText classifiers vs. high-quality domains like Wikipedia/OpenWebText/books; DSIR importance weights = log-likelihood ratio between target- and source-domain language models).
- Deduplication: MinHash fuzzy dedup at multiple similarity thresholds plus exact-duplicate detection via a Bloom filter (1% error rate), applied sequentially from newest to oldest CommonCrawl dump.
- Ablation studies (Llama-2-style decoder-only, 468M and 1.6B params) show fuzzy deduplication + Gopher-rule filtering gives the best/most-consistent aggregate benchmark scores among RPv2 filter combinations, narrowly behind RefinedWeb but more consistently ranked across the widest range of tasks; Gopher-repetition-only filters ("Gopher-natlang") outperform full Gopher rules; fastText-classifier vs. DSIR filtering shows no significant difference.
- **SlimPajama** (mentioned in Related Work, Fig. 1) is explicitly identified as a cleaned + deduplicated derivative of RedPajama-V1 from Together AI — directly relevant since the proposal's training corpus is SlimPajama-6B.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Methodology) RedPajama/SlimPajama's documented composition (CommonCrawl-dominant web text, ArXiv, GitHub, Wikipedia, StackExchange, Books mix) as the basis for describing what domains SlimPajama-6B draws from when reporting per-domain persistent-TTT results (Q2 domain comparison).
- (Related Work) RedPajama-V1/SlimPajama as the open, transparent, LLaMA-recipe-reproducing pretraining corpus family used for this project's training data; cite RPv1's token breakdown by domain (Table 2) directly when describing the corpus.
- (Discussion) The quality-signal / deduplication methodology (fuzzy MinHash dedup, Gopher-rule filtering) is relevant if discussing document boundary construction or near-duplicate document effects on carry (e.g., whether repeated/near-duplicate documents in the corpus could artificially inflate or confound carry-attributable gains).

## Verification
- Token counts (Table 2), quality-signal categories, deduplication method, ablation conclusions (Section 4.3.2), and the SlimPajama provenance claim (Fig. 1, Related Work) were all read directly from the PDF (pages 2–10). Only the first 15 of 30 pages were read; remaining pages are appendix tables/additional detail not needed for this note's scope.

# FILE: Revisiting Catastrophic Forgetting in Large Language Model Tuning_importance.md

# Revisiting Catastrophic Forgetting in Large Language Model Tuning

**Citation:** Hongyu Li, Liang Ding, Meng Fang, Dacheng Tao, 2024, arXiv (cs.CL), submitted 7 Jun 2024 (arXiv:2406.04836v1); affiliations Wuhan University, University of Sydney, University of Liverpool, Nanyang Technological University

## Important parts for my research
- Central claim: catastrophic forgetting (CF) severity in LLM fine-tuning correlates directly with the flatness of the loss landscape (LLS) — sharper landscape after continual fine-tuning ⇒ more forgetting of general capability. Shown via loss-surface visualization (Fig. 1, using f(α,β)=L(θ₀+αδ₁+βδ₂), Gaussian-noise direction vectors) across three continual-tuning gaps: Alpaca (baseline) → Alpaca→Open-Platypus → Alpaca→Auto-Wiki, with gap difficulty ranked (a)≤(b)≤(c).
- Three flatness metrics defined and used: Surface Curvature (SC, Eq. 3–4, mean |K(x,y)| where K=f_xx+f_yy), Average Gradient (AG, Eq. 5), Mean Absolute Gradient (MAG, Eq. 6, mean |t_i − t_{i-1}|). All three increase monotonically with CF severity (Table 1: SC/AG/MAG rise 52.87→52.98→53.77, 105.37→106.41→111.03, 65.53→68.91→70.71 as MMLU general performance drops 40.53→33.46→23.31).
- Fix: Sharpness-Aware Minimization (SAM, Foret et al. 2021) — min_w max_{||ε||≤ρ} f(w+ε), approximated (Eq. 2) as min_w f(w + ρ∇f(w)/||∇f(w)||₂), solved by a two-step gradient process. SAM applied at fine-tuning time (not test time) flattens the LLS and mitigates CF.
- Quantitative mitigation (Table 2a, Llama2-7B, ρ=2): ΔCF without SAM was -6.08 (ShareGPT), -6.70 (Open-Platypus), -6.80 (Meta-Math) average-performance drop from the Alpaca baseline; with SAM these become +5.71, +7.01, +3.79 (net gains over baseline, at matched training epochs, w/o=2 epochs vs w/=1 epoch for fair compute).
- CF severity scales with model size: ΔDK (domain-knowledge score) drop from Alpaca→Open-Platypus is only 0.52% for TinyLlama-1.1B but 23.19% for Llama2-7B and 41.37% for Llama2-13B (Table 2b) — larger models forget more, and SAM's mitigation effect grows correspondingly with model size.
- SAM is complementary/orthogonal to existing anti-forgetting methods: combined with Wise-FT it adds +0.97% average performance; combined with rehearsal it adds +3.02% (Table 2c), on top of Wise-FT's own +5.82% and rehearsal's own +3.79% standalone gains.
- Evaluation suite spans four categories: Domain Knowledge (MMLU: STEM/Humanities/Social Sciences/Other), Understanding (RACE, OpenBookQA-fact, Csl-dev), Reasoning (SuperGLUE: AX_b, AX_g, RTE, COPA, Hellaswag, BoolQ, Siqa), Exams (ARC-c); plus domain-specific TruthfulQA/Hellaswag capability-preservation check (Table 4: SAM improves both, e.g. Hellaswag 73.16→73.49, TruthfulQA 23.87→23.99).
- Limitation acknowledged by the authors: study is confined to the fine-tuning phase; does not address CF from post-deployment / continual updates at inference time — directly the gap our carry-based TTT sits in.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Discussion — explains flat/regressing carry) Loss-landscape sharpness as a forgetting mechanism gives a candidate lens for why repeated TTT carry updates might destabilize rather than compound gains: if Hebbian/outer-product updates sharpen the loss surface near the adapted weights, later documents could be hurt rather than helped, consistent with a flat/non-growing carry signal.
- (Methodology) The three flatness metrics (SC, AG, MAG, Eq. 3–6) and the loss-surface-visualization protocol (Gaussian random directions, Eq. for f(α,β)) are directly reusable as a stability-check diagnostic for our carry-reset baseline — could probe whether carried fast weights sharpen the loss landscape over sessions.
- (Related Work) Direct citation for "forgetting analyzed in LLMs specifically during tuning" — same failure-mode category as our stability check on carried TTT weights, but at fine-tuning-time granularity rather than test-time.
- (Discussion) Their finding that forgetting severity scales with model size is relevant when discussing why Qwen3 0.6B–4B might show different carry/forgetting dynamics across scale.

## Verification
- All numbers (Table 1 SC/AG/MAG/MMLU values, Table 2a/b/c average deltas, Table 4 Hellaswag/TruthfulQA, SAM equations 1–2, flatness metric equations 3–6) were read directly from the PDF text and tables. The ρ=2 hyperparameter and epoch settings (w/o=2, w/=1) are stated explicitly in Section 4/Appendix A.2. No inferred figures.

# FILE: State-space modeling in long sequence processing: a survey on recurrence in the transformer era_importance.md

# State-Space Modeling in Long Sequence Processing: A Survey on Recurrence in the Transformer Era

**Citation:** Matteo Tiezzi, Michele Casoni, Alessandro Betti, Marco Gori, Stefano Melacci, 2025, arXiv (survey, 61 pages, arXiv genPDF)

## Important parts for my research
- Positions the expressivity gap between recurrent/state-space models and full attention: Jelassi et al. show a state-space model cannot solve the Copy task unless its latent state grows linearly with sequence length L — attention can copy by explicitly storing/retrieving via QK', while any fixed-width state (Mamba, S4, GSS, etc.) compresses the whole past into a bottleneck. Empirically, Transformers need ~10⁴ examples to learn copying; Mamba needs >10⁶ and still generalizes worse to L>1000 (Section 8.1).
- Table 6 (memory-intensive synthetic probes, from cited papers): Induction Heads / Associative Recall accuracy — Transformer 100%/100%, S4D 35.6%/86.0%, GSS 6.8%/78.0%, H3 100%/99.8%, Mamba 100%/100%. Shows selection mechanisms (Mamba's input-dependent transition matrices) close much of the gap that fixed/LTI SSMs (S4, GSS) cannot.
- Directly relevant theoretical result: recurrent models with only positive-eigenvalue diagonal state-transition matrices (typical linear RNNs like Mamba, DeltaNet) cannot represent negative eigenvalues, which blocks state-tracking/parity-type tasks; allowing negative/non-diagonal eigenvalues (as in later variants) improves stability and state-tracking. Directly informs why our Hebbian-only (pure outer-product, no error correction) update rule may be architecturally limited relative to delta-rule variants for tracking/carrying information — ties to our decision that delta rule exists but is unused (hebbian-only research path).
- Covers fast-weight/linear-attention lineage relevant to our mechanism directly: linear attention's recurrent form S_t = S_{t-1} + v_t⊗φ(k_t) (Eq. 10/15) is literally an outer-product ("associative") memory write — our Hebbian rule's exact mechanism, confirmed equivalent to Fast Weight Programmers (Schlag et al.) once the φ(q_t) normalization is added. Delta Network (Eq. 13) upgrades this to an "error-correcting delta rule": g_t=σ(w_g'u_t); S_t = S_{t-1} + g_t(v_t − S_{t-1}φ(k_t))⊗φ(k_t) — i.e., it writes the *residual* between the new value and what's already retrievable, with a dynamic learning rate g_t, explicitly contrasted against plain associative writes for interference/capacity reasons. Gated DeltaNet, DeltaProduct (multiple Householder transforms/token, tunable rank), and RWKV-7 (generalized delta rule) are named as follow-ups (per Section 8.1, pages 41-50).
- Section 8.3 "Infinite-length Sequences: Lifelong Learning" is highly relevant framing: explicitly identifies that almost no recurrent-model research addresses learning online from a continuous, unsegmented, potentially infinite stream without resets — requiring (i) gradients computable at time t without unrolling the whole history, and (ii) retained adaptation over time (akin to Online/Streaming Continual Learning). States this scenario "remains largely unexplored" and flags RTRL and its recent memory/compute-efficient refinements as the main online-learning alternative to BPTT for RNNs — precedent for describing cross-document persistence (our carry) as an open problem the field itself has flagged.
- Notes the original LSTM/RNN online-learning ambition (Williams & Peng 1990) was explicitly "local in space and time," requiring no manual resets or stream segmentation — the historical goal our session-persistent TTT work revives, in contrast to In-Place TTT's per-document reset.
- Empirical Transformer-vs-SSM comparison (Section 7.8, LRA Table 7): Transformer weakest on ListOps/short tasks but SSMs (S5, LRU) dominate very long sequences (Path-X 16K: S5 97.98% vs Transformer 7.07%); Mega (hybrid diagonal-state-kernel + attention) best average accuracy overall — supports a hybrid framing for future carry+attention combinations.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Cite for the fast-weight/linear-attention/delta-rule lineage (linear attention, DeltaNet, Gated DeltaNet, DeltaProduct, RWKV-7) as a compact, up-to-date map of the mechanism family our TTT carry extends, with named comparison points for update-rule expressivity.
- (Discussion — explains flat/non-compounding carry) The paper's explicit framing of the delta rule as "error-correcting" (writing v_t minus what the memory already retrieves, Eq. 13) versus plain associative/Hebbian writes (Eq. 10, no correction, pure accumulation) directly motivates why a Hebbian-only carry could plateau or saturate rather than compound: repeated outer-product writes with no subtraction of existing content can saturate/interfere in the memory matrix, while delta-rule writes are designed precisely to avoid this. Also cite the positive-eigenvalue/diagonal-transition-matrix state-tracking limitation (Section 8.1) as a second, complementary mechanistic candidate.
- (Related Work / Discussion) Section 8.3's explicit statement that infinite-length/streaming online learning without resets is "largely unexplored" in the recurrent-model literature is a strong citable claim to justify the novelty of cross-document persistence as a research gap.
- (Related Work) Table 6/Section 8.1 expressivity-gap results (Copy task, Induction Heads/Associative Recall) usable to frame why plain linear-attention-style fast weights (no selection/gating) are theoretically bounded, contextualizing our own architecture's choices.

## Verification
- Directly read and visually confirmed pages 1–20 (Sections 1–3.4: intro, Table 1 taxonomy, RNN/BPTT background Eq. 1-6, linear attention Eq. 7-12, Fast Weight Programmers/Delta Network Eq. 13 with the exact "error-correcting delta rule" wording, RWKV Eq. 18-23, RetNet Eq. 29-30, GLA Eq. 32-33) and pages 41–50 (Section 7.8 empirical comparison incl. Table 6 IH/AR and Table 7 LRA, Section 8 Discussion incl. 8.1 expressivity limits/Copy task/eigenvalue constraint/Gated DeltaNet/DeltaProduct/RWKV-7 mentions, 8.2 hardware, 8.3 infinite-length lifelong learning, Section 9 Conclusions). Pages 21–40 (Sections 4 Deep SSMs continued, 5 Enhancing RNNs, 6 Learning in Recurrent Models, start of 7 Benchmarks) were NOT visually inspected — no claim in this note depends on content exclusive to those pages; flagging this gap explicitly rather than inferring.

# FILE: TENT: FULLY TEST-TIME ADAPTATION BY ENTROPY MINIMIZATION_importance.md

# Tent: Fully Test-Time Adaptation by Entropy Minimization

**Citation:** Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell, 2021, ICLR (Published as a conference paper at ICLR 2021; arXiv:2006.10726v3)

## Important parts for my research
- Defines "fully test-time adaptation": only model parameters θ and unlabeled target data x^t are available — no source data, no labels, at test time (Table 1 contrasts with fine-tuning, domain adaptation, and TTT/Sun et al. 2019, which all require source data or a pretrain-time proxy loss). Directly the same "adapt with only test data" setting our carry-based TTT sits in, but Tent updates only normalization/affine parameters rather than fast weights, and never persists state across independent input batches beyond the current corruption stream.
- Method: minimize Shannon entropy H(ŷ) = −Σ_c p(ŷ_c)log p(ŷ_c) of model predictions, batched (to avoid the trivial single-prediction collapse where all probability mass goes to one class). Only two parameter types are updated: batch-norm statistics μ,σ (re-estimated on target batches, no gradient) and the affine transform γ,β (updated by gradient of entropy, <1% of model parameters) — motivated explicitly by needing updates that are efficient, stable, and don't risk divergence from the pretrained θ.
- Results: on ImageNet-C, Tent reaches 44.0% error online, improving over the prior SOTA robust-training method ANT (50.2%) and test-time normalization alone (49.9%) — "without altering training." On CIFAR-100-C at max severity (Table 2), Tent achieves 37.3% error vs. source 67.2%, RG 38.9%, UDA-SS 47.0%, TTT 45.0%, BN 42.6%, PL 41.2% — Tent is lowest across all baselines despite needing the least data/computation. On SVHN→MNIST domain adaptation, Tent with 10 epochs reaches 8.2% error (source-only 18.2%), and with just 1 online epoch reaches 12.5% — using ~80x less computation than methods needing source-data joint training (RG, UDA-SS use 10+10 epochs).
- Critical negative/limiting result relevant to our own null-result framing: on the harder MNIST→SVHN shift, Tent *fails*, increasing error to 79.8% from a 71.3% source-only baseline — the paper explicitly reports where fully-source-free adaptation cannot help and joint source+target optimization remains necessary (DIRT-T needed for that shift). This is a clean precedent for reporting adaptation failure/regression honestly rather than only success cases.
- Ablation: "adapting the full model parameters θ never improves over the unadapted source model" (Section 4.3) — supports keeping test-time updates low-dimensional/constrained; relevant analog to why we restrict carry to a narrow subset of weights (MLP down-projection in In-Place TTT) rather than full-model TTT.
- Entropy-error and entropy-corruption correlations are explicitly measured (ρ=0.61 entropy-vs-loss across corruption types/severities, Figure 2; ρ=0.22 rank correlation between Δentropy and Δloss after Tent adaptation, Figure 6) — a template for using an unsupervised proxy signal to validate that an intervention is working, relevant if we want a cheap stability probe for carried TTT weights that doesn't require held-out loss.
- Stability/generality checks performed: adaptation generalizes from train-time-only updates to unseen test points (train-adapt CIFAR-100-C error 37.3%→test error 34.2%, i.e. adapting on a different subset than evaluated on still works); Tent adapts multiple architectures without retuning (self-attention SAN, equilibrium-solving MDEQ) with consistent improvement over source and BN baselines (Table 4).

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Table 1's four-way taxonomy (fine-tuning / domain adaptation / TTT / fully-test-time-adaptation) is a clean, citable framework for positioning our own method along the source-data-availability and supervision axes — cite directly when defining where session-persistent TTT sits (closer to fully-test-time-adaptation's "no source, no labels" but with fast-weight carry rather than normalization-only updates).
- (Discussion — framing for null result) The explicit MNIST→SVHN failure case (Tent increases error 71.3%→79.8%) is a strong precedent for rigorously reporting where a fully test-time method underperforms or regresses, directly supporting our own framing of a null/negative-leaning carry result as a legitimate finding rather than a failure to hide.
- (Methodology) The constrained-parameter-update ablation ("full θ update never improves over source model," Section 4.3) supports our architectural choice to restrict TTT updates to a narrow parameter subset instead of full-model adaptation.
- (Methodology) Entropy-loss rank correlation (ρ) as a validation technique for an unsupervised test-time signal — reusable pattern for sanity-checking whether our carry mechanism's internal signal (e.g., loss reduction from carried context) tracks actual downstream improvement.

## Verification
- Read Sections 1–6 directly from the PDF (through page 9, Discussion; References on page 10 not needed for claims made). Table 1, Table 2 (CIFAR-10/100-C errors: Source 67.2, RG 38.9, UDA-SS 47.0, TTT 45.0, BN 42.6, PL 41.2, Tent 37.3, all C100-C column), Table 3 (SVHN→MNIST/MNIST-M/USPS numbers, MNIST→SVHN failure case), Table 4 (SAN/MDEQ ablation), Section 4.3 full-θ ablation claim, Figure 2/6 correlation coefficients (ρ=0.61, ρ=0.22) all read directly and double-checked against the table image, no inferred figures.

# FILE: Test-Time Training with Masked Autoencoders_importance.md

# Test-Time Training with Masked Autoencoders

**Citation:** Yossi Gandelsman, Yu Sun, Xinlei Chen, Alexei A. Efros, "Test-Time Training with Masked Autoencoders," NeurIPS 2022 (36th Conference on Neural Information Processing Systems).

## Important parts for my research
- Substitutes MAE reconstruction (mask 75% of ViT patches, reconstruct via pixel MSE) for the self-supervised task in Sun et al. 2020's TTT framework — same Y-shaped architecture (shared feature extractor `f`, self-supervised head `g`, main-task head `h`).
- TTT loop per test input: start from trained `f_0, g_0`, run gradient descent on the reconstruction loss alone to get `f_x, g_x`, predict with `h ∘ f_x`, then **discard `f_x, g_x` and reset to `f_0, g_0` before the next test input** (Sec 3, "Test-time training" paragraph) — every sample is treated independently, no persistence/carry across samples by design. Directly analogous to In-Place TTT's per-document reset; this paper never even attempts cross-sample carry, treating independence as a stated design choice, not a limitation to be lifted.
- Optimizer matters for iteration-scaling behavior: AdamW (used for MAE pretraining) requires early stopping during TTT — performance rises then falls over ~10-20 steps (Fig 2 left). SGD (momentum 0.9, weight decay 0.2, batch size 128, lr 5e-3) instead **monotonically improves and does not degrade even after 20 iterations** (Fig 2 right, confirmed Sec 4.1). This is presented as desirable because there's no validation set at test time to tune stopping — but it's also evidence that a pure gradient/Hebbian-style accumulation on a single fixed target can plateau/rise smoothly rather than blow up, contrasting with AdamW's adaptive per-parameter learning rates that overshoot.
- Rotation-prediction TTT (TTT-Rot, Sun et al. 2020's task) actively **hurts** accuracy on "rotation-invariant" classes (top-down symmetric objects like jellyfish, chiton) because the self-supervised task's signal is decorrelated from/misleading for the main task on those inputs (Table 1) — a concrete case of a self-supervised auxiliary loss producing a fast-weight update that is actively counterproductive rather than neutral, i.e., self-supervised TTT signal quality is target-dependent, not universally beneficial.
- Theoretical section (Sec 5): formalizes TTT as a bias-variance trade-off. Fixed pretrained model = biased (fit to train distribution only); training from scratch per test point = high variance/unbiased. TTT with hyperparameter α ∈[0,1] blends training covariance Σ with the single test point's rank-1 covariance x̃x̃ᵀ: M(α) = (1−α)Σ + α·x̃x̃ᵀ, then predicts via PCA on M(α). Theorem: risk is strictly lower for any α>0 than baseline (α=0), proved for linear/PCA autoencoding under orthogonal corruption R. This is a clean formal argument for *why* self-supervised TTT can help — but note it assumes the corruption is exactly the axis the self-supervised task is sensitive to; it does not address whether repeated/carried updates compound.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Per-sample TTT frameworks in vision explicitly reset fast weights after every test input, treating independence across inputs as the intended design rather than a limitation to be removed (Gandelsman et al. 2022) — our work asks what happens if that reset is instead replaced by a carry." (Related Work / motivation for cross-document persistence)
- "Prior TTT work shows self-supervised auxiliary losses can be actively counterproductive for some inputs when the auxiliary signal is decorrelated from the main task (rotation-invariant classes, Gandelsman et al. 2022) — this parallels the concern that a Hebbian carry accumulating irrelevant statistics from cross-source documents could hurt downstream perplexity rather than help it." (Discussion — motivates source-mismatch swap test)
- "TTT-MAE finds that SGD-based test-time optimization improves performance monotonically over many iterations without the overshoot/early-stopping requirement of adaptive optimizers like AdamW (Gandelsman et al. 2022), suggesting plain first-order accumulation is comparatively well-behaved but not necessarily indefinitely compounding." (Discussion — relevant to explaining a flat, non-growing carry)
- "Gandelsman et al. (2022) formalize TTT as a bias-variance trade-off, proving (for linear/PCA autoencoding) that blending training statistics with test-instance statistics strictly lowers prediction risk over any α>0 — a theoretical grounding for why inference-time adaptation can help even without persistence across inputs." (Methodology / Theory framing)

## Verification
- Reset-per-sample claim verified directly against Sec 3 "Test-time training" paragraph text ("We always discard fx and gx after making a prediction on each test input x, and reset the weights to f0 and g0 for the next test input"). SGD vs AdamW behavior verified against Sec 3 "Optimizer for TTT" paragraph and Figure 2 caption. Rotation-invariant class numbers (Table 1, e.g. Jellyfish -3 baseline vs 0 ours, Holster -1 vs 4) read directly from the table. The bias-variance theorem statement (α=0 strictly dominated, M(α) formula) verified against Sec 5 equations 5-7 and the Theorem statement. All other numeric results (ImageNet-C/A/R/Portraits tables) were not needed for this proposal's specific angles and were not individually re-verified beyond confirming table existence.

# FILE: Test-Time Training with Self-Supervision for Generalization under Distribution Shifts_importance.md

# Test-Time Training with Self-Supervision for Generalization under Distribution Shifts

**Citation:** Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt, "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts," Proceedings of the 37th International Conference on Machine Learning (ICML) 2020, PMLR 119.

## Important parts for my research
- Foundational TTT formulation (already cited in proposal Introduction — confirmed accurate). Y-structured network: shared feature extractor θ_e, self-supervised branch θ_s (rotation prediction, 4-way classification per Gidaris et al. 2018), main task branch θ_m. Training-time joint loss = main task loss + self-supervised loss, summed over both, gradients taken jointly (Eq. 2).
- **Standard TTT**: for a single test input x, minimizes ONLY the self-supervised loss l_s(x; θ_s, θ_e) w.r.t. θ_e (Eq. 3) — 10 SGD gradient steps, batches formed from augmented copies of the same single image (crop + flip). After predicting, θ_e* is **discarded**, next test point again starts from the jointly-trained θ_e, θ_m (i.e., resets — same reset-per-sample pattern as later TTT-MAE).
- **Online TTT (TTT-Online)** is this paper's persistence mechanism and the closest documented precedent to this proposal's "carry": rather than discarding θ(x_t) after each test point, it initializes θ(x_t) from the *already-updated* θ(x_{t−1}) — i.e., the feature extractor is carried forward, one gradient step per new sample, across a whole test stream. Critically, this assumes samples x_1...x_n come from the **same (or slowly-changing) distribution** Q — it is designed for sequential i.i.d./near-i.i.d. images from one benchmark, not for independently-authored, source-heterogeneous documents. This is the key differentiation point for the proposal's cross-document carry, which persists a fast weight across documents from potentially different corpus sources (ArXiv, Books, C4, etc.) rather than within one homogeneous distribution.
- TTT-Online shows genuine **compounding improvement with more exposure**: on ImageNet-C, sliding-window accuracy is still visibly rising even after 40,000-50,000 samples evaluated (Fig. 2 lower panel) — "as if we were training on the test set." This is a directly citable positive contrast case: this paper's online carry DOES compound over a long homogeneous stream, unlike this proposal's flat/non-growing carry on repeated exposure to a single held-out document — suggesting the compounding here may depend on stream diversity/volume (thousands of distinct same-distribution samples) rather than repeated exposure to one fixed document.
- Weight decay and momentum are set to **zero** during test-time optimization (inspired by He et al. 2018 / Liu et al. 2018 practice), even though they're nonzero during training — a specific hyperparameter choice for TTT stability worth comparing against.
- Formal sufficient condition (Theorem 1, convex/smooth case): TTT is guaranteed to reduce main-task loss after one gradient step whenever the inner product of gradients ⟨∇l_m, ∇l_s⟩ (w.r.t. shared parameters) exceeds a threshold ε — i.e., positive gradient correlation between the self-supervised and main task losses is the formal condition for TTT to help. Empirically validated on the deep model (Fig. 4): correlation coefficient between gradient inner product and improvement = 0.93 (standard TTT) and 0.89 (online TTT) across 75 test sets (15 corruption types × 5 severity levels).
- Comparison against UDA-SS (Sun et al. 2019, which gets the *entire* unlabeled test set at training time, an oracle-like advantage): TTT-Online still outperforms UDA-SS on 13/15 CIFAR-10-C corruptions despite seeing strictly less information (only past samples in the stream). Explanation offered: UDA-SS must find one representation invariant across both train and test distributions, while TTT-Online only needs to adapt to the current test distribution — forgetting the training distribution is explicitly framed as *not harmful* in this online setting.
- "Gradually changing distribution shifts" experiment: relaxes the fixed-Q assumption to Q_t → Q_{t+1} (interpolating corruption severity from level 1 to level 5 across the stream). TTT-Online still improves under this drift and outperforms UDA-SS on the first two noise types — precedent for evaluating persistence under a slowly-shifting rather than i.i.d. or fixed source.
- CIFAR-10.1 (real unknown distribution shift, not synthetic corruption): TTT achieves the first-ever improvement of an existing model's performance on this benchmark, but the margin is small (error 16.7% joint-training baseline → 15.9% TTT, i.e., 0.8 point absolute, "relative improvement of 4.8%") — explicitly flagged by the authors themselves as modest ("we recognize this improvement is small... but see it as an encouraging first step").

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Test-Time Training was first formulated as fine-tuning a shared feature extractor on a single test input's self-supervised loss alone, with the update discarded before the next input (Sun et al. 2020); this proposal's cross-document carry instead asks what happens when that discard step is removed across document boundaries rather than within a single homogeneous stream." (Introduction / Related Work)
- "Sun et al. (2020)'s online variant carries an updated feature extractor across a stream of same-distribution test samples and shows sliding-window accuracy still rising after tens of thousands of samples — a genuine compounding effect under near-i.i.d. conditions — which sets up a natural question this proposal investigates: whether persistence compounds similarly across independently-authored, source-heterogeneous documents." (Related Work — positioning ammunition, and Discussion — contrast for flat carry)
- "Sun et al. (2020) prove that test-time self-supervised updates are guaranteed to reduce main-task loss (in the convex case) exactly when the gradients of the two losses are positively correlated, and confirm this empirically (r=0.93) on a deep model — motivating this proposal's own diagnostic checks on whether the Hebbian carry's gradient signal remains usefully correlated with next-token prediction loss across document/source boundaries." (Methodology / Discussion)
- "Even TTT's own introducing paper reports its improvement on a genuine unknown distribution shift (CIFAR-10.1) as modest — 0.8 points absolute — and explicitly frames it as 'an encouraging first step' rather than a large effect, which is precedent for reporting a small or null persistence-attributable gain as a legitimate, citable contribution rather than a failure." (Discussion — framing a null/modest result as legitimate)
- "TTT-Online outperforms an oracle-like unsupervised domain adaptation baseline that sees the entire unlabeled test set, because forgetting the training distribution is not harmful when the goal is adapting to the current test distribution alone (Sun et al. 2020) — suggesting persistence value may lie less in what's retained long-term and more in short-horizon adaptation, relevant context for interpreting a flat/non-growing carry." (Discussion)

## Verification
- Standard vs. online TTT mechanism (Eq. 2, 3; discard vs. carry-forward θ(x_{t-1})) verified directly against Section 2 "Online Test-Time Training" paragraph text. Zero weight-decay/momentum-during-TTT detail verified against "Optimization details" paragraph in Section 3. Theorem 1 statement and correlation coefficients (0.93/0.89) verified against Section 4 text and Figure 4 caption. CIFAR-10.1 numbers (16.7% → 15.9%, "relative improvement of 4.8%") verified directly against Table 4 and the surrounding paragraph text in Section 3.3. UDA-SS "13 of 15 corruptions" claim verified against the "Comparison with unsupervised domain adaptation" paragraph. The claim that sliding-window accuracy is "still rising" at 40k-50k samples is a visual read of Figure 2's lower panel plots plus the caption/body text ("still rising towards the end... indicating room for additional improvements") — the qualitative trend is textually confirmed but the precise numeric magnitude of continued rise was not extracted (not stated as a number in the text).

# FILE: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning_importance.md

# The Surprising Effectiveness of Test-Time Training for Few-Shot Learning

**Citation:** Akyürek, Damani, Zweiger, Qiu, Guo, Pari, Kim, Andreas (2025), arXiv:2411.07279v2 (MIT). Code/data cited as public.

## Important parts for my research
- TTT here is implemented as per-task LoRA adapters (rank 64) fine-tuned at test time on a synthetic dataset built from the few-shot demonstrations, then discarded after the task — an ephemeral, single-task adaptation, not a persisted fast weight. Directly relevant precedent for framing "LoRA-as-TTT-mechanism" vs. persistent fast weights.
- Dataset construction: "leave-one-out" ICL tasks built by holding out each demonstration pair as a synthetic test example, plus invertible geometric augmentations (rotations/flips/color permutations for ARC). Direct input-output (I/O) formulation without ICL framing underperforms badly (11-task/38% drop on ARC ablation) — the ICL-style leave-one-out format is critical to TTT's effectiveness.
- Loss ablation: taking the LM loss over both demonstration outputs and the final test output ("all outputs") beats test-output-only loss (26%→29% tasks solved on the 80-task ARC dev set) — modest but consistent.
- Parametrization ablation: task-specific LoRA adapters outperform a single shared adapter across tasks by 7 tasks (24%) on ARC (tasks interfere under one adapter), but on BBH a shared adapter *improves* performance (tasks are mutually helpful, less confounding since inputs/instructions differ).
- Numbers: ARC pass@2 with 8B Llama-3 goes from 5% (fine-tuned, no TTT) to 29% (fine-tuned+TTT) on the 80-task dev subset (~6x); 53.0% on full public validation set; 61.9% ensembled with BARC's program-synthesis system, matching average human (60.2%). BBH: TTT reaches 57.8% vs. ICL 50.5% (+7.3pp), zero-shot baseline 40.9%.
- Scaling oddity: FT-only accuracy improves monotonically with model size (1B→3B→8B), but after TTT the 1B and 3B models converge to *identical* accuracy (29 tasks solved each); only 8B pulls ahead (36 tasks). TTT "closes the gap" for small models — scaling behavior after TTT is not simply monotonic.
- Task-level heterogeneity on BBH: TTT gives the largest gains on tasks with clear structural/grammatical rules (Dyck Languages, Ruin Names, Movie Recommendation, Hyperbaton: 18.5%→68.8pp range of improvement); it *hurts* on Boolean Expressions (85.7%→80.4%), a task requiring explicit sequential algorithmic computation rather than pattern-based transduction — TTT's gains are pattern/structure-dependent, not general reasoning gains.
- Removing rule-based augmentation ("No Transformations") causes a 16-task (55%) drop on ARC — augmentation diversity, not just gradient steps, drives most of the improvement.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "TTT via per-task LoRA adapters yields 5%→29% (6x) gains on ARC and 50.5%→57.8% on BBH, using leave-one-out ICL-formatted synthetic data; direct I/O formatting without ICL framing underperforms by 38%" (Related Work / Methodology — precedent for LoRA-as-TTT design and for why format of the TTT loss matters).
- "TTT gains are highly task-dependent: large on structured/grammatical tasks, negative on tasks requiring explicit step-by-step computation (Boolean Expressions, −5.3pp)" (Discussion — supports framing domain-dependence of gains, relevant to Q2 and to explaining any flat/negative results in specific domains).
- "After TTT, 1B and 3B models reach identical accuracy despite different base capacity" (Discussion — nonmonotonic scaling interaction with TTT, cautionary comparandum if my own scaling sweep is non-monotonic).
- "Task-specific LoRA adapters outperform a shared adapter on ARC, but a shared adapter helps on BBH, attributed to gradient interference vs. mutual task benefit" (Methodology — relevant to any design choice about per-document vs. shared/persisted adapters).

## Verification
- All percentages (5%→29%, 53.0%, 61.9%, 60.2% human avg, 50.5%→57.8%, 40.9% zero-shot) directly read from Figure 1, Section 4.6/Table 1, and Section 5.3 text.
- Loss/parametrization ablation deltas (26%→29%, 7-task/24% adapter gap, 11-task/38% direct I/O drop, 16-task/55% no-transformations drop) read directly from Section 4.3/4.4 text and Figure 5 captions.
- Task-specific BBH numbers (Dyck Languages 18.5→68.8 etc., Boolean Expressions 85.7→80.4) read from Figure 9 and Section 5.4 text.
- No inferred numbers; all figures cross-checked against the corresponding page text, not extrapolated from charts alone.

# FILE: Titans: Learning to Memorize at Test Time_importance.md

# Titans: Learning to Memorize at Test Time

**Citation:** Behrouz, Zhong, Mirrokni (2025), arXiv:2501.00663v1, Google Research. (Proposal cites as NeurIPS 2025.)

## Important parts for my research
- Neural long-term memory M is an MLP (deep memory, L_M ≥ 1 layers) trained online at test time via gradient descent on an associative-memory loss ℓ(M_{t-1}; x_t) = ‖M_{t-1}(k_t) − v_t‖² (Eq. 12), with k_t = x_tW_K, v_t = x_tW_V — same key/value framing as attention, but compressed into MLP weights rather than a growing KV cache.
- Update rule (Eq. 9-10, 13-14) is momentum + weight-decay gradient descent, not plain Hebbian/outer-product: M_t = (1−α_t)M_{t-1} + S_t, S_t = η_tS_{t-1} − θ_t∇ℓ(M_{t-1};x_t). "Surprise" (gradient norm) drives what gets written; α_t is an adaptive forget gate; η_t is a data-dependent surprise-decay (can gate out irrelevant/off-context surprise by η_t→0). Framed explicitly as gradient descent with momentum and weight decay reinterpreted as forgetting — relevant precedent for justifying why Hebbian-only (no momentum, no decay) carry might stay flat: paper's own ablation (Table 5) finds momentum and weight decay are the two largest single contributors to performance among all components.
- Three integration variants: Memory as Context (MAC) — retrieves h_t = M*_{t-1}(q_t) and concatenates it with persistent-memory tokens and the current segment before attention (Eq. 21-23), i.e., literally "supplies attention with historical context," matching the proposal's description, but only for this one variant. Memory as Gate (MAG) — combines a sliding-window-attention branch with the memory output via a learned non-linear gate (Eq. 26-28), not fed into attention. Memory as Layer (MAL) — memory module is a sequential layer before attention (Eq. 29-31). MAC and MAG perform similarly on language modeling/reasoning but MAC is notably better on long-context NIAH; both beat MAL.
- Persistent memory: a fixed set of N_p learnable, input-independent vectors prepended to the sequence (Eq. 19), argued to mirror FFN weights when FFN nonlinearity is softmax-like and to redistribute attention away from the "attention sink" bias toward initial tokens.
- Numbers (760M params/30B tokens, Table 1): Titans(MAC) Wiki ppl 19.93 vs. Transformer++ 25.21, TTT-layer 24.17, DeltaNet 24.17, Gated DeltaNet 21.18. On S-NIAH from RULER (Table 2) at 16K length, Titans(MAC) retains 94.7-98.6% accuracy across subtasks while TTT-layer collapses to 4.4% and Mamba2 to 0.0%. On BABILong, Titans(MAC) outperforms GPT-4, Llama3.1-8B+RAG, and Llama3.1-70B despite far fewer parameters (few-shot and fine-tuned settings, Fig. 6). Titans scale to >2M context window. Ablation (Table 5): removing weight decay costs the most (ppl 27.01→29.04), then momentum, then convolution, then persistent memory.
- Theorem 4.1: Titans, unlike Transformers, diagonal linear recurrent models, and DeltaNet (all TC⁰-limited), can solve problems beyond TC⁰ — theoretically more expressive on state-tracking tasks.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Titans' MAC variant retrieves historical context from a neural long-term memory and concatenates it into the attention input alongside persistent memory tokens" (Related Work / Introduction — confirms and sharpens the proposal's existing description, but note it applies specifically to the MAC variant, not MAG or MAL).
- **Persistence across document boundaries — verification result:** all experiments (language modeling, NIAH, BABILong, DNA, time series) are conducted on a single continuous sequence/context per example (training length 4K tokens, evaluated up to >2M-token single contexts). The paper contains no discussion of resetting or persisting the memory across independently-bounded documents or sessions — memory continuity is demonstrated only within one long context, not carried across separately-bounded documents. This is a genuine differentiation point for the Research Gap paragraph: Titans solves long single-context memory, not cross-document/session persistence.
- "Titans' update rule uses momentum and adaptive weight decay, framed as gradient descent with momentum on a surprise-based loss; ablations show weight decay and momentum are the two largest contributors to overall performance" (Discussion — a relevant comparandum for why a pure-Hebbian, no-decay, no-momentum carry rule (as used in this project) might fail to compound: Titans' own results suggest decay/momentum are precisely the mechanisms doing the work).

## Verification
- Persistence-across-documents claim double-checked by reading Sections 5.1-5.9 (Experimental Setup through Efficiency) plus Appendix A/C: no mention of "document," "session," or memory reset appears anywhere in the main text or appendices; all evaluations are single-context.
- MAC equations (21-23) and "supplies attention with historical context" description read directly from Section 4.1, not inferred; confirmed this description does not extend to MAG (gating, Eq. 26-28) or MAL (sequential layer, Eq. 29-31), which the proposal's phrasing does not distinguish.
- Numbers (ppl 19.93, NIAH 94.7%/4.4%, ablation ordering) read directly from Table 1, Table 2, and Section 5.9 text.

# FILE: Towards Lifelong Learning of Large Language Models: A Survey_importance.md

# Towards Lifelong Learning of Large Language Models: A Survey

**Citation:** Zheng, Qiu, Shi, Ma (2024), arXiv:2406.06391v1, South China University of Technology. (Manuscript submitted to ACM.)

## Important parts for my research
- Taxonomy splits lifelong learning into Internal Knowledge (continual pretraining: vertical-domain, language-domain, temporal-domain; continual finetuning: task-specific and task-agnostic incl. instruction-tuning, knowledge editing, alignment) and External Knowledge (retrieval-augmented, tool-based) — 12 scenarios total, all defined as updating (or supplementing) parameters through discrete training-time task sequences {D^(1),...,D^(T)}, not inference-time updates.
- Standard evaluation metrics, directly reusable: average accuracy (AA_t = mean performance over t learned tasks), average incremental accuracy (AIA), forgetting measure (FGT_t = mean of [max prior performance − current performance] per old task), backward transfer (BWT, performance change of old tasks after learning new ones), forward transfer (FWD, gain on a new task vs. a randomly-initialized model trained only on that task). These formalize exactly the stability/plasticity split needed to report carry-attributable gain vs. forgetting in a dynamic-evaluation setup.
- Four common technique families across all 12 scenarios: replay-based (experience/generative), regularization-based (weight regularization e.g. EWC; feature regularization), architecture-based (prompt tuning, prefix tuning, LoRA, adapters, MoE, model expansion), distillation-based (from new/old/pseudo-old data).
- **Verified negative finding:** the terms "test-time," "fast weight," "carry," and "session" do not appear anywhere in the paper (confirmed via full-text search). The survey's "lifelong learning" is exclusively about training-time parameter updates (full/partial fine-tuning, continual pretraining) or non-parametric external augmentation (RAG, tools) — it does not recognize inference-time / test-time parameter adaptation as a lifelong-learning axis at all.
- Section 6.1 names catastrophic forgetting, the plasticity-stability dilemma, computation cost, and unavailability of weights/data as the four core open challenges; Section 6.3 lists multimodal, efficient (pruning/merging/expansion), and "general" (RL/agent-driven) lifelong learning as future directions — none anticipate test-time fast-weight persistence.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A comprehensive 2024 survey covering 12 lifelong-learning scenarios for LLMs defines the field entirely around training-time parameter updates or external retrieval/tools; it contains no treatment of test-time/inference-time parameter adaptation" (Related Work — establishes that session-persistent TTT occupies a genuinely distinct axis not covered by mainstream "lifelong learning" taxonomy).
- "Standard lifelong-learning metrics — forgetting measure (FGT), backward transfer (BWT), forward transfer (FWD) — formalize stability vs. plasticity tradeoffs" (Methodology — candidate framing/vocabulary for reporting carry-attributable gain alongside forgetting/instability, even though computed here at inference time rather than after training-time tasks).

## Verification
- Full-text keyword search (pdftotext) for "test-time," "test time," "fast weight," "carry," "session" returned zero matches — confirmed absence directly, not inferred from skimming.
- Taxonomy (Fig. 2), metrics (Eq. 1-5, Section 2.2), technique families (Section 2.3, Fig. 3-4), and challenges/future directions (Section 6.1-6.3) all read directly from the PDF text.

# FILE: Training Compute-Optimal Large Language Models_importance.md

# Training Compute-Optimal Large Language Models

**Citation:** Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al., 2022, DeepMind (arXiv:2203.15556, published version 2023 — "Chinchilla" paper)

## Important parts for my research
- Central finding: for a fixed FLOPs budget, model size (N) and training tokens (D) should scale in equal proportion (N_opt ∝ C^0.5, D_opt ∝ C^0.5 across three independent estimation methods), contradicting Kaplan et al. (2020) which recommended model size scale faster (a=0.73) than data (b=0.27).
- Chinchilla (70B params, 1.4T tokens) uses the same compute budget as Gopher (280B params, 300B tokens) — i.e., 4x smaller model, 4x more tokens — and uniformly outperforms Gopher, GPT-3 (175B), Jurassic-1 (178B), and MT-NLG (530B) on a large range of downstream tasks (MMLU 67.6% 5-shot vs Gopher's 60.0%, a 7.6% absolute improvement).
- Table 3 gives concrete compute-optimal token counts by parameter count relevant to the proposal's target sizes: a 1B-parameter model is optimally trained on ~20.2B tokens (1.21e20 FLOPs); a 10B-parameter model on ~205.1B tokens. No exact entries for 0.6B–4B, but this establishes the reference ratio (~20 tokens/parameter) for judging whether SlimPajama-6B (a ~6B-token corpus) is adequate or undersized relative to Qwen3 0.6B–4B targets.
- Parametric loss form: L̂(N,D) = E + A/N^α + B/D^β — an entropy floor term (E) plus separate terms for model-size-limited underperformance and finite-optimization-steps underperformance. Not directly about fast weights, but a useful template for describing loss decomposition if the paper wants to argue carry contributes a bounded, saturating term rather than unbounded improvement.
- Notes that current LLMs are "significantly undertrained" relative to their parameter count — a general scaling caution but not specific to TTT/persistence mechanisms.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Model size and training-data volume should be scaled in tandem for compute-optimal pretraining; the ratio implied by Hoffmann et al. (2022) suggests small-scale TTT experiments on models in the 0.6B–4B range should be checked against comparably scaled token budgets" (Methodology — justifying model/dataset scale choices).
- "Chinchilla's 4x-smaller, 4x-longer-trained model outperforming Gopher demonstrates that undertrained large models leave headroom that better data exposure — not just added parameters — can close, a caution against over-interpreting small perplexity gains as capability gains without accounting for training-token adequacy" (Discussion, if used to caveat magnitude of observed effects).

## Verification
- Directly verified from PDF text: N_opt ∝ C^0.50, D_opt ∝ C^0.50 (Table 2, Approach 1); Chinchilla 70B/1.4T tokens vs Gopher 280B/300B tokens (Table 1, Section 4); MMLU Chinchilla 67.6% vs Gopher 60.0% (Table 6); parametric loss equation L̂(N,D)=E+A/N^α+B/D^β (Eq. 2); Table 3 entries for 1B (20.2B tokens) and 10B (205.1B tokens) parameters read directly.
- No specific entry exists in the paper for 0.6B–4B parameter models — the "~20 tokens/parameter" comparison to SlimPajama-6B is my inference from interpolating Table 3, not a number stated directly in the paper. Flagged as inference, not a verified paper claim.

# FILE: Transformer-XL: Attentive Language Models beyond a Fixed-Length Context_importance.md

# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

**Citation:** Dai, Yang, Yang, Carbonell, Le, Salakhutdinov (2019), ACL 2019; arXiv:1901.02860v3, Carnegie Mellon University & Google Brain.

## Important parts for my research
- **Segment-level recurrence with state reuse** (Section 3.2, the core mechanism): during training, the hidden-state sequence h^n_τ computed for segment τ at layer n is cached with a stop-gradient, SG(h^n_τ), and concatenated along the length dimension with the next segment's layer-(n−1) hidden states to form an extended context h̃^{n−1}_{τ+1} = [SG(h^{n−1}_τ) ∘ h^{n−1}_{τ+1}]. Keys and values for the new segment are computed from this extended context (k, v ← h̃^{n−1}_{τ+1}W_k, W_v), while the query still comes from the current segment only. **No gradient flows into the cached states, and no model parameter is modified by this process** — the same fixed W_q,W_k,W_v,W_o are reused for every segment; what persists across the segment boundary is only the cached activations (a KV-style extension), not any updated weight. This precisely confirms the proposal's existing characterization ("cached activations rather than an updated weight").
- The memory m^n_τ can span multiple previous segments (length M, set equal to or a multiple of segment length L in experiments), explicitly likened to memory-augmented neural networks (Neural Turing Machines, Memory Networks) rather than to a written/updated weight matrix.
- **Relative positional encoding** (Section 3.3) is required to make state reuse coherent: reusing the same absolute positional encoding U for both segments would make token x_{τ,j} and x_{τ+1,j} indistinguishable in position, "resulting in a sheer performance loss." The fix decomposes the attention score into four terms — content-based addressing (a), content-dependent positional bias using relative distance R_{i−j} (b), a learnable global content bias u (c), and a learnable global positional bias v (d) — extending Shaw et al. (2018), which only has terms (a) and (b).
- **Context fragmentation**: fixed-length segments chunked without respecting sentence/semantic boundaries starve early-segment predictions of context. Segment recurrence fixes this even when no long-term dependency exists at all — demonstrated via a controlled ablation on the (sentence-shuffled) One Billion Word dataset, where recurrence alone improves ppl from 27.1 → 25.2 (Table 7) purely by resolving fragmentation, not by capturing longer dependency.
- Quantified dependency length via a new Relative Effective Context Length (RECL) metric: Transformer-XL reaches RECL of ~900 words vs. 500 (QRNN) and 400 (LSTM) at r=0.1 — "80% longer than RNNs and 450% longer than vanilla Transformers." SOTA results: enwik8 bpc 0.99, text8 bpc 1.08, WikiText-103 ppl 18.3, One Billion Word ppl 21.8, Penn Treebank ppl 54.5 (no finetuning). Up to 1,874x faster than the vanilla segment-chunked Transformer at evaluation (Table 9) due to activation reuse.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "Transformer-XL's segment-level recurrence caches prior-segment hidden activations via a stop-gradient and extends attention context across segment boundaries, updating no model parameter — a purely activation-level (KV-cache-like) persistence mechanism" (Research Gap — verified, sharpens the existing citation with the precise stop-gradient/no-weight-update mechanism).
- "Relative positional encoding was introduced specifically to make cross-segment state reuse coherent, since absolute encodings collide across segment boundaries" (Related Work/Methodology — relevant if carry across documents requires similarly rethinking positional handling between sessions).
- "Segment recurrence alone (independent of long-range dependency) fixes context fragmentation, improving even a shuffled, no-long-range-dependency dataset" (Discussion — cautions that gains attributed to "persistence" could partly be a fragmentation-avoidance artifact rather than genuine long-range carry, a possible confound to consider when isolating the persistence-attributable gain).

## Verification
- The stop-gradient / no-weight-update mechanism, memory notation m^n_τ, and the "memory augmented neural network" framing read directly from Section 3.2 text and its equations.
- Relative positional encoding four-term decomposition (a)-(d) and the comparison to Shaw et al. (2018) read directly from Section 3.3.
- Context-fragmentation ablation numbers (27.1→25.2) read directly from Table 7 and the surrounding text (Section 4.2, "isolating the effects of resolving the context fragmentation problem").
- RECL numbers (900/500/400 words) and SOTA perplexities read directly from Section 4.3 text and Tables 1-5.

# FILE: Transformers Learn In-Context by Gradient Descent_importance.md

# Transformers Learn In-Context by Gradient Descent

**Citation:** von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov (2023), ICML 2023 (PMLR 202); arXiv:2212.07677v2, 31 May 2023.

## Important parts for my research
- Hypothesis 1 (the paper's central claim): training Transformers autoregressively makes their forward-pass in-context learning implement gradient-based optimization of an implicit inner loss constructed from the in-context data itself.
- Proposition 1: for a 1-head **linear** self-attention (LSA) layer (softmax removed) with tokens e_j = (x_j, y_j), one can explicitly construct P, W_K, W_Q, W_V such that the layer's token update e_j ← e_j + (0, −ΔWx_j) is *exactly* one step of gradient descent (learning rate η) on the regression loss L(W) = (1/2N)Σ‖Wx_i − y_i‖² (Eq. 2-3): ΔW = −(η/N)Σ(Wx_i − y_i)x_i^T. Trained single-LSA-layer Transformers on synthetic linear-regression tasks converge to weights matching this construction almost exactly (near-1 cosine similarity between GD-derived and trained-layer predictions/sensitivities, Figure 2) — a mechanistic, not just behavioral, equivalence.
- K-layer LSA-only (or recurrent 2-layer) Transformers implement K steps of GD, but *trained* multi-layer models systematically **outperform plain GD** and instead match "GD++" — a curvature-corrected GD variant that transforms input data as x_j ← H(X)x_j with H(X) = (I − γXX^T) before each step (Appendix A.10). I.e., Transformers meta-learn an accelerated/preconditioned gradient descent, not vanilla gradient descent — direct evidence that gradient-descent-like in-context adaptation can spontaneously acquire momentum/curvature correction rather than staying "plain."
- MLPs inserted before self-attention let Transformers solve *nonlinear* regression by learning a linear model via GD on deep MLP representations (Proposition 2), equivalent to kernel regression with kernel k(x,y) = m(x)^Tm(y) induced by the MLP — empirically matched against meta-learned MLPs fine-tuned by one GD step on sine-wave regression (Figure 4).
- Resolving the (x,y)-token assumption: softmax (not linear) self-attention in an early layer learns a copying/merging mechanism resembling induction heads (Olsson et al. 2022) that concatenates separate input and target tokens into the (x_j,y_j) format Proposition 1 requires; only after this could later layers perform GD-like updates (Proposition 3, Section 4). Linear-only attention could not be trained to do this copying step — softmax attention was necessary.
- Explicit conceptual link to fast weights: linear self-attention is framed as equivalent to fast-weight programming (Schmidhuber 1992; Schlag et al. 2021) — the same mechanism family underlying TTT/carry.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "In-context learning in trained Transformers is mechanistically equivalent to an implicit step (or steps) of gradient descent on a loss constructed from the context, verified via near-exact weight/prediction alignment, not just matched loss curves" (Discussion / Methodology — the central validity concern for this project: any perplexity improvement observed on a replayed held-out document could in principle be reproduced by ordinary in-context adaptation over the tokens seen so far, independent of whether fast weights are actually carried).
- "Trained multi-layer Transformers implement an accelerated GD variant with learned curvature correction (GD++), outperforming plain gradient descent" (Discussion — relevant precedent when interpreting why plain, correction-free Hebbian carry might underperform a GD-based or momentum-based update rule, echoing the Titans finding that momentum/decay matter).
- "Because each dynamic-evaluation session resets the in-context window while only the fast weight persists, any gain beyond first-exposure performance cannot be attributed to ordinary in-context learning as characterized by von Oswald et al., since ICL here is shown to require the context tokens to be present in the same forward pass" (Methodology — justifies why the carry-reset baseline isolates a persistence-attributable effect distinct from ICL).

## Verification
- Proposition 1 construction and Eq. 2-3 read directly from Section 2; the GD++ curvature-correction claim read directly from Section 3 ("Multiple steps of gradient descent vs. multiple layers of self-attention") and confirmed against Figure 3's caption.
- Kernel-regression equivalence (Proposition 2) and the induction-head/copying claim (Proposition 3, Section 4) read directly from the corresponding proposition statements and surrounding text, not inferred.
- The "validity concern for carry vs. ICL" framing in What to cite is my own methodological inference from the paper's findings, not a claim made by the authors themselves — flagged as inferred.

# FILE: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention_importance.md

# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

**Citation:** Katharopoulos, Vyas, Pappas, Fleuret, 2020, ICML (Proceedings of the 37th International Conference on Machine Learning, PMLR 119)

## Important parts for my research
- Formalizes the exact mechanism this proposal's "carry" is a special case of: replacing softmax attention's exponential similarity with a kernel feature map φ(·) (using φ(x)=elu(x)+1, Eq. 7) makes self-attention a linear dot-product of feature maps, reducing complexity from O(N²) to O(N) via the associative property of matrix multiplication (Eq. 4–6).
- Section 3.4 explicitly rewrites a causally-masked linear-attention layer as an RNN with two hidden/"fast weight" states updated by simple recurrences (Eq. 16–20):
  - s_i = s_{i-1} + φ(x_i W_K)(x_i W_V)^T — an **outer-product, purely additive accumulator** (no error term, no decay, no gating) — this is the textbook Hebbian update.
  - z_i = z_{i-1} + φ(x_i W_K) — a normalizer accumulated the same additive way.
  - Output: y_i = f_l(φ(x_i W_Q)^T s_i / φ(x_i W_Q)^T z_i + x_i).
  - s_i is literally a "fast weight" matrix (state size D×M, independent of sequence length) that is read multiplicatively via the query and written additively via key⊗value outer products at every step — this is the direct mathematical ancestor of the fast-weight "carry" state in this proposal.
- Because the update is a running sum of outer products with no forgetting/decay/error-correction term, the state s_i by construction never "forgets" and never self-corrects — every new outer product is added in full regardless of whether it agrees or conflicts with what's already stored. This is a clean formal illustration of why a purely Hebbian accumulator (no delta-rule error term) can plateau in *usable* signal even while the raw sum keeps growing: with a fixed-magnitude nonlinearity like elu(·)+1, the *unnormalized* value written per step doesn't shrink, but because retrieval always divides by the co-accumulated normalizer z_i, the readout is bounded even though the underlying store isn't error-corrected — an argument for why a Hebbian carry's readout could look flat/non-compounding rather than divergent.
- Establishes the "transformers ARE RNNs" framing that Titans, In-Place TTT, and this proposal all implicitly build on: any transformer layer with causal masking can be viewed as maintaining and updating an internal fast-weight state, connecting the fast-weight-programmer lineage (Schmidhuber) to modern efficient transformers.
- Explicitly distinguishes its "recurrence over time" from Universal Transformers' "recurrence over depth" (footnote in Section 3.4) — useful for keeping the Universal Transformers note's carry claim distinct from this one.
- Empirically: linear transformer inference is up to ~4000x faster (CIFAR-10 image generation, Table 2: 4,462x throughput) than softmax attention at comparable bits/dim, and up to 3x faster training in an ASR task (Table 3) — pure engineering result, not directly relevant to Q1/Q2 but useful if this proposal needs to justify a fast-weight approach's efficiency motivation vs. attention.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "A linear-attention transformer layer can be rewritten exactly as an RNN whose per-token update is an outer-product accumulation of key and value features into a matrix-valued state — the same fast-weight formalism this proposal's session-persistent carry extends across document boundaries rather than resetting per-sequence" (Related Work / Methodology — grounding the carry mechanism).
- "The canonical linear-attention recurrence (Katharopoulos et al., 2020) accumulates writes to its fast-weight state purely additively, with no error-correcting or forgetting term — the same Hebbian-only regime used in this proposal's update rule, and a plausible mathematical reason a carry state can plateau rather than compound: new writes are never checked against, or subtracted from, what's already stored" (Discussion — explains flat/non-compounding carry).
- "Dehghani et al.'s Universal Transformers recur over depth, whereas the fast-weight RNN view of linear attention (Katharopoulos et al., 2020) recurs over time/sequence position — the carry mechanism in this proposal is the latter kind of recurrence, extended past a single sequence's boundary" (Related Work — disambiguating recurrence types).

## Verification
- Directly verified from PDF: Eq. 16–20 (s_0=0, z_0=0, s_i=s_{i-1}+φ(x_iW_K)(x_iW_V)^T, z_i=z_{i-1}+φ(x_iW_K), y_i=f_l(...)); feature map φ(x)=elu(x)+1 (Eq. 7); complexity reduction O(N²)→O(N) (Section 3.2); CIFAR-10 4,462x throughput figure (Table 2); ASR "more than 3x faster" claim (Section 4.3 text) — read directly.
- The interpretation that the readout stays bounded "even though the underlying store isn't error-corrected" because of division by the co-accumulated normalizer z_i is my own analytical inference from the stated equations, not a claim the paper itself makes about carry stability or compounding — flagged as inference, to be used cautiously and reasoned through independently before citing as if the paper argued it.

# FILE: Universal transformers_importance.md

# Universal Transformers

**Citation:** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser (2019), ICLR 2019 (arXiv:1807.03819v3)

## Important parts for my research
- Proposes the Universal Transformer (UT): a Transformer with **weight sharing across depth** — instead of a fixed stack of distinct layers, the same self-attention + transition-function block is applied recurrently T times to all sequence positions in parallel (Eq. 4–5). T is a genuine recurrence in *depth* (per-symbol processing steps), not in sequence position, so it retains full parallelizability over the sequence while adding an RNN-like inductive bias.
- **Adaptive Computation Time (ACT, dynamic halting)**: each position gets its own per-symbol halting probability, so different tokens receive different numbers of refinement steps (Appendix C gives full pseudocode). Halted positions simply copy state forward. This is a directly reusable idea for "how much compute/update should this token/document get" — conceptually adjacent to how much within-document TTT adaptation a token should receive.
- UT is proven Turing-complete under the assumption of unbounded steps T (Sec. 4/Appendix B), unlike a standard fixed-depth Transformer, which is not computationally universal at finite precision (depth doesn't scale with input size).
- Empirical evidence recurrent per-symbol refinement (not just depth) helps: bAbI QA state-of-the-art (UT w/ dynamic halting: avg error 0.21/0/20 failed tasks at 10K train-joint vs. vanilla Transformer 15.2/10 failed); subject-verb agreement task, UT closes gap to best LSTM (99.2% avg vs. Transformer 96.2%, more attractors = bigger UT advantage); LAMBADA LM perplexity 142→UT dynamic halting 142 vs. vanilla Transformer 142 (comparable) but UT clearly better at high-attractor-count agreement and algorithmic tasks; WMT14 En-De +0.9 BLEU over base Transformer (28.9 vs 28.0) at matched parameter count.
- Near-perfect scores on algorithmic tasks requiring iterative computation (Copy, Reverse, Addition; Learning-to-Execute) where vanilla Transformers fail badly (e.g., Addition seq-acc: Transformer 0.0 vs. UT 1.0) — recurrence in depth specifically helps tasks needing iterative/sequential state updates, i.e. an argument for why an update mechanism that iterates (rather than being computed once per pass) captures algorithmic structure a single feed-forward pass misses.
- Note: dynamic halting *helped* accuracy on bAbI/agreement/LAMBADA but *slightly hurt* MT BLEU — halting is not a free win, direction-dependent.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The Universal Transformer demonstrates that adding a recurrent, per-position, weight-tied refinement process on top of a Transformer improves performance on tasks requiring iterative/sequential computation (algorithmic tasks, subject-verb agreement with many attractors) while retaining a Transformer's parallel training profile (Dehghani et al., 2019)" (Related Work — precedent for recurrent/iterative mechanisms layered on Transformers, relevant to framing TTT as adding a depth/time-like recurrent update to an otherwise feed-forward backbone).
- "Dehghani et al. (2019) show that a fixed-depth Transformer is not computationally universal, while unrolling the same weights recurrently for enough steps yields a Turing-complete model" (Related Work/Background — motivates why persistent, iterative parameter updates at inference time could add expressivity feed-forward inference cannot).
- "Adaptive Computation Time in the UT allocates variable per-token compute based on a learned halting probability (Dehghani et al., 2019), a candidate mechanism for allocating variable carry/adaptation strength per token or per document" (Methodology — reusable idea, not directly used).

## Verification
- All numeric results (bAbI error/failed-task counts, subject-verb agreement 99.2%/96.2%, LAMBADA PPL 142/129/134, WMT14 BLEU 28.9/28.0/28.4, algorithmic task accuracies) read directly from Tables 1–7 in the PDF.
- Turing-completeness claim and ACT description checked against Sec. 2.2, Sec. 4, and Appendix B/C text and pseudocode, not merely the abstract.
- No inferred/uncertain claims — all points above trace to specific tables, equations, or appendix sections read directly.

# FILE: Using fast weights to attend to the recent past_importance.md

# Using Fast Weights to Attend to the Recent Past

**Citation:** Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu, 2016 (arXiv:1610.06258v3, NeurIPS 2016)

## Important parts for my research
- Core update rule (Eq. 1): `A(t) = λ·A(t−1) + η·h(t)h(t)^T` — decay rate λ multiplies the previous fast-weight matrix, then the outer product of the current hidden state with itself is added, scaled by learning rate η. This is the direct ancestor of the proposal's `hebbian` update rule: pure outer-product accumulation with exponential decay, no error term, no target/prediction being compared against current state.
- Explicitly frames this as a **one-shot, non-iterative, non-error-correcting** storage rule, contrasted directly against "iterative error correction storage rule" (Gardner 1988) that would let weights retrieve each bit of a pattern from the others. The authors chose outer-product/Hebbian over error-correcting specifically for simplicity, not because it's more accurate — this is a citable admission that Hebbian storage sacrifices retrieval fidelity/capacity for simplicity.
- Closed form unrolled across time (Eq. 3–4): `A(t) = η Σ_{τ=1}^{t} λ^{t−τ} h(τ)h(τ)^T`. Because every past outer-product term is decayed by λ^{t−τ} and never "corrected" (no subtraction of already-stored information before writing new information), successive similar inputs superimpose additively rather than refining/replacing prior stored content — this is a mechanistic reason a purely Hebbian carry state would plateau at a level set by decay/interference rather than compound: new writes reinforce along directions already present in A up to the fixed point λA = A + ... i.e. steady-state under repeated identical input is a fixed geometric sum, not unbounded growth, but also not "learning" via error reduction.
- Under repeated identical (or near-identical) input, the fast weight matrix converges to a geometric-series fixed point (bounded by 1/(1−λ) at λ<1) — directly explains why a Hebbian carry would be flat/constant from the first repeat rather than showing a rising-then-plateauing curve: the "plateau" is reached essentially immediately when inputs repeat, since Eq. 3 is a straightforward decayed sum, not an incremental refinement process.
- Capacity: Hopfield-style outer-product/one-shot storage achieves ~0.15N binary vectors for an N×N weight matrix — far below the N-vector theoretical upper bound achievable with iterative error-correcting rules (Gardner 1988). Explicit efficiency-capacity trade-off number.
- Layer normalization applied inside the inner-loop iteration (Eq. 5) was found necessary to stop the scalar product h(τ)^T·h_s(t+1) from vanishing/exploding — stabilizes the fast-weight-driven state update. Reported hyperparameters used throughout: fast learning rate η=0.5, decay λ=0.95 (main text) / λ=0.9 (appendix experimental setting for associative retrieval) — note the two values differ slightly between text and appendix.
- Mechanistically, fast weights are described as providing "a kind of attention to the recent past" where attention strength = scalar product between current and past hidden states, weighted by decay — i.e., functionally equivalent to linear/kernel attention with recency decay, not a separately-parameterized attention computation.
- Empirical result: fast weights only outperform LSTM/IRNN when the number of recurrent/hidden units is small (e.g., associative retrieval error 1.81% for fast weights vs 60.81% LSTM at R=20 hidden units); the advantage shrinks or reverses as hidden capacity grows (facial expression: ConvNet 88.23% > Fast Weights 86.34% > LSTM 81.32%). Fast weights are best framed as compensating for limited slow-weight/hidden capacity, not as a universally superior mechanism.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "The Hebbian fast-weight update rule used here follows the outer-product, exponentially-decayed accumulator of Ba et al. (2016), A(t) = λA(t−1) + ηh(t)h(t)^T, chosen for its simplicity over iterative error-correcting alternatives." (Methodology / Background)
- "Ba et al. (2016) explicitly note that outer-product Hebbian storage sacrifices capacity and retrieval precision relative to iterative error-correction storage rules (Gardner 1988) — a known property of the update-rule family we adopt, not an implementation defect specific to our setting." (Discussion — explains flat carry)
- "Under a pure Hebbian accumulator, the closed-form state is a decayed geometric sum over past outer products (Ba et al. 2016, Eq. 3); repeated near-identical inputs drive this sum to a fixed point almost immediately rather than a gradually-compounding trajectory, consistent with the flat, non-rising carry we observe across repeated-document sessions." (Discussion — explains flat carry)
- "Fast-weight/attention-to-the-past mechanisms have historically shown their largest gains in regimes of constrained hidden/slow-weight capacity (Ba et al. 2016); at the scale of a multi-billion-parameter language model, this advantage may be substantially diminished, offering one candidate explanation for our modest RULER/perplexity movement." (Discussion)

## Verification
- Update equation, decay/learning-rate values (λ=0.95 main text vs λ=0.9 appendix; η=0.5 both), capacity figure (~0.15N for Hopfield one-shot storage), and all reported table numbers (Table 1 associative retrieval, Table 2 MNIST, Table 3 facial expression) were read directly from the PDF text and cross-checked against Sections 3, 3.1, and 4.
- The "fixed point under repeated input" and "compensates for limited hidden capacity" framings are my own mechanistic inferences from the closed-form equations and the reported capacity-dependent results (Sections 3 and 4.1–4.3), not statements the authors make explicitly in those terms — flagged as interpretation, not direct quotation.

# FILE: What learning algorithm is in-context learning? Investigations with linear models_importance.md

# What learning algorithm is in-context learning? Investigations with linear models

**Citation:** Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, Denny Zhou (2023), ICLR 2023 (arXiv:2211.15661v3)

## Important parts for my research
- Central question: is in-context learning (ICL) implicitly implementing a known learning algorithm on the in-context examples, encoded purely in activations, with **no parameter update at all**? This is the key validity contrast for this proposal's carry mechanism: ICL achieves adaptation via forward-pass activations over a context window, whereas TTT/carry achieves it via actual weight updates that persist after the context is gone — the two must be distinguished when claiming a persistence-attributable gain.
- Constructive proof (Theorem 1, Sec. 3.2): a Transformer decoder with O(d) hidden size and constant depth can implement **one step of gradient descent** on a linear regression objective purely through self-attention + MLP layers (explicit weight construction via mov/mul/div/aff primitives, Lemma 1). n steps of GD require O(n) stacked layers (i.e., iterative algorithms map to depth, not to any persistent state).
- Constructive proof (Theorem 2, Sec. 3.3): a Transformer can also implement **closed-form ridge regression** via a Sherman–Morrison rank-one update (Eq. 13–14), again with O(d²) hidden size and constant depth.
- Empirical result (Sec. 4.3, Fig. 1): trained ICL transformers' predictions closely match **ordinary least squares (OLS)** on noiseless data (squared prediction difference and implicit-linear-weight-difference both near 0), far more than they match nearest-neighbor baselines.
- Empirical result (Sec. 4.3, Fig. 2): under label noise, ICL predictions match the **minimum-Bayes-risk ridge regression** predictor for the correct noise/prior ratio σ²/τ² — i.e., trained ICL behaves like an implicit Bayesian estimator, not just OLS, when the problem is genuinely uncertain.
- **Algorithmic phase transitions with model capacity** (Sec. 4.3, Fig. 3): varying depth, shallow (1-layer) models match one step of gradient descent; slightly deeper (2–4L) models match ridge regression; deep (8L+) models match OLS. A similar phase transition occurs varying hidden size. This shows the *same architecture* can realize qualitatively different implicit algorithms depending on scale — directly relevant framing for why a small/shallow TTT module might show different "effective algorithm" behavior than a larger one.
- Probing results (Sec. 5, Fig. 4): both the moment matrix XᵀY and the OLS weight vector w_OLS are linearly/non-linearly decodable from hidden states at specific layers (moments emerge ~layer 7, weights ~layer 12 in a 16-layer model), and attention probes localize *where* (at which token position) each quantity is computed — evidence that intermediate learning-algorithm state is explicitly represented in activations, not just implicit in outputs.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "In-context learning can be an activation-level implementation of a learning algorithm (e.g., one step of gradient descent, or closed-form ridge/OLS regression) entirely within a forward pass, with no persistent parameter update (Akyürek et al., 2023) — this is the alternative explanation that must be ruled out when attributing gains to carried fast weights rather than to in-context adaptation over the replayed document" (Discussion/Methodology — validity distinction between ICL and carry, directly parallel to von Oswald et al.).
- "Akyürek et al. (2023) show the same Transformer architecture transitions between implementing different implicit learning algorithms (one-step GD, ridge regression, OLS) as depth and hidden size scale" (Related Work — precedent for capacity-dependent qualitative behavior of learned adaptation mechanisms).
- "Intermediate quantities computed by textbook learning algorithms (moment matrices, weight vectors) are linearly decodable from mid-to-late-layer activations of a trained in-context learner (Akyürek et al., 2023)" (Related Work/Methodology — precedent for probing-based verification methodology, potentially reusable to check whether carried fast weights encode analogous quantities).

## Verification
- All theorems, equations (Eq. 8–19), and empirical claims (Fig. 1–4, Sec. 4.3 minimum-Bayes-risk match) read directly from the PDF main text and figures.
- Probing methodology (position-attention probe, Eq. 20–21) and layer numbers (moments ~layer 7, weights ~layer 12) confirmed from Sec. 5 and Fig. 4 text/caption directly, not inferred.
- No numeric claims were taken from the appendix derivations (Appendix A/B, which are pages of matrix algebra); those support but are not needed for the cited claims above.

# FILE: YARN: EFFICIENT CONTEXT WINDOW EXTENSION OF LARGE LANGUAGE MODELS_importance.md

# YaRN: Efficient Context Window Extension of Large Language Models

**Citation:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (2023/2026 rev), arXiv:2309.00071v3 (Nous Research, EleutherAI, University of Geneva)

## Important parts for my research
- Addresses a fundamentally different mechanism for extending effective context than this proposal's carry: YaRN modifies **Rotary Position Embeddings (RoPE)** at inference/fine-tune time — no persistent weight update, no fast-weight state — pure positional-encoding interpolation. Useful as a contrast case: RoPE extension changes what positions the (unchanged) attention mechanism can address; TTT/carry changes the weights themselves. These are complementary, not competing, mechanisms for long-context/cross-session capability.
- **NTK-by-parts interpolation** (Sec. 3.2, Def. 1): per-RoPE-dimension ratio r(d) = L/λ_d (pretrained context length over that dimension's wavelength) determines treatment — dimensions with r(d) < α are linearly interpolated (scaled, as in Position Interpolation) to avoid extrapolation; dimensions with r(d) > β are left untouched (since their wavelength already exceeds context, they only carry absolute positional info that must not be disturbed); a ramp function γ(r) blends the two regimes for intermediate dimensions (Eq. 11–13). Empirically tuned α=1, β=32 for Llama-family models.
- **YaRN = NTK-by-parts + attention temperature scaling**: applies a temperature t inside the softmax (Eq. 14), equivalent to scaling both q and k by √(1/t); recommended √(1/t) = 0.1 ln(s) + 1 where s is the context scale factor (Eq. 15) — an empirically-fit, near-universal correction (works across LLaMA/Llama-2 model sizes without refitting) that has zero inference/training overhead since RoPE embeddings are precomputed.
- **Compute efficiency headline result** (Table 4): YaRN reaches state-of-the-art context extension using ~10x fewer tokens and 2.5x fewer training steps than prior interpolation methods; e.g., extending Llama-2 7B to 128k context took 256+128 A100-GPU-hours vs. ~64000 GPU-hours for a comparable NTK-aware/direct fine-tune (Xiong et al., 2023) — evidence that positional/architectural fixes can be dramatically cheaper than full retraining or naive long-context fine-tuning, a useful compute-budget comparator when framing the cost/benefit of TTT-based approaches.
- **Extrapolation beyond fine-tuning data** (Table 1, Sec. 4.2): a Llama-2 7B model fine-tuned only on 64k-token data with YaRN successfully extrapolates to 128k context (sliding-window PPL 2.42 at 65536, 2.37 at 131072 for the s=32 model) — i.e., the interpolation method generalizes past the exact context length it was tuned on, similar in spirit (though different mechanism) to the "test on longer/repeated sessions than trained" evaluation stance this proposal takes.
- Minimal capability regression from context extension (Table 2/3): YaRN fine-tuning causes only ~0.49% average drop across ARC-c/HellaSwag/MMLU/TruthfulQA vs. baseline, notably better preserved than PI or NTK-aware alone — relevant precedent for how to report "did the intervention hurt short-context/base capability" alongside long-context gains, analogous to this proposal's need to report whether TTT/carry hurts RULER or held-out short-context performance.
- Passkey retrieval (Sec. 4.3, Fig. 3 right): YaRN retains higher passkey-retrieval accuracy across prompt lengths than PI/NTK-aware/NTK-by-parts at matched training budget — a concrete long-context probe methodology (place a random passkey at a random position, measure retrieval accuracy vs. prompt length) that is methodologically similar to targeted long-context evals like RULER's needle tasks.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "YaRN extends usable context length via a training-free-compatible modification to rotary position embeddings and an empirically-fit attention temperature correction, requiring no persistent weight update, at roughly 10x less fine-tuning data and 2.5x fewer steps than prior RoPE-interpolation methods (Peng et al., 2023)" (Related Work — contrasts positional/architectural long-context extension with this proposal's parametric/fast-weight approach to cross-document adaptation).
- "Peng et al. (2023) show that naively extending context (e.g., direct fine-tuning or uniform positional interpolation) is far more compute-expensive than a targeted positional correction (~64000 vs. ~384 A100-GPU-hours for comparable 128k Llama-2 7B extension), underscoring that not all long-context gains require costly retraining" (Related Work — compute-budget contrast point).
- "Context-extension methods can generalize beyond the exact length used in their own fine-tuning data (Peng et al., 2023 extrapolate from 64k fine-tuning to 128k evaluation) without degrading short-context benchmark performance (~0.5% average drop)" (Related Work/Discussion — precedent for reporting base-capability preservation alongside a targeted capability gain, and for testing generalization past the trained regime).

## Verification
- Equations (Eq. 7–15, ramp function, temperature formula) and α=1/β=32 values for Llama read directly from Sec. 3.2–3.3 of the PDF.
- GPU-hour comparison (Table 4: 256, 256+128 vs. ~64000, ~6400) and perplexity/passkey numbers (Table 1, Fig. 3) read directly from the tables/figures.
- The ~10x tokens / 2.5x steps headline figures are from the abstract and confirmed consistent with the Table 4 GPU-hour comparison (order-of-magnitude match, not an exact restatement of a single number in the abstract).

# FILE: iCaRL: Incremental classifier and representation learning_importance.md

# iCaRL: Incremental Classifier and Representation Learning

**Citation:** Sylvestre-Alvise Rebuffi (University of Oxford / IST Austria), Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert (IST Austria). arXiv:1611.07725v2 [cs.CV], 14 Apr 2017 (CVPR 2017).

## Important parts for my research

- Defines **class-incremental learning** by three required properties: (i) trainable from a stream where classes arrive at different times, (ii) competitive multi-class classifier at any point for all classes seen so far, (iii) bounded/slowly-growing compute and memory w.r.t. number of classes seen — a clean formal template for what "session-persistent" adaptation should satisfy, independent of vision-specific content.
- iCaRL combines three components: (1) **nearest-mean-of-exemplars classification** — class prototype μ_y = mean feature vector over a stored exemplar subset P_y, prediction y* = argmin_y ‖φ(x) − μ_y‖ (Eq. 2), which decouples the classification rule from a fixed weight vector so it *automatically* tracks representation drift rather than becoming stale — directly relevant to any persistent-state design where the "readout" needs to stay valid as underlying representations keep changing; (2) **prioritized exemplar selection via herding** — Algorithm 4 greedily picks exemplars p_k so the running average of selected exemplar features best approximates the true class mean, giving a fixed-size, priority-ordered set that can be truncated (Algorithm 5: keep first m) without recomputing — an efficient/incremental way to keep a compact summary of a growing stream, useful methodological analogy for compact carry-state design; (3) **representation learning via knowledge distillation + exemplar rehearsal** — loss (shown boxed in Algorithm 3) combines a classification term for new classes with a distillation term matching pre-update network outputs (q_i^y, recorded *before* the parameter update) on old classes, explicitly to prevent catastrophic forgetting while still updating shared representation weights.
- **Catastrophic forgetting is demonstrated starkly**: plain finetuning without any of iCaRL's mechanisms collapses to ~0% accuracy on early classes by the time 100 classes are seen (Fig. 2, Fig. 3d confusion matrix — all predictions land in the most-recent batch of classes). This is the canonical empirical demonstration of what unmitigated continual parameter updates do, i.e., the failure mode any persistence mechanism (including cross-document TTT carry) must guard against.
- Ablation (Table 1a/Fig. 4, §4.2) isolates each component's contribution: mean-of-exemplars classification helps most at small class-batch sizes (more frequent representation updates); distillation loss helps most at larger batch sizes/fewer updates and can even hurt at very small batch sizes; exemplar rehearsal is the single most important factor for preventing forgetting (comparison to LwF.MC, which uses distillation but no exemplars, shows a large gap). With enough exemplars (K≥1000 on iCIFAR-100), the mean-of-exemplars classifier performs about the same as the more expensive nearest-class-mean; below that, exemplar-based classification is worth the mean/prototype approach over raw network outputs.
- Results: iCaRL clearly and increasingly outperforms finetuning, "fixed representation," and LwF.MC as more classes accumulate on iCIFAR-100 (multi-class accuracy) and iILSVRC (top-5 accuracy, both small-100-class and full-1000-class settings) — the margin over alternatives widens with more incremental steps, i.e. iCaRL's relative advantage compounds over the session/stream length rather than being fixed.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) — the three-part formal definition of class-incremental learning (streaming arrival, always-competitive classifier, bounded resource growth) as a template/precedent for defining what a "session-persistent" evaluation protocol must satisfy.
- (Related Work — catastrophic forgetting backdrop) — plain-finetuning collapse to near-0% on early classes (Fig. 2/3) as the canonical illustration of unconstrained continual parameter updates failing, motivating why any cross-document carry mechanism needs an explicit stability check (echoes the "stability checked before trusting any gain" step in our own evaluation protocol).
- (Methodology, optional analogy) — herding-based prioritized exemplar selection (Algorithm 4) as a general pattern for maintaining a compact, incrementally-updatable summary of accumulated experience under a fixed budget; relevant if discussing bounded/decayed carry state design.
- (Discussion) — knowledge distillation against pre-update outputs as an alternative persistence-stabilization mechanism to pure parameter-carry-with-clipping; worth a sentence contrasting distillation-based forgetting prevention (used here) against the norm-clipping approach used in In-Place TTT and our own carry mechanism.

## Verification
- Read main text pages 1–9 directly (through start of Conclusion), including all five algorithm boxes (Classify, IncrementalTrain, UpdateRepresentation, ConstructExemplarSet, ReduceExemplarSet) and Figures 2–4. Ablation claims (Table 1a discussion, hybrid1/hybrid2/hybrid3 comparisons) and the K≥1000 exemplar-budget claim (Fig. 4 discussion) are transcribed directly from the text, not inferred. Did not read the appendix (additional confusion matrices for iILSVRC) — not needed for this note.

# FILE: s1: Simple test-time scaling_importance.md

# s1: Simple test-time scaling

**Citation:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto, 2025, arXiv (cs.CL), v3 1 Mar 2025 (arXiv:2501.19393); Stanford, University of Washington, Allen Institute for AI, Contextual AI

## Important parts for my research
- "Test-time scaling" here means spending more decoding-time compute (longer chain-of-thought), NOT parameter updates — a different mechanism from TTT/fast-weight carry despite the shared "test-time" terminology; important to distinguish explicitly in Related Work to avoid conflation.
- s1K: a curated 1,000-example SFT dataset (from 59K candidates) selected via three criteria — Quality, Difficulty (measured by reasoning-trace token length as a proxy, using two grader models), Diversity (50 domains via Claude-based MSC taxonomy classification). Ablations show all three criteria are necessary: random selection or diversity-only or difficulty-only selection each lose ~30% on AIME24 vs the combined s1K selection; training on the full 59K pool gives no substantial gain over the 1K subset.
- Budget forcing: a decoding-time intervention — force early stop by appending the end-of-thinking delimiter (+"Final Answer:") to cap thinking tokens, or force continued reasoning by suppressing the end-of-thinking token and appending "Wait" to encourage self-correction/backtracking. This is a purely inference-time, weight-frozen control knob — no gradient updates at test time.
- Three evaluation metrics proposed for test-time scaling methods: Control (Eq. 1, fraction of runs landing within a target compute window), Scaling (Eq. 2, average slope of performance vs. compute), Performance (Eq. 3, max accuracy achieved). Budget forcing achieves 100% control and best overall scaling/performance among methods compared (Table 3: BF beats token/step/class-conditional control and rejection sampling).
- Result: s1-32B (Qwen2.5-32B-Instruct + SFT on s1K + budget forcing) reaches 56.7% AIME24 / 93.0% MATH500 / 59.6% GPQA Diamond, exceeding o1-preview on competition math by up to 27 points; scaling test-time compute via repeated "Wait" extrapolates AIME24 from 50.0% to 57.0% before flattening at ~6x forced extensions (Figure 4).
- Sequential scaling (budget forcing) outperforms parallel scaling (majority voting) on the base model — majority voting on Qwen2.5-32B-Instruct cannot match s1-32B's performance even at 1M output tokens (Figure 4b), supporting the general principle that scaling depends on how compute is spent, not just how much.
- Rejection sampling shows an inverse scaling trend: forcing longer generations via length-conditioned rejection sampling actually hurts accuracy (42%→22% as target length grows 3072→5120 tokens, Figure 6) — longer natural generations correlate with harder/more error-prone problems, a confound worth flagging when interpreting length-vs-performance correlations generally.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Cite to explicitly delineate "test-time scaling" (inference-time compute/decoding control, frozen weights) from "test-time training" (parameter updates at inference) — clarifies terminology boundary for readers, positions our carry-based TTT as a distinct and complementary axis of test-time adaptation.
- (Methodology) The Control/Scaling/Performance metric triad (Eq. 1–3) is a reusable framework for rigorously characterizing any test-time intervention's compute-accuracy tradeoff; could adapt "Scaling" (slope of performance vs. compute/repeats) as a way to quantify whether our carry signal has a non-trivial (non-flat) slope across repeated sessions.
- (Discussion — framing for null result) The inverse-scaling-via-rejection-sampling finding is a useful precedent for reporting a counterintuitive/negative scaling result rigorously rather than suppressing it, and for the general point that "more test-time intervention" does not monotonically help.

## Verification
- All numbers (s1K/59K ablation results, Table 1 benchmark scores, Table 2 1K-subset ablations with bootstrap CIs, Table 3 BF vs. other methods, Figure 4/6 trends) read directly from the PDF. Note s1-32B's reported AIME24 improvement "50% to 57%" appears in the abstract; the main text figure shows the extrapolated endpoint at 57.0%/56.7% depending on table vs. figure — flagged as a minor rounding discrepancy between abstract and Table 1, not an inference.

# FILE: xLSTM: Extended Long Short-Term Memory_importance.md

# xLSTM: Extended Long Short-Term Memory

**Citation:** Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter (2024), arXiv:2405.04517v2 (NXAI Lab / ELLIS Unit, LIT AI Lab, JKU Linz)

## Important parts for my research
- **mLSTM's covariance/Hebbian update rule is directly analogous to this proposal's carry mechanism**: matrix memory cell C_t = f_t C_{t-1} + i_t v_t k_tᵀ (Eq. 18–19) — a decayed (forget gate f_t), gated (input gate i_t) outer-product accumulation of key-value pairs, explicitly identified in the paper as equivalent to Fast Weight Programmers (Schmidhuber 1992; Schlag et al. 2021) with a constant decay rate generalized to a *learned, input-dependent* decay/write-strength. This is essentially a gated-Hebbian generalization of the pure Hebbian rule this proposal uses — relevant contrast since the active hebbian rule has no such adaptive gating.
- Retrieval: h̃_t = C_t q_t / max(|n_tᵀ q_t|, 1) (Eq. 21), with normalizer state n_t = f_t n_{t-1} + i_t k_t tracking the accumulated (gated) key mass — a concrete normalization scheme for a growing outer-product accumulator, relevant to the stability concerns of carrying a fast weight across many documents/repeats.
- **Exponential gating** (input/forget gates use exp instead of sigmoid, Eq. 12–13, 25–26) is introduced specifically to let the network *revise* stored associations (overwrite when a more relevant new key arrives) — motivated by a "Nearest Neighbor Search" toy task where plain (sigmoid-gated) LSTM fails to update a stored value when a more similar key appears later (Fig. 2 left), while exponential gating fixes this. Requires a stabilizer state m_t (Eq. 15–17) to prevent overflow. This is a mechanistic, citable explanation of *why* a non-error-correcting accumulator (like sigmoid-gated or pure Hebbian) would show flat/non-improving behavior on tasks requiring revision of previously stored information — directly relevant to explaining a flat carry.
- Rare Token Prediction (Fig. 2 right): plain LSTM's scalar memory cell has limited storage capacity, so it does worse on rare tokens; mLSTM's matrix memory (higher-capacity, richer storage) fixes this — a capacity argument for why scalar/low-rank fast-weight state might underperform on the "long tail" of a document.
- Scale/quality results (15B tokens SlimPajama, 350M-param-matched models, Table 1): xLSTM[1:0] (pure mLSTM) achieves best validation perplexity 13.43 vs. GPT-3-style Transformer 14.25/Llama 14.25, Mamba 13.70, RWKV-6 15.03, GLA 16.15 — the covariance-update/matrix-memory mechanism is competitive with or beats both softmax attention and other linear/SSM recurrences at this scale.
- Length extrapolation (1.3B models, Fig. 7): trained at context 2048, evaluated to 16384 — xLSTM maintains low, flat perplexity (8.92–9.01 PPL at 16k) while a standard Transformer (Llama) perplexity explodes to 337.83 at 16k. Directly relevant precedent for evaluating carry/session-persistent TTT stability far outside the original training-context length.
- Ablation (Table 2): isolating the two innovations shows both matter — adding exponential gating alone (xLSTM[0:1], sLSTM) drops SlimPajama-15B PPL from 26.01 (backbone only) to 17.70; adding the matrix memory + covariance update (xLSTM[7:1]) further drops to 13.48 — i.e., gating and matrix-memory/covariance-update contribute independently and multiplicatively.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- "xLSTM's matrix-memory mLSTM cell uses the same covariance/outer-product (Hebbian-style) update as Fast Weight Programmers, C_t = f_t C_{t-1} + i_t v_t k_tᵀ, but with learned input- and gate-dependent decay and write strength rather than a fixed rate (Beck et al., 2024)" (Related Work — positions the fixed, ungated Hebbian rule used in this proposal as a simplified special case of a broader, gated family).
- "Beck et al. (2024) show, via a nearest-neighbor-revision toy task, that gating without the ability to overwrite stale associations (as in a plain, non-exponential-gated accumulator) fails to update a stored value when new, more relevant information arrives" (Discussion — mechanistic explanation for a flat/non-compounding carry under a pure Hebbian rule with no error-correction or adaptive forgetting).
- "The mLSTM/xLSTM architecture demonstrates stable, low perplexity when extrapolated 8x beyond its 2048-token training context (PPL 8.92–9.01 at 16k vs. a Transformer's collapse to PPL 337.83), showing that recurrent, additively-updated state can generalize far past its trained regime given proper normalization and gating (Beck et al., 2024)" (Related Work/Discussion — precedent for the kind of stability check this proposal applies to carry across repeated sessions).

## Verification
- All equations (cell/normalizer/hidden state updates, exponential gates, stabilizer state) read directly from Sec. 2.2–2.3 of the PDF, not inferred.
- Perplexity numbers (Table 1: 13.43/14.25/13.70/15.03/16.15; Table 2 ablation: 26.01→17.70→13.48; Fig. 7: 8.92/9.01/337.83 at 16k) read directly from the tables/figure text in the PDF.
- The "equivalent to Fast Weight Programmers ... constant decay rate ... input gate to the learning rate" framing is a direct paraphrase of text in Sec. 2.3, not an inference.

# FILE: Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training_importance.md

# Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training

**Citation:** Xiangchen Song, Zhenhao Chen, Lingjing Kong, Shaoan Xie, Xinshuai Dong, Guangyi Chen, Kun Zhang (Carnegie Mellon University, MBZUAI). arXiv:2607.00368v1 [cs.CL], 1 Jul 2026. Preprint, no venue stated.

## Important parts for my research
- Names the failure mode **"evidence migration"**: "evidence that directly supports one evaluation regime is carried into a stronger deployment narrative without the behavioral tests required for that narrative" (§1). This is the exact risk for our paper if perplexity results are described as memory.
- Distinguishes **in-sequence adaptation** (update on a support chunk, evaluate on future tokens of the same stream) from **deployment-time behavioral learning** (sparse facts, preferences, corrections, procedures used later after the context is removed) (§1).
- **Three claim levels** (§4, Fig. 2):
  - **S-level**, stream or domain adaptation: updated on a recent stream, target-domain data, retrieved examples or task attempts, and evaluated by "nearby loss, same-stream performance, domain accuracy, long-context performance, or reward". Evidence: PPL/loss, future-token gain, long-context accuracy, throughput, reward.
  - **B-level**, bridge or internalization: "information can be stored, compressed, or transformed inside model state, but does not yet establish sparse delayed user-facing behavior". Evidence: answer NLL or ranking after support removal, needle recall, ablation tests.
  - **D-level**, deployment-time behavioral learning: recall, paraphrase, delay, locality, conflict, action use after context removal.
  - Each paper gets "the strongest claim level directly supported by its reported evaluations instead of the level suggested by its broadest motivating statement".
- **Audit** of 24 coded papers (§4, Table 2): In-Place TTT, TTT layers, LaCT, TTT-E2E and dynamic evaluation are coded **S-level**; MemoryLLM, PERK, Locas, SEAL, Absorber LLM are **B-level**; assistant-memory benchmarks (LoCoMo, LongMemEval, MemoryBench) are **D targets**.
- **Why perplexity is a narrower question** (§2):
  - dense same-stream evidence reflects "continuation fitting, topic priming, or memorization of nearby regularities, which is precisely what dynamic evaluation was designed to exploit";
  - teacher forcing is easier than behavioural extraction;
  - "A stream protocol can conflate immediate adaptation to the current support item with cumulative adaptation from all preceding chunks", so **reset-vs-stream comparisons** are needed.
- **Diagnostic** (§5, Table 3):
  - Setup: Qwen3-1.7B/4B/8B, one-step LoRA update on nonce access-code facts, 48 probes per model.
  - Losses improve: support ΔNLL −0.97 to −1.22; answer ΔNLL −0.46 to −0.71.
  - Greedy generated recall is **0.0%** for direct, paraphrased and delayed questions at all three scales.
  - A stronger "stress" update raises recall to 72.9% (direct and delayed) and 54.2% (paraphrased), but locality falls from 97.9% to 9.7%.
- **Protocol** (§6, Table 4):
  - Future-token loss only supports "Improves stream adaptation, target-domain adaptation, or task performance under the stated objective".
  - Matched baselines should include a no-update model, exact-context or long-context prompting, and retrieval or memory, matched on "update tokens, trainable parameters, optimization steps, latency, memory footprint, context length".
- Discussion (§7): "perplexity is already behavioral for language models" is "reasonable in dense-stream settings"; the claim only fails when used for sparse user memory.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Introduction / Scope) State our claim level explicitly: our evidence is S-level (stream and domain adaptation measured by perplexity), and we make no deployment-memory claim — Song et al. (2026).
- (Related Work) In-Place TTT and related TTT layers are coded as S-level evidence by Song et al. (2026); our carried state extends the same evidence type across documents.
- (Methodology) The reset-vs-stream comparison recommended by Song et al. (2026) is exactly our reset-per-document vs carried arm.
- (Methodology / Discussion) The no-update, in-context (exact-context) and memory baselines with matched budgets are the controls a memory-like claim needs.
- (Limitations) Perplexity gains can coexist with zero behavioural recall (Table 3), so our results do not show that the carry stores retrievable facts.

## Verification
- Read the full main text (pages 1–10, sections 1–8) from the PDF via pdftotext. All quotes and numbers above (claim-level definitions, Table 2 codings, Table 3 ΔNLL and 0.0% recall, the 72.9%/54.2%/97.9%→9.7% stress numbers, Table 4 rows) were read directly. Appendices A–F (search accounting, extra diagnostics) were not read.

# FILE: Test-Time Training with Next-Token Prediction_importance.md

# Test-Time Training with Next-Token Prediction

**Citation:** Xuan Ouyang, Zefan Cai, Junjie Hu (University of Wisconsin–Madison). arXiv:2606.21803v2 [cs.CL], 30 Aug 2026. Accepted to Findings of EMNLP 2026 (per the arXiv comment).

## Important parts for my research
- Direct successor of In-Place TTT. Same fast-weight site, the MLP down-projection of selected SwiGLU blocks ("Following Feng et al. (2026)"), and the same chunk-parallel rank-one writes. It changes only the **write target**.
- **Target**: the same layer's next-position contextual hidden state through a learned linear projection, v = W_proj h_{ℓ,t+1}. The write is ΔW = (W_proj h_{ℓ,t+1}) z_tᵀ (Eq. 5–7). This replaces In-Place TTT's "small learned convolution over a local window of hidden states".
- The training-time chunk write "sums bare outer products and drops the residual term that would appear in a literal least-squares regression step", because that term "is not prefix-summable" (§3.4). It is a Hebbian-style accumulation for parallelism, like ours.
- At inference it instead solves a ridge regression in closed form over the prompt (Eq. 16–17).
- **No persistence across inputs**: "We apply W_down ← W_down + ηΔW^CF, decode the answer, and restore the original weight after each sample" (§3.4). Adaptation is per prompt only.
- **Results**:
  - RULER Full-13, averaged over 4k–32k: +3.9 Llama-3.1-8B, +3.0 Mistral-7B, +4.1 Qwen3-4B, **+2.9 Qwen3-0.6B** over the released backbone.
  - LongBench-v2: +5.6 and +3.7.
  - Commonsense and knowledge benchmarks are roughly unchanged.
- **Key matched comparison**: "Under matched data, compute, fast-weight placement, and chunk size, the prior in-place TTT method does not improve over the released backbone on RULER" (§1). Continual pretraining alone and In-Place TTT "frequently degrade long-context retrieval" (§4.3).
- Target ablation: next-position supervision beats Past-5, Next-5 and Bi-dir-5 convolutional targets "by at least five RULER points at every evaluated length".

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) The closest follow-up to In-Place TTT keeps the down-projection fast weight but changes its write target to the next contextual hidden state, and still restores the original weight after every sample — Ouyang et al. (2026). Its adaptation never crosses inputs.
- (Related Work / Discussion) Under matched continual pretraining, the original In-Place TTT target did not improve RULER over the backbone (Ouyang et al., 2026). This is a caution that small TTT gains depend on the write target, consistent with our naive carry's null cross-document result.
- (Methodology) Like our Hebbian rule, its training-time write drops the error term to stay chunk-parallel (Ouyang et al., 2026, §3.4).
- (Related Work) It includes Qwen3-0.6B, the model we use (+2.9 RULER).

## Verification
- Read the abstract, introduction, related work, method (§1–3.4) and the experimental setup and baselines (§4.1–4.2) directly from the PDF via pdftotext. The quoted sentences, equation forms and the four RULER deltas were read directly. The per-length RULER tables and LongBench breakdown were only skimmed; the "+11.7 at 16k on Qwen3-4B" figure appears in §4.3 text.

# FILE: Can a Language Model Continually Learn Facts in its Weights?_importance.md

# Can a Language Model Continually Learn Facts in its Weights?

**Citation:** Charles O'Neill, Max Kirkby, Jonathon Liu, Michael Psenka (Base Labs). arXiv:2607.11020v3 [cs.CL], 5 Oct 2026. NeurIPS 2026 Workshop on Continual Learning in the Era of Foundation Models and Embodied Agents (per the PDF footer).

## Important parts for my research
- Studies continual learning in the **iterative regime**: 20 invented facts are written one at a time, each into "a Qwen3 model already modified by previous writes". Each write trains a fresh rank-16 LoRA (or full fine-tuning) that is merged before the next.
- Conditions:
  - Training data: **repeated statement** (the same fact in two formats) vs. **varied examples** (24 restatements, QA, applications, contrasts).
  - Method: SFT vs. off-policy and on-policy distillation.
  - Models: Qwen3-4B and 8B.
- Evaluation is **behavioural**: held-out questions of five types (recall, paraphrase, application, composition, counterfactual), graded strictly and leniently.
  - Original-model floor 4%; fact-in-prompt ceiling 91% (lenient).
  - 95% bootstrap intervals over facts, paired across conditions.
- **Results**:
  - After 20 sequential writes, strict accuracy on facts 1–19 is **1%** with repeated statements and **46%** with varied examples.
  - Distillation gives 13% (off-policy), 27% (on-policy forward KL) and 31% (reverse KL).
  - Retention is set by how the **later** facts are trained: varying the later writes changes retention by **+37.6 points [28.9, 46.3]**; varying the original fact's own data changes it by −2.4 [−9.2, 4.1].
  - The O-LoRA orthogonality penalty does not help (46.7% vs. 44.4%).
- "Behavioral forgetting of an earlier fact does not erase its presence from the log-probabilities". For facts that failed every question, a median of 69% of the write-induced log-probability increase remained (§3.4, Fig. 4).
- The writes are gradient-based and merged; there is no fast weight, test-time memory or document boundary.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Sequentially writing facts into weights with merged LoRA updates retains only 1–46% of earlier facts after 20 writes, and retention depends mostly on how later writes are made — O'Neill et al. (2026). A carried state faces the same interference problem.
- (Discussion) O'Neill et al. (2026) find that log-probability evidence of a written fact can remain after behaviour has failed, so likelihood gains such as ours do not imply usable knowledge.
- (Discussion) Interference from later writes is the parallel to our finding that a carry built over a long chain of documents harms the next document.

## Verification
- Read the abstract, introduction, problem formulation and results (§1–3.4, PDF pages 1–7) via pdftotext. All numbers above (1%/46%, 13/27/31%, +37.6 [28.9, 46.3], −2.4 [−9.2, 4.1], 46.7 vs 44.4, 4% floor, 91% ceiling, 69% median retained signal) were read directly. The discussion, appendices and second fact set were not read.

# FILE: Delayed Supervision for Test-Time Language Models_importance.md

# Delayed Supervision for Test-Time Language Models

**Citation:** Jinha Kim, Taksh Kothari (MIT CSAIL). arXiv:2609.32312v1 [cs.AI], 26 Sep 2026. Preprint.

## Important parts for my research
- Observes that next-token training "does not explicitly require a fact to remain accessible after many subsequent memory updates" in test-time memories (LaCT nonlinear fast weights, DeltaNet delta rule, RWKV-7).
- Adds a **delayed QA loss** during post-training, L = αL_NTP + βL_QA.
  - A question about the simulator state is asked "only after a long interval of unrelated events": at least 4,096 tokens and 16 intervening writes. DeltaNet gaps run 4,105 to 20,354 tokens (median 10,099).
  - Questions run on **disposable branches**, so answers never enter the stream.
  - The probes separate **retention** (a fact stays valid) from **revision** (answer with the latest value).
- "The native test-time update rule is unchanged: delayed supervision trains the slow parameters that shape writing, retention, and reading" (§1). Gradients from the delayed answer reach earlier writes: "the answer loss can train how evidence is encoded and survives subsequent updates" (§3.2).
- Data: TextWorld trajectories, 512 worlds, 128 outer updates.
- **Results** (Table 1, delayed QA vs. event-only NTP):

  | Model | BABILong | RULER single-needle |
  |---|---|---|
  | LaCT-760M | 31.08 vs. 25.60 | 88.85 vs. 87.39 |
  | DeltaNet-1.3B | 22.48 vs. 21.16, paired interval [−0.16, 2.84] | 83.52 vs. 80.24, [1.70, 4.97] |
  | RWKV-7 (separate panel) | +4.60 | +7.00 |

  - DeltaNet regresses at 1K and 2K. On the complementary panel, gains are "selective transfer rather than a general increase".
- Honest scoping: "The present comparisons do not identify delay as the sole cause of improvement". One training seed, and the intervals capture example variation, not seed variation.
- All persistence is within one continuous event stream, not across documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work / Discussion) Kim and Kothari (2026) argue that next-token training gives a weak signal for keeping information through many later memory updates. They add a delayed loss whose gradient reaches earlier writes, the same principle as our training through the carry, applied within one stream rather than across documents.
- (Discussion) Their retention vs. revision split is a useful frame for our carry: we measure transfer, not retention of specific facts.
- (Methodology) Precedent for honestly scoping a single-seed result whose cause is not isolated.

## Verification
- Read the full main text (§1–5.3, PDF pages 1–7) via pdftotext. All quoted sentences, delay thresholds, Table 1 scores, the DeltaNet bootstrap intervals and the per-length remarks were read directly. The related work, discussion and appendix after §5.3 were not read.

# FILE: Adapting Language Models to Compress Contexts_importance.md

# Adapting Language Models to Compress Contexts (AutoCompressors)

**Citation:** Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen (Princeton University). arXiv:2305.14788v2 [cs.CL], 4 Nov 2023. Published at EMNLP 2023 (main conference), pp. 3829–3846, doi:10.18653/v1/2023.emnlp-main.232.

## Important parts for my research
- **Method:**
  - Builds on the Recurrent Memory Transformer.
  - Fine-tunes pretrained LMs (OPT-1.3B/2.7B, Llama-2-7B) to emit κ "summary vectors" at the end of each segment. These are passed as soft prompts to later segments.
  - **Summary accumulation** concatenates the summaries of all earlier segments.
  - **Randomized segmenting** varies the segment length during training.
- Trained with plain next-token loss over the whole document, so the summaries learn to "store useful information" for future segments.
- **BPTT with stop-gradients**: summary vectors are cached "and stop their gradients after 2 compression steps", assuming "it is sufficient to predict the tokens in the adjacent S_{i+1}". This "incurs no penalty" and reduces memory (§3.1). This is the same design choice as our carry window of two slices.
- **Language modelling** (Table 1, OPT-2.7B, 2,048-token target): perplexity falls from 6.28 to 5.93 in domain (−5.6%) with 3 compressed segments (6,144 tokens) using only 150 summary vectors. Extended full attention reaches 5.94 with 2,048 extra tokens.
- **Cross-document use** (§6.1, retrieval-augmented LM):
  - Summary vectors of retrieved passages are fused, "despite a mismatch in training since we draw independent summary vectors from separate documents".
  - The model was "only ever trained to accumulate 3 sets of summary vectors, and yet it benefits from fusing the summary vectors of up to 10 documents".
  - With 50-token passages, fused summaries outperform fused plain passages and REPLUG; with 512-token passages, REPLUG top-10 still wins over fused summaries.
- Storage cost: compressing 10B tokens takes 5 TB per domain in half precision.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) AutoCompressors compress earlier segments into summary vectors trained by next-token loss and reuse them across documents in retrieval — Chevalier et al. (2023). Their carried object is a set of soft-prompt activations, not a fast weight.
- (Methodology) Chevalier et al. (2023) also stop gradients after two compression steps without loss of quality, the same choice as our carry window of two slices.
- (Discussion) As with our carry, AutoCompressors are trained within documents yet still help when summaries from independent documents are combined. Training within one unit does not preclude transfer across units.

## Verification
- Read the abstract, introduction, related work, method (§3) and the language-modelling evaluation (§4.1–4.2, Tables 1–3) directly via pdftotext. The retrieval section (§6.1, Table 5) was read through the extracted text, and its quoted sentences and the ranking of methods were read directly (Table 5's numbers were garbled in extraction, so no gains are quoted). In-context learning (§5) and passage re-ranking (§6.2) were not read. The EMNLP 2023 venue was confirmed on the ACL Anthology (https://aclanthology.org/2023.emnlp-main.232).

# FILE: Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass_importance.md

# Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass

**Citation:** Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Benjamin Van Durme, Luke Zettlemoyer, Jianfeng Gao, Hao Cheng (University of Washington, Microsoft). arXiv:2411.05877v1 [cs.LG], 8 Nov 2024.

## Important parts for my research
- Maps a context directly to a low-rank adapter added to a **frozen** LM, "using only forward passes", "inspired by fast weights (Ba et al., 2016; Schmidhuber, 1992)".
- **Generator:** for each adapted layer, W_Δ = (A₁A₂)(Σ_m h_m ⊗ h_m)(B₁B₂). This is an outer-product (Hebbian-like) accumulation of the base model's hidden states, projected by learned matrices (Eq. 1).
- **Streaming update:** S_t ← S_{t−1} + A₂H_tᵀH_tB₁, so the adapter accumulates over a stream of context chunks without storing them (Eq. 3–4). The accumulated adapter is used for later queries.
- **Stability:** "using the naive outer product for generating weights led to instability during training". The generated matrix sends inputs to near-zero or extremely large values "due to its skewed distribution of its singular values". The fix is **SVD normalization**, which resets all singular values to one (norm(M) = UVᵀ), keeping only the directions (§2.4). This parallels our finding that a read-time clip keeps only the direction of the carried state.
- **Training:** self-supervised on 1B SlimPajama tokens with reconstruction plus completion losses, then instruction tuning. Main experiments adapt only the attention output projection of Mistral-7B-Instruct or Llama2-7B-Chat.
- **Results:**
  - StreamingQA F1 of 31.5 vs. 19.5 for supervised fine-tuning, at contexts up to 32K.
  - MetaICL average 44.9 across 26 tasks.
  - **Multi-Session Conversation (MSC)**, which asks about a user's earlier sessions: closed-book F1 8.1, full-conversation prompting **66.0**, Generative Adapter **40.2**, at about a quarter of prompting's computation.
  - So prompting with the full history remains much more accurate.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Generative Adapter builds a context-dependent adapter in a single forward pass by accumulating outer products of hidden states over a stream of chunks, a learned fast weight carried across context chunks and sessions — Chen et al. (2024). Unlike our carry, it adapts attention output projections through a separately trained generator, and it is evaluated on question answering rather than perplexity.
- (Discussion) Chen et al. (2024) found naive outer-product weights unstable and normalised their singular values to one, keeping only directions. This is an independent case of a fast-weight memory that works with its magnitude discarded, like our saturated clip.
- (Discussion) On multi-session conversations, keeping the full history in context (F1 66.0) clearly beats the forward-pass adapter (40.2). This matches our finding that attention on earlier text is far stronger than a carried state.

## Verification
- Read the abstract, introduction and method (§1–2.4, PDF pages 1–5) and the experimental settings and document-QA setup (§3–4.1) directly via pdftotext. MSC Table 1 values (8.1, 66.0, 40.2) and the "4x" cost statement were read from the extracted text of §4.3. MetaICL details and §5 ablations beyond the first paragraph were not read. No venue is stated on the arXiv PDF.

# FILE: Cartridges: Lightweight and general-purpose long context representations via self-study_importance.md

# Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study

**Citation:** Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Ré (Stanford, Caltech, University at Buffalo). arXiv:2506.06266v3 [cs.CL], 13 Jun 2025.

## Important parts for my research
- **Problem:** serving many queries about the same large corpus with in-context learning is costly, because the KV cache grows with input length.
- **Method:** "training a smaller KV cache offline on each corpus" by backpropagating into key and value vectors, "essentially equivalent to prefix tuning". The trained cache (the **Cartridge**) is loaded at inference.
- **Naive next-token training fails at generality:**
  - it memorizes the corpus "perfectly using 107× less memory than the KV-cache";
  - but the resulting Cartridges "degrade the LM's ability to respond to diverse questions beyond regurgitating the corpus".
- **Self-Study fix:**
  - generate synthetic conversations about the corpus;
  - train with a context-distillation objective toward the model with the corpus in context.
- **Results:**
  - Matches ICL quality "while using 38.6× less memory and enabling 26.4× higher throughput".
  - Extends effective context, from 128k to 484k tokens on MTOB.
  - Cartridges "can be composed at inference time without retraining".
  - A KV-cache parameterisation beats a LoRA parameterisation.
- The memory is built per corpus by **offline gradient training**, amortised over many queries. It is not built during the forward pass and does not cross independent corpora by itself.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Cartridges trade compute for memory in the opposite way from a carried fast weight: a compact KV cache is trained offline per corpus and reused for many queries, matching in-context quality with 38.6× less memory — Eyuboglu et al. (2025).
- (Discussion) Eyuboglu et al. (2025) find that next-token training on the corpus alone memorizes but does not generalize; generality needed context distillation. This is a caution that a state trained only by next-token loss may carry surface statistics rather than usable knowledge.
- (Related Work) Compressed contexts can be composed across documents without retraining, a further point of comparison for cross-document transfer.

## Verification
- Read the abstract, introduction and the start of the preliminaries (PDF pages 1–3) directly via pdftotext. The quoted sentences and numbers (107×, 38.6×, 26.4×, 128k→484k, composability, KV vs LoRA) were read directly from that text. The method details (§4), full results tables and appendices were not read.

# FILE: MemoryLLM: Towards Self-Updatable Large Language Models_importance.md

# MemoryLLM: Towards Self-Updatable Large Language Models

**Citation:** Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, Julian McAuley (UC San Diego, Amazon, UCLA). Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235. arXiv:2402.04624v2.

## Important parts for my research
- **Design:**
  - Adds a fixed-size **memory pool** of hidden vectors ("memory tokens") to every layer of Llama2-7B, about 1B memory parameters in total.
  - The pool is the "self-updatable parameters" θ; the LLM weights φ stay static.
  - During generation every token attends to all memory tokens.
- **Self-update without backpropagation:**
  - New text is processed together with the last K memory tokens.
  - The resulting K new memory tokens replace K randomly dropped old ones, θ′ = U(θ, x).
  - Over a stream x₁…xₙ, θₙ = U(…U(θ, x₁)…, xₙ) (Eq. 1–2).
- **Built-in forgetting:** dropping K of N tokens per update gives exponential forgetting at rate K/N, "inspired by the Ebbinghaus Forgetting Curve" (§3.1.3).
- **Stated requirements** (§2.1): efficiency ("potentially eliminating the need for back-propagation"), efficacy, knowledge retention with a gradual phase-out, integrity, non-redundancy.
- **Results:**
  - Evaluated on model-editing benchmarks, long-context QA and custom retention tests.
  - "Operational integrity without any sign of performance degradation even after nearly a million memory updates" (abstract).
- Beyond Perplexity (2026) codes MemoryLLM as B-level evidence.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) MemoryLLM keeps a fixed-size pool of memory tokens in every layer, updated by forward passes over a stream of texts, with designed exponential forgetting — Wang et al. (2024). It persists state across many inputs, but as attended activations rather than as a fast weight inside the MLP.
- (Discussion) Its controlled forgetting rate (K/N per update) is a designed alternative to our fixed EMA decay of λ = 0.9, which harmed after long chains of unrelated documents.

## Verification
- Read the abstract, introduction, problem statement and structure design (§1–3.1.3, PDF pages 1–3) via pdftotext. The quoted sentences, the 1B pool size and the K/N forgetting argument were read directly. The training procedure, results tables and the "nearly a million updates" experiment details were not read beyond the abstract and introduction. The ICML 2024 venue is printed on the PDF.

# FILE: Larimar: Large Language Models with Episodic Memory Control_importance.md

# Larimar: Large Language Models with Episodic Memory Control

**Citation:** Payel Das, Subhajit Chaudhury, Elliot Nelson, Igor Melnyk, Sarathkrishna Swaminathan, Sihui Dai, Aurélie Lozano, Georgios Kollias, Vijil Chenthamarakshan, Jiří Navrátil, Soham Dan, Pin-Yu Chen (IBM AI Research, Princeton). Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, pp. 10109–10126. arXiv:2403.11901v4 [cs.LG], 21 Aug 2024.

## Important parts for my research
- Brain-inspired, after **complementary learning systems** theory: a hippocampus-like fast-learning **episodic memory** coupled to a neocortex-like slow-learning LLM decoder.
- **Memory write and read:**
  - Uses a Kanerva-Machine-style memory with "Generative Pseudo-inverse Memory".
  - "Reformulates these Bayesian updates as finding least-square solutions to linear systems" (§2–3), e.g. "Compute posterior memory M = W₀†Z_ξ".
  - The write is therefore error-correcting (a least-squares fit), unlike a Hebbian sum.
- **"Dynamic, one-shot updates of knowledge without the need for computationally expensive re-training or fine-tuning".**
- **Results:**
  - Fact-editing accuracy comparable to strong baselines, including sequential editing, with "speed-ups of 8-10x".
  - Also supports selective fact forgetting, information-leakage prevention, and input-context-length generalisation via recursive search (abstract, §1).
- The memory is explicit and external to the LLM weights, conditioning the decoder. It is aimed at fact editing, not at perplexity across documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Larimar adds an external episodic memory to an LLM, written in one shot by least-squares solutions rather than by gradient steps, for fast knowledge editing and forgetting — Das et al. (2024). Like the delta rule, its write corrects what is already stored, whereas our Hebbian carry only adds.
- (Discussion) Its explicit mechanisms for selective forgetting and leakage prevention show what a deployable carried memory would need, since our carry has none and a state carried across users could leak information.

## Verification
- Read the abstract, introduction and contributions (PDF pages 1–2) and searched the method text for the memory update (least-squares / pseudo-inverse lines read directly via pdftotext). The editing benchmarks, tables and appendix were not read. The arXiv PDF shows no venue; the ICML 2024 venue was confirmed on PMLR (https://proceedings.mlr.press/v235/das24a.html).

# FILE: Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention_importance.md

# Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

**Citation:** Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal (Google). arXiv:2404.07143v2 [cs.CL], 9 Aug 2024. Preprint, under review at the time.

## Important parts for my research
- Each attention layer keeps a **compressive memory**, an associative matrix per head, alongside local causal attention within a segment.
- The memory stores old KV states "instead of discarding them like in the standard attention mechanism" and passes them to the next segment, "building in a recurrence in each attention layer" (§3).
- **Update rules** (Eq. 8–9):
  - "Linear": M_s ← M_{s−1} + σ(K)ᵀV, with a normaliser z. This is "an associative binding operator (Smolensky, 1990; Hebb, 2005)", i.e. Hebbian.
  - "Linear + Delta": subtracts the retrieved value first, so the memory is left "unmodified if the KV binding already exists".
- **Mixing:** memory output and local attention are combined by a learned per-head gate, sigmoid(β) (Eq. 10). After training, heads specialise: near 0 (local), near 1 (memory), or mixers near 0.5.
- **Results:**
  - Beats Transformer-XL and Memorizing Transformers on long-context language modelling with "114x" less memory than a 65K-token retrieval memory.
  - 1M-token passkey retrieval with a 1B model.
  - 500K-token book summarisation with an 8B model.
  - Linear and Linear+Delta perform similarly; perplexity is 2.21 vs. 2.20 with 100K training.
- The memory is carried across segments of one long input, not across separate documents.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Infini-attention carries a compressive associative memory across the segments of one long input, written by a Hebbian or delta rule and mixed with local attention through a learned gate — Munkhdalai et al. (2024). Our carry is a similar fixed-size associative state, but it lives in the MLP fast weight and crosses document boundaries.
- (Discussion) Its learned per-head memory gate, which specialises towards memory or local attention, is a parallel to our separate, gated carry path.

## Verification
- Read the abstract, introduction, background and method (§1–3.1, PDF pages 1–5) via pdftotext. Eq. 8–10, the quoted sentences, the 114× memory figure, the gating-score observation and the 2.21/2.20 perplexities were read directly. Full result tables were skimmed only.

# FILE: Test-Time Training Done Right_importance.md

# Test-Time Training Done Right (LaCT)

**Citation:** Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan (MIT, Adobe Research). arXiv:2505.23884v1 [cs.LG], 29 May 2025.

## Important parts for my research
- **Large Chunk Test-Time Training (LaCT):**
  - Updates fast weights once per very large chunk, 2,048 to 1M tokens, instead of every 16–64 tokens.
  - Raises GPU utilisation "up to 70% on NVIDIA A100s" from "often below 5%".
  - Allows nonlinear fast-weight state of up to 40% of model size (a "state-to-parameter size ratio ≥ 40%", against 0.1%–5% before).
- **Update and apply:** W ← W − η∇_W L(f_W(k), v) and o = f_W(q), with a SwiGLU-MLP fast weight (§2.1, §3).
- **Stability is the central issue:** "Fast-weight updates in TTT repeatedly accumulate gradients, and thus suffer from magnitude explosion or decayed memory" (§3.2).
  - Their fix normalises the weights after every update, weight-update(W, g) = L2-Normalize(W − g), or uses a Muon update that normalises the gradient's spectral norm (Muon(g) ≈ UVᵀ).
  - "We do not use explicit weight-decay term as in previous methods."
  - Muon makes the learning rate "only reflect the relative importance of tokens within a chunk as Muon normalizes the absolute scale".
- **Language modelling:** chunk size is a hyperparameter (2,048 or 4,096), and sliding-window attention handles within-chunk causality. Results are "competitive ... compared to SoTA methods such as DeltaNet".
- **Other domains:** novel view synthesis with over 1M context, and a 14B autoregressive video diffusion model.
- Fast weights are per sequence; nothing is carried across sequences.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) LaCT shows that accumulated fast-weight updates "suffer from magnitude explosion or decayed memory" and fixes this by L2-normalising the fast weights after each update — Zhang et al. (2025). This is a designed bound that keeps only the direction, the same effect our read-time clip has once it saturates.
- (Related Work) Large-chunk TTT scales fast-weight state to 40% of model parameters, which is the regime where a carried state becomes as large as an attention cache.
- (Methodology) Precedent for treating a whole chunk of text as one fast-weight update, as In-Place TTT and our slices do.

## Verification
- Read the abstract, introduction, preliminaries and method (§1–3.3, PDF pages 1–6) and the language-modelling setup (§4.2) via pdftotext. The quoted sentences, Eq. 8–10, the 70%/5% utilisation and ≥40% state ratio figures were read directly. The language-model result tables (§5.2) and appendices were not read in detail.
