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
