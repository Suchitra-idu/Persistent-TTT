2. Introduction
In Large Language Models (LLMs), a main issue is their inability to improve and learn at inference
time. After training is completed, all responses come from the same parameters. Even though this
method enables effective utilisation of modern GPU resources, it limits the model's ability to
specialise to the specific context in which the model is being used.
The idea of Test-Time Training (TTT), which was first introduced for image classification under
distribution shift (Sun et al., 2020), can be used to enable inference-time model improvement. TTT
can update a model's parameters at inference time using self-supervised loss on the incoming input.
Sun et al. (2025) first applied TTT to large language models. They introduced TTT-Linear and
TTT-MLP, sequence layers where the hidden state improves as new tokens arrive. Titans
introduces a neural long-term memory module that supplies attention with historical context
(Behrouz, Zhong and Mirrokni, 2025).
Most prior research has attempted to use TTT to replace the attention mechanisms of LLMs.
However, recent work such as In-Place TTT attempts to improve model performance at long
context by updating the MLP down-projection with test-time training (Feng et al., 2026). However,
In-Place TTT's fast weights reset at document boundaries, where learned fast weights do not persist
to the next session. Therefore, whether learned fast weights can be carried to the next session is left
for future work.
Research Gap
Recent work has approached test-time persistence from different angles. Tandon et al. (2025) meta-
learn a test-time-training initialisation for long-context modelling, but this operates over a single
continuous context through sliding-window attention, not across independently-bounded
documents. Wang et al. (2026) retain information beyond a local window through a context
distillation objective, where a teacher supervises what a student model should preserve, rather than
through the reconstruction objective that In-Place TTT's fast weights use. Bansal et al. (2025) apply
gradient updates within a single long context to sharpen retrieval on Qwen3 models, the same
model family this proposal targets, but again within one context rather than across sessions.
McDermott, Heath and Parhi (2026) place fast-weight programmers, state-space models, and TTT
layers within one family of parametric attention for lifelong context, though this is an architectural
framing rather than an empirical persistence mechanism. Transformer-XL carries hidden
representations across segment boundaries through cached activations rather than through an
updated weight (Dai et al., 2019). None of this work carries a feed-forward fast weight across
independently-bounded documents. In-Place TTT, the closest starting point for this kind of
mechanism, resets its fast weights at every document boundary by its own authors' account (Feng
et al., 2026).Research Problem
The research problem this study attempts to address is that existing test-time training methods only
operate within a single document boundary. Currently, no published method attempts to implement
session-persistent fast-weight training. Addressing this problem could potentially enable continual
learning and adaptation of large language models at inference time.
Rationale
There are multiple reasons that justify this research question. By integrating session-persistent
state, we open new possibilities for enabling inference-time learning in large language models. It
is also highly valuable to answer, if a session-persisten carry state is capable of increasing model
capabilities substantially. And also, by characterising how session-persistent test-time adaptation
compounds across different types of domains enables an understanding of which domains yield
the highest marginal gains from TTT.
3. Objectives
Primary objective
Develop and empirically analyse a session-persistent test-time training method for pretrained
language models.
4. Proposed research methodology
This research lies in the field of machine learning, specifically the field of decoder-only transformer
language models. We will target models in the 0.6B to 4B parameter range. These are small enough
to run multiple training and evaluation experiments while keeping compute cost low.
Research questions


Q1. How much does carry (information retained from the previous session) contribute to
language model improvement?
Q2. What kind of domains perform best with session-persistent TTT?
Methods of data collection
In order to train our TTT architecture, we require a dataset consisting of a large number of text
documents. Specifically, each document must be at least 1024 tokens long. This is because, for any
learning to occur, the document size must exceed the batch size of the TTT layers. To answer Q2,
the dataset must have documents across multiple domains.
Due to the popularisation of LLMs, a large number of multi-source language modelling datasets
are publicly available. For this specific project, we identified SlimPajama as a suitable base corpus
(Soboleva et al., 2023). It is a deduplicated, multi-source reproduction of the RedPajama dataset.
However, due to the computational constraints of our experimental setup, we opted for the
SlimPajama-6B subset, a 10% subsample of the first chunk of the full 627B-token corpus(Yoon,2023). This selection is mainly because the dataset consists of documents from multiple
sources, such as ArXiv, Books, and GitHub, and is already labelled according to source. This multi-
source setup is essential for answering Q2.
Methods of analysis of data
Because the chosen dataset is already deduplicated and labelled, only minimal work is required at
the analysis and preprocessing stage. To ensure the dataset is in the required shape and format, we
will go through random samples of the dataset. Even though the dataset consists of 6 billion tokens,
our research requires only a small subset of that corpus. Due to this high data redundancy, it is
possible to filter the dataset with minimal negative impact. For this research, we require each
document to be at least 1024 tokens long. Therefore, the dataset This filtering and dataset quality
control is essential for answering Q2, because training and domain-specific empirical verification
are highly dependent on dataset quality and proper domain representation.
For answering Q1, it is highly important to have a contamination-free evaluation. To ensure this,
we create another subset for evaluation and testing, which is not used for any training. This subset
follows the exact same procedure as the training set, but a uniform ratio between sources will be
enforced.
Methods of implementation of the solution
Our method will be implemented as an extension of test-time training approaches that adapt a
model's feed-forward weights at inference time, in the general direction explored by In-Place TTT
(Feng et al., 2026). The specific update rule and the layers it targets will be finalised during
implementation, since committing to one exact mechanism at the proposal stage would
unnecessarily limit the design space. To enable proper empirical analysis, two logically
independent components will be exposed.


Within-document adaptation, the standard within-document update. This does not have any
session persistence (Q2).
Cross-document persistence, a mechanism that carries the fast weight across documents
instead of resetting it per document (Q1, Q2).
The training loop will support toggling each component independently. This is to enable empirical
evaluation of each component and architectural modification across different domains.
Expected results and application


A working model that utilises session-persistent test-time training.
Results of the cross-domain, cross-component empirical analysis.
Methods of comparing models and results
Our evaluation is built around one central question, whether inference-time performance improves
over the course of a session because of carry, in the same sense that dynamic evaluation showedinference-time gradient updates improve perplexity over a stream (Krause et al., 2019). Our
contribution asks the same question of a fast-weight mechanism instead of a full-parameter update.
The evaluation pipeline compares the two components, within-document adaptation and cross-
document persistence, with each other across configurations and domains. This answers Q1 and
Q2. A held-out document is replayed across repeated sessions while the carry is left to accumulate,
and perplexity is tracked across the repeats to measure whether it improves and where the
improvement saturates. The same measurement is repeated with the carry reset between repeats as
a baseline, so the gain attributable to persistence can be isolated. Before this gain is trusted, the
carry itself is checked for stability across a long stream of sessions, since a carry that grows without
bound would produce a misleading result rather than a real improvement. A further check applies
a carry trained on one source while evaluating on another, to confirm that any benefit is source-
specific rather than incidental.
This evaluation is deliberately scoped to inference-time improvement rather than a claim that the
model remembers specific information after it leaves context. A perplexity trend on its own does
not establish that a model has retained and can later use information, and this study does not attempt
evidence at that level, which would require behavioural testing beyond what this timeframe
supports. Individual RULER tasks are used as one additional source of held-out probes rather than
as a benchmark to be matched against published results (Hsieh et al., 2024).
Resources
Compute: Modal cloud GPU credits, estimated at ~USD 40 to 60 for the full set of runs, available
through the researcher's existing account.




Models: Publicly available pretrained Qwen 3 accessed via Hugging Face.
Datasets: SlimPajama-6B (public).
Personnel: The researcher, with internal supervisor guidance.
Ethics approval: Not required. The study uses only publicly available textual corpora.
Due to the utilization of existing cloud compute credits, research funding will not be required.
5. Timeline
The timeline spans from early July to the end of October. Majority of time is given to Literature
review and Evaluation. It is due to the complex literature landscape around LLMs and having all
our research questions (Q1,Q2) answered at evaluation step.6. Status of permits and clearance
The study uses only publicly available textual corpora and pretrained model checkpoints. No
permits or ethical clearance are required.
7. References
Feng, G., Luo, S., Hua, K., Zhang, G., He, D., Huang, W. and Cai, T. (2026) In-Place Test-Time
Training. Proceedings of the International Conference on Learning Representations (ICLR
2026).
Bansal, R., Zhang, A., Tiwari, R., Madaan, L., Duvvuri, S.S., Khatri, D., Brandfonbrener, D.,
Alvarez-Melis, D., Bhargava, P., Kale, M.S. and Jelassi, S. (2025) Let's (not) just put things
in Context: Test-Time Training for Long-Context LLMs. arXiv:2512.13898. arXiv.
Behrouz, A., Zhong, P. and Mirrokni, V. (2025) Titans: Learning to Memorize at Test Time.
Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q.V. and Salakhutdinov, R. (2019) Transformer-
XL: Attentive Language Models Beyond a Fixed-Length Context. Proceedings of the 57th
Annual Meeting of the Association for Computational Linguistics (ACL 2019).
French, R.M. (1999) Catastrophic Forgetting in Connectionist Networks. Trends in Cognitive
Sciences, 3(4), pp. 128 to 135.
Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y. and Ginsburg, B.
(2024) RULER: What's the Real Context Size of Your Long-Context Language Models?
Proceedings of the First Conference on Language Modeling (COLM 2024).Krause, B., Kahembwe, E., Murray, I. and Renals, S. (2018) Dynamic Evaluation of Neural
Sequence Models. Proceedings of the 35th International Conference on Machine Learning,
PMLR 80, pp. 2766 to 2775.
Maharana, S.K., Mishra, S., Zhang, Y., Niu, S., Rafi, T.H., Hamm, J., Pedersoli, M., Dolz, J. and
Guo, Y. (2026) Continual Test-Time Adaptation in Computer Vision: Methods,
Benchmarks, and Future Directions. arXiv:2607.08164. arXiv.
McDermott, L., Heath, R.W. and Parhi, R. (2026) Lifelong In-Context Learning with Transformers
Requires Parametric Forms of Attention. arXiv:2606.25342. arXiv.
Soboleva, D., Al Khateeb, F., Myers, R., Steeves, J.R., Hestness, J. and Dey, N. (2023) SlimPajama
a 627B token, cleaned and deduplicated version of RedPajama. Cerebras. Available at
https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-
version-of-redpajama (Accessed 8 July 2026).
Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A.A. and Hardt, M. (2020) Test-Time Training with
Self-Supervision for Generalization under Distribution Shifts. Proceedings of the 37th
International Conference on Machine Learning, PMLR 119, pp. 9229 to 9248.
Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo,
S., Hashimoto, T. and Guestrin, C. (2025) Learning to (Learn at Test Time): RNNs with
Expressive Hidden States. Proceedings of the 42nd International Conference on Machine
Learning.
Tandon, A., Dalal, K., Li, X., Koceja, D., Rød, M., Buchanan, S., Wang, X., Leskovec, J., Koyejo,
S., Hashimoto, T., Guestrin, C., McCaleb, J., Choi, Y. and Sun, Y. (2025) End-to-End Test-
Time Training for Long Context. arXiv:2512.23675. arXiv.
Wang, Z., Dang, X., Zhu, R.-J., Wen, Z., Fu, H., Chai, W. and Lee, J.D. (2026) Learning What to
Remember: Test-Time Training via Context Distillation. arXiv:2608.01672. arXiv.
Yoon,
D.
(2023)
SlimPajama
6B.
Hugging
Face.
Availabl