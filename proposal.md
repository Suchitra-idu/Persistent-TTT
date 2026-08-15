
Kandy Innovation Centre,
National Institute of Business Management.

UNDERGRADUATE RESEARCH PROPOSAL




Session-Persistent Test-Time Training for Language Models: Methodology and Empirical Analysis

M A S I Malimbada: KIC-BSCCSAI-25.1-P-014


Kandy Innovation Centre,
National Institute of Business Management,
1095, Kandy – Colombo Rd, Getambe.
Date of Submission: 12 July 2026

1. Details of the Individual Research Project 3
Proposed Study Title 3
Details of the Researcher 3
Details of Internal Supervisor(s) 3
Name of External Supervisor(s) 3
2. Introduction 4
Research Problem 4
Rationale 4
3. Objectives 5
Primary objective 5
Specific research objectives 5
4. Proposed research methodology 5
Research questions 5
Methods of data collection 5
Methods of analysis of data 6
Methods of implementation of the solution 6
Expected results and application 6
Methods of comparing models and results 7
Resources 7
5. Timeline 7
6. Status of permits and clearance 8
7. References 8
8. Annexture 8
Ethical clearance application 8



1. Details of the Individual Research Project
Proposed Study Title
Session-Persistent Test-Time Training for Language Models: Methodology and Empirical Analysis
Details of the Researcher
Name	M A S I Malimbada
Institutional Affiliation	Kandy Innovation Centre, National Institute of Business Management, 1095, Kandy – Colombo Rd, Getambe.
Student Registration Number	KIC-BSCCSAI-25.1-P-014
Telephone no.	0728623676

e-mail	KIC-BSCCSAI-251P-014@student.nibm.lk
Details of Internal Supervisor(s)
Name	Prof. Roshan D. Yapa
Institutional Affiliation(s)	University of PeradeniyaPeradeniya, 20400, Sri Lanka
Telephone no(s).	+94 81 239(4254)
e-mail(s)	dharshanayp@yahoo.com
Name of External Supervisor(s)
Name	
Institutional Affiliation(s)	
Telephone no(s).	
e-mail(s)	


2. Introduction
In Large Language Models (LLMs), a main issue is their inability to improve and learn at inference time. After training is completed, all responses come from the same parameters. Even though this method enables effective utilisation of modern GPU resources, it limits the model's ability to specialise to the specific context in which the model is being used.
The idea of Test-Time Training (TTT), which was first introduced for image classification under distribution shift (Sun et al., 2020), can be used to enable inference-time model improvement. TTT can update a model's parameters at inference time using self-supervised loss on the incoming input. Sun et al. (2024) first applied TTT to large language models. They introduced TTT-Linear and TTT-MLP, sequence layers where the hidden state improves as new tokens arrive. Titans introduces a neural long-term memory module that supplies attention with historical context (Behrouz, Zhong and Mirrokni, 2024).
Most prior research has attempted to use TTT to replace the attention mechanisms of LLMs. However, recent work such as In-Place TTT attempts to improve model performance at long context by updating the MLP down-projection with test-time training (Feng et al., 2026). However, In-Place TTT's fast weights reset at document boundaries, where learned fast weights do not persist to the next session. Therefore, whether learned fast weights can be carried to the next session is left for future work.
Research Problem
The research problem this study attempts to address is that existing test-time training methods only operate within a single document boundary. Currently, no published method attempts to implement session-persistent fast-weight training and meta-learned fast-weight initialisation. Addressing these problems could potentially enable continual learning and adaptation of large language models at inference time.
Rationale
There are multiple reasons that justify this research question. By integrating session-persistent state, we open new possibilities for enabling inference-time learning in large language models. It is also highly valuable to answer, if a session-persisten carry state is capable of increasing model capabilities substantially. And also, by characterising how session-persistent test-time adaptation compounds across different types of domains enables an understanding of which domains yield the highest marginal gains from TTT.



3. Objectives
Primary objective
Develop and empirically analyse a session-persistent test-time training method for pretrained language models.
Specific research objectives
·	O1. Design and develop a TTT architecture with cross-document persistence of fast weights.
·	O2. Implement a learned, per-source initialisation system to initialise fast weights.
·	O3. Conduct empirical analysis that characterises model behaviour across domains, session length, and key configuration choices.
4. Proposed research methodology
This research lies in the field of machine learning, specifically the field of decoder-only transformer language models. We will target models in the 0.6B to 4B parameter range. These are small enough to run multiple training and evaluation experiments while keeping compute cost low.
Research questions
·	Q1. How much does carry (information retained from the previous session) contribute to language model improvement?
·	Q2. Does domain-specific initialisation benefit model improvement, and if so, how much?
·	Q3. What kind of domains perform best with session-persistent TTT, and which domains benefit most from domain-specific initialisation?
Methods of data collection
In order to train our TTT architecture (O1), we require a dataset consisting of a large number of text documents. Specifically, each document must be at least 1024 tokens long. This is because, for any learning to occur, the document size must exceed the batch size of the TTT layers. To achieve objective 3 (O3) and answer Q3, the dataset must have documents across multiple domains.
Due to the popularisation of LLMs, a large number of multi-source language modelling datasets are publicly available. For this specific project, we identified SlimPajama as a suitable base corpus (Soboleva et al., 2023). It is a deduplicated, multi-source reproduction of the RedPajama dataset. However, due to the computational constraints of our experimental setup, we opted for the SlimPajama-6B subset, a 10% subsample of the first chunk of the full 627B-token corpus (Yoon, 2023). This selection is mainly because the dataset consists of documents from multiple sources, such as ArXiv, Books, and GitHub, and is already labelled according to source. This multi-source setup is essential for O3 and for answering Q3.
Methods of analysis of data
Because the chosen dataset is already deduplicated and labelled, only minimal work is required at the analysis and preprocessing stage. To ensure the dataset is in the required shape and format, we will go through random samples of the dataset. Even though the dataset consists of 6 billion tokens, our research requires only a small subset of that corpus. Due to this high data redundancy, it is possible to filter the dataset with minimal negative impact. For this research, we require each document to be at least 1024 tokens long. Therefore, the dataset will be filtered accordingly. The subset will then be chosen to match the desired ratio of sources. This filtering and dataset quality control is essential for O1, O3, and for answering Q3, because training and domain-specific empirical verification are highly dependent on dataset quality and proper domain representation.
For answering Q1 and Q2, it is highly important to have a contamination-free evaluation. To ensure this, we create another subset for evaluation and testing, which is not used for any training. This subset follows the exact same procedure as the training set, but a uniform ratio between sources will be enforced.
Methods of implementation of the solution
Our method will be implemented on top of the existing In-Place Test-Time Training architecture, utilising its chunk-parallel rank-one updates to the MLP down-projection of selected transformer layers (Feng et al., 2026). However, to enable effective cross-session persistence (O1) and learned initialisation (O2), their architecture will be substantially modified. To enable proper empirical analysis (O3), three logically independent components will be exposed.
·	Within-document adaptation — the standard within-forward chunk-scan update. This does not have any session persistence (O3, Q3).
·	Cross-document persistence — a mechanism that carries the fast weight across documents instead of resetting it per document (O1, O3, Q1, Q3).
·	Meta-learned per-source initialisation — learned, per-source fast weights that persist during training and are used as the initialisation of the persistent state (O2, Q2, O3, Q3).
The training loop will support toggling each component independently. This is to enable empirical evaluation of each component and architectural modification across different domains.
Expected results and application
·	For O1 — a working model that utilises session-persistent test-time training.
·	For O2 — learned initialisation weights for each selected domain.
·	For O3 — results of the cross-domain, cross-component empirical analysis.

Methods of comparing models and results
Our evaluation pipeline will consist of two parts. The first is comparing the three components (fresh, within-only, session-persistent) with each other across configurations and domains. This will answer Q1, Q2, and Q3. The second part is comparing this system with other work in the field and benchmarking against it. Our session-persistent TTT will be compared against methods such as In-Place TTT (Feng et al., 2026) and benchmarked against the RULER-8k benchmark (Hsieh et al., 2024). This is to identify where the proposed methodology for session-persistent TTT competes against other methodologies in the field.
Resources
Compute: Modal cloud GPU credits, estimated at ~USD 40 to 60 for the full set of runs, available through the researcher's existing account.
·	Models: Publicly available pretrained Qwen 3 accessed via Hugging Face.
·	Datasets: SlimPajama-6B (public).
·	Personnel: The researcher, with internal supervisor guidance.
·	Ethics approval: Not required. The study uses only publicly available textual corpora.
Due to the utilization of existing cloud compute credits, research funding will not be required.
5. Timeline
The timeline spans from early July to the end of October. Majority of time is given to Literature review and Evaluation. It is due to the complex literature landscape around LLMs and having all our research questions (Q1,Q2,Q3) answered at evaluation step. 

6. Status of permits and clearance
The study uses only publicly available textual corpora and pretrained model checkpoints. No permits or ethical clearance are required.
7. References
Feng, G., Luo, S., Hua, K., Zhang, G., He, D., Huang, W., and Cai, T. (2026) In-Place Test-Time Training.  
Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. (2024) RULER: What’s the Real Context Size of Your Long-Context Language Models? arXiv:2404.06654. arXiv. 
Soboleva, D., Al Khateeb, F., Myers, R., Steeves, J.R., Hestness, J. and Dey, N. (2023) SlimPajama a 627B token, cleaned and deduplicated version of RedPajama. Cerebras. Available at https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama (Accessed 8 July 2026). 
Yoon, D. (2023) SlimPajama 6B. Hugging Face. Available at https://huggingface.co/datasets/DKYoon/SlimPajama-6B (Accessed 8 July 2026)..
Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., and Guestrin, C. (2024) Learning to (Learn at Test Time): RNNs with Expressive Hidden States.  
Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A.A., and Hardt, M. (2020) Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. arXiv:1909.13231.

8. Annexture
Ethical clearance application
Ethical clearance application will be attached according to the template.

