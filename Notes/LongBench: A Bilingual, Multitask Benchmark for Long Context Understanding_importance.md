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
