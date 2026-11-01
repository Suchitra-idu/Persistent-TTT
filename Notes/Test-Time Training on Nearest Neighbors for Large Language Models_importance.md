# Test-Time Training on Nearest Neighbors for Large Language Models

**Citation:** Moritz Hardt, Yu Sun. ICLR 2024. arXiv:2305.18466.

## Important parts for my research
- **Method (TTT-NN):** for each test input, retrieve its nearest neighbours from a distributed embedding index of the Pile and fine-tune the model on their text, one gradient iteration per neighbour (long neighbours split into chunks). Main experiments use 50 neighbours; the abstract reports gains with "as few as 20 neighbors, each for only one gradient iteration".
- **Reset per input:** after each test instance "we reset its trainable parameters to their original states". Nothing is carried between inputs.
- **Models and optimiser:** GPT-2 Small (117M), GPT-2 Large (774M), GPT-Neo (1.3B); Adam, lr 2e-5 (GPT-2) and 5e-6 (GPT-Neo), default HuggingFace settings without tuning.
- **Results:** improves more than 20 Pile language-modelling tasks. GPT-2 Large bits per byte, base to TTT-NN: pile_github 1.95 to 0.51, pile_enron 1.44 to 0.72, pile_all 1.07 to 0.85. It narrows the gap between small GPT-2 and a GPT-Neo model more than 10 times larger.
- **Relatedness matters:** "Sufficient index quality and size, however, are necessary"; training from the nearest to the farthest neighbour works well, the reverse order "performs significantly worse". Unrelated or random training text is not tested.

## What to cite (This is NOT a rule and you do not have to cite it with this prasing, Just a suggesion and only use it like that If it matches the paper)
- (Related Work) Test-time training on related text improves language modelling, with gradient steps on the slow weights and a reset after every input — Hardt and Sun (2024). Our carry is built by forward passes in a fast weight and is not reset.
- (Discussion) Precedent that the benefit of test-time adaptation depends on how close the adaptation text is to the target, as in our relatedness gradient.

## Verification
- Read the arXiv abstract and the HTML version (method, reset sentence, Table 1, appendix hyperparameters) on 2026-11-01. The PDF was not read.
