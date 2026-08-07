"""Frequent words extraction, ported from NVIDIA/RULER (Apache-2.0)
freq_words_extraction.py. Zipf weights computed directly (no `scipy.special.zeta`
dependency) since the vocabulary is finite and only needs to sum to 1 over it.
"""

from __future__ import annotations

from ttt.core.ruler_scoring import recall_score
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks import _common
from ttt.extensions.ruler_tasks._registry import register

NAME = "fwe"
TEMPLATE = (
    "Read the following coded text and track the frequency of each coded "
    "word. Find the three most frequently appeared coded words. {context}\n"
    "Question: Do not provide any explanation. Please ignore the dots "
    "'....'. What are the three most frequently appeared words in the above "
    "coded text?"
)
ANSWER_PREFIX = (
    " Answer: According to the coded text above, the three most frequently "
    "appeared words are:"
)
TOKENS_TO_GENERATE = 50
VOCAB_SIZE = 200
CODE_LEN = 6
ALPHA = 2.0
NOISE_TOKEN = "...."


def _vocab(rng) -> list[str]:
    words: set[str] = set()
    while len(words) < VOCAB_SIZE:
        words.add("".join(chr(ord("a") + rng.integers(0, 26)) for _ in range(CODE_LEN)))
    return [NOISE_TOKEN] + sorted(words)[1:]


def _weights(n: int) -> list[float]:
    return [1.0 / (rank**ALPHA) for rank in range(1, n + 1)]


class _FreqWordsExtraction:
    name = NAME
    version = 1

    def build(
        self, rng, tokenizer, seq_len: int, ctx: RulerContext
    ) -> RulerExample:
        vocab = _vocab(rng)
        weights = _weights(len(vocab))
        total = sum(weights)
        targets = tuple(vocab[1:4])

        def make(num_words: int) -> tuple[str, tuple[str, ...]]:
            counts = [max(1, round(num_words * w / total)) for w in weights]
            flat = [word for word, count in zip(vocab, counts) for _ in range(count)]
            rng.shuffle(flat)
            return TEMPLATE.format(context=" ".join(flat)), targets

        prompt, targets = _common.fit_size(
            make, tokenizer=tokenizer, seq_len=seq_len, reserve=TOKENS_TO_GENERATE
        )
        return RulerExample(
            task=NAME,
            length_bucket=seq_len,
            prompt=prompt,
            answer_prefix=ANSWER_PREFIX,
            targets=targets,
        )

    def score(self, prediction: str, example: RulerExample) -> float:
        return recall_score(prediction, example.targets)


FREQ_WORDS_EXTRACTION = register(_FreqWordsExtraction())
