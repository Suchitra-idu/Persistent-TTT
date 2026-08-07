"""Common words extraction, ported from NVIDIA/RULER (Apache-2.0)
common_words_extraction.py. Vocabulary comes from `RulerContext.haystack_words`
rather than `wonderwords` + a vendored word list, same reasoning as `niah.py`.
"""

from __future__ import annotations

from ttt.core.ruler_scoring import recall_score
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks import _common
from ttt.extensions.ruler_tasks._registry import register

NAME = "cwe"
TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more "
    "often than others. Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the 10 most common words in the above list?"
)
ANSWER_PREFIX = " Answer: The top 10 words that appear most often in the list are:"
TOKENS_TO_GENERATE = 120
COMMON_COUNT = 10
COMMON_MULTIPLIER = 5


def _vocab(ctx: RulerContext) -> list[str]:
    pool = sorted({w.lower() for w in ctx.haystack_words if w.isalpha() and len(w) >= 3})
    if len(pool) <= COMMON_COUNT:
        raise ValueError("haystack_words has too few distinct words for cwe")
    return pool


class _CommonWordsExtraction:
    name = NAME
    version = 1

    def build(
        self, rng, tokenizer, seq_len: int, ctx: RulerContext
    ) -> RulerExample:
        vocab = _vocab(ctx)
        common = _common.sample(rng, vocab, COMMON_COUNT)
        noise_pool = [w for w in vocab if w not in common]

        def make(num_noise: int) -> tuple[str, tuple[str, ...]]:
            repeat_common = -(-num_noise // len(noise_pool)) * COMMON_MULTIPLIER
            words = list(common) * repeat_common + _common.repeated(
                noise_pool, num_noise
            )
            rng.shuffle(words)
            context = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
            return TEMPLATE.format(context=context), tuple(common)

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


COMMON_WORDS_EXTRACTION = register(_CommonWordsExtraction())
