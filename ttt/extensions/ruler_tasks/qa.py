"""QA-in-a-haystack, ported from NVIDIA/RULER (Apache-2.0) qa.py. Documents
come from `RulerContext.qa_pairs` (a real QA set, e.g. SQuAD, pulled once at
prepare time) rather than a vendored `squad.json` copy.
"""

from __future__ import annotations

from ttt.core.ruler_scoring import qa_score
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks import _common
from ttt.extensions.ruler_tasks._registry import register

NAME = "qa"
TEMPLATE = (
    "Answer the question based on the given documents. Only give me the "
    "answer and do not output any other words.\n\nThe following are given "
    "documents.\n\n{context}\n\nAnswer the question based on the given "
    "documents. Only give me the answer and do not output any other words."
    "\n\nQuestion: {query}"
)
DOCUMENT = "Document {i}:\n{document}"
ANSWER_PREFIX = " Answer:"
TOKENS_TO_GENERATE = 32


class _Qa:
    name = NAME
    version = 1

    def build(
        self, rng, tokenizer, seq_len: int, ctx: RulerContext
    ) -> RulerExample:
        if not ctx.qa_pairs:
            raise ValueError("RulerContext has no qa_pairs for the qa task")
        target = ctx.qa_pairs[rng.integers(0, len(ctx.qa_pairs))]
        distractors = [p.context for p in ctx.qa_pairs if p is not target]

        def make(num_docs: int) -> tuple[str, tuple[str, ...]]:
            docs = [target.context] + _common.repeated(
                distractors or [target.context], max(0, num_docs - 1)
            )
            rng.shuffle(docs)
            context = "\n\n".join(
                DOCUMENT.format(i=i + 1, document=doc) for i, doc in enumerate(docs)
            )
            prompt = TEMPLATE.format(context=context, query=target.query)
            return prompt, target.answers

        prompt, targets = _common.fit_size(
            make, tokenizer=tokenizer, seq_len=seq_len, reserve=TOKENS_TO_GENERATE, start=2
        )
        return RulerExample(
            task=NAME,
            length_bucket=seq_len,
            prompt=prompt,
            answer_prefix=ANSWER_PREFIX,
            targets=targets,
        )

    def score(self, prediction: str, example: RulerExample) -> float:
        return qa_score(prediction, example.targets)


QA = register(_Qa())
