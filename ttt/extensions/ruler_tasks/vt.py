"""Variable tracking, ported from NVIDIA/RULER (Apache-2.0) variable_tracking.py."""

from __future__ import annotations

from ttt.core.ruler_scoring import recall_score
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks import _common
from ttt.extensions.ruler_tasks._registry import register

NAME = "vt"
TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above."
)
ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)
CHUNK_WORDS = 20
TOKENS_TO_GENERATE = 30
NUM_CHAINS = 3
CHAIN_LEN = 4
NUM_DIGITS = 7


def _var_name(rng, index: int) -> str:
    letters = "".join(chr(ord("A") + rng.integers(0, 26)) for _ in range(4))
    return f"VAR_{letters}{index}"


def _chain(rng, index: int) -> tuple[list[str], str]:
    value = str(rng.integers(10 ** (NUM_DIGITS - 1), 10**NUM_DIGITS))
    names = [_var_name(rng, index * CHAIN_LEN + hop) for hop in range(CHAIN_LEN)]
    lines = [f"VAR {names[0]} = {value}"]
    lines += [f"VAR {a} = VAR {b}" for a, b in zip(names[1:], names)]
    return lines, value


def _chunks(words: list[str]) -> list[str]:
    return [
        " ".join(words[i : i + CHUNK_WORDS])
        for i in range(0, len(words), CHUNK_WORDS)
    ]


class _VariableTracking:
    name = NAME
    # v2: chain steps used to get scrambled by a flat shuffle across all
    # chains, making some examples unanswerable regardless of the model.
    version = 2

    def build(
        self, rng, tokenizer, seq_len: int, ctx: RulerContext
    ) -> RulerExample:
        chains = [_chain(rng, i) for i in range(NUM_CHAINS)]
        queried = rng.integers(0, NUM_CHAINS)
        query = chains[queried][1]
        targets = tuple(_extract_names(chains[queried][0]))
        groups = [lines for lines, _ in chains]
        total_lines = sum(len(g) for g in groups)

        def make(num_words: int) -> tuple[str, tuple[str, ...]]:
            chunks = _chunks(_common.repeated(ctx.haystack_words, num_words))
            context = _common.interleave_groups(rng, chunks, groups)
            return TEMPLATE.format(context=context, query=query), targets

        prompt, targets = _common.fit_size(
            make,
            tokenizer=tokenizer,
            seq_len=seq_len,
            reserve=TOKENS_TO_GENERATE,
            start=CHUNK_WORDS * (total_lines + 2),
        )
        return RulerExample(
            task=NAME,
            length_bucket=seq_len,
            prompt=prompt,
            answer_prefix=ANSWER_PREFIX.format(num_v=len(targets), query=query),
            targets=targets,
        )

    def score(self, prediction: str, example: RulerExample) -> float:
        return recall_score(prediction, example.targets)


def _extract_names(chain_lines: list[str]) -> list[str]:
    return [line.split()[1] for line in chain_lines]


VARIABLE_TRACKING = register(_VariableTracking())
