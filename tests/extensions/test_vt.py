"""vt-specific: a chain's own steps must stay in order (see _common.py's
interleave_groups) — the bug this guards against made every step's position
random, so `VAR C = VAR B` could precede `VAR B = ...` and be unanswerable."""

from __future__ import annotations

from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.numpy_rng import NumpyRng
from ttt.core.ruler_types import RulerContext
from ttt.extensions.ruler_tasks.vt import VARIABLE_TRACKING

SEQ_LEN = 4000


def _context() -> RulerContext:
    words = tuple(f"filler{i}" for i in range(4000))
    return RulerContext(haystack_words=words)


def test_a_chains_own_steps_stay_in_order_across_many_draws():
    for seed in range(20):
        example = VARIABLE_TRACKING.build(NumpyRng(seed), FakeTokenizer(), SEQ_LEN, _context())

        positions = [example.prompt.index(f"VAR {name} =") for name in example.targets]

        assert positions == sorted(positions)
