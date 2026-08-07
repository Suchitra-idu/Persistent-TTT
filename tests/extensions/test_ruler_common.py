from __future__ import annotations

from ttt.adapters.numpy_rng import NumpyRng
from ttt.extensions.ruler_tasks import _common


def test_interleave_groups_preserves_each_groups_own_order():
    chunks = [f"chunk{i}" for i in range(20)]
    groups = [["a1", "a2", "a3"], ["b1", "b2"], ["c1"]]

    text = _common.interleave_groups(NumpyRng(0), chunks, groups)
    words = text.split()

    for group in groups:
        positions = [words.index(item) for item in group]
        assert positions == sorted(positions)


def test_interleave_groups_includes_every_item_exactly_once():
    chunks = [f"chunk{i}" for i in range(10)]
    groups = [["a1", "a2"], ["b1", "b2", "b3"]]

    text = _common.interleave_groups(NumpyRng(1), chunks, groups)

    for item in ("a1", "a2", "b1", "b2", "b3"):
        assert text.count(item) == 1


def test_interleave_groups_tolerates_an_empty_group():
    text = _common.interleave_groups(NumpyRng(0), ["x", "y"], [[], ["only"]])

    assert "only" in text
