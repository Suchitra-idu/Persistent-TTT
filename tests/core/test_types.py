"""The core vocabulary — mostly, the states these types make unrepresentable."""

from __future__ import annotations

import pytest
import torch

from tests.core import _builders as build
from ttt.core.types import Carry, Session, WorkItem


def test_a_work_item_knows_its_token_count():
    assert WorkItem(doc_idx=0, start=100, end=350).n_tokens == 250


def test_an_empty_work_item_is_rejected():
    """A zero-token item would be a forward pass with nothing to predict."""
    with pytest.raises(ValueError, match="empty work item"):
        WorkItem(doc_idx=0, start=10, end=10)


def test_a_backwards_work_item_is_rejected():
    with pytest.raises(ValueError, match="empty work item"):
        WorkItem(doc_idx=0, start=50, end=10)


def test_a_session_sums_the_tokens_of_its_items():
    session = Session(items=tuple(build.work_items((0, 0, 100), (0, 100, 250))))

    assert session.n_tokens == 250


def test_an_empty_session_is_rejected():
    """A session boundary with no work would reset the carry for nothing."""
    with pytest.raises(ValueError, match="at least one work item"):
        Session(items=())


def test_a_carry_is_keyed_by_base_model_layer_index():
    carry = Carry(deltas={1: torch.zeros(2, 2), 3: torch.zeros(2, 2)})

    assert carry.layer_indices == (1, 3)


