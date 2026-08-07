"""The core vocabulary — mostly, the states these types make unrepresentable."""

from __future__ import annotations

import pytest
import torch

from tests.core import _builders as build
from ttt.core.types import Carry, DocRef, EvalRow, PplRow, Session, SliceRow, WorkItem


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


def test_a_session_reports_its_documents():
    session = Session(items=tuple(build.work_items((3, 0, 10), (7, 0, 10))))

    assert session.doc_indices == (3, 7)


def test_an_empty_session_is_rejected():
    """A session boundary with no work would reset the carry for nothing."""
    with pytest.raises(ValueError, match="at least one work item"):
        Session(items=())


def test_a_document_without_a_source_label_is_rejected():
    with pytest.raises(ValueError, match="empty source label"):
        DocRef(index=0, source="", n_tokens=100)


def test_a_document_with_negative_tokens_is_rejected():
    with pytest.raises(ValueError, match="n_tokens must be >= 0"):
        DocRef(index=0, source="c4", n_tokens=-1)


def test_a_carry_is_keyed_by_base_model_layer_index():
    carry = Carry(deltas={1: torch.zeros(2, 2), 3: torch.zeros(2, 2)})

    assert carry.layer_indices == (1, 3)


def test_a_carry_rejects_non_integer_keys():
    """The old code had an enumeration-keyed twin; there is only one now."""
    with pytest.raises(TypeError, match="must be base-model layer indices"):
        Carry(deltas={"layer_1": torch.zeros(2, 2)})


def test_an_empty_carry_is_empty():
    assert Carry.empty().is_empty and len(Carry.empty()) == 0


def test_a_carry_returns_none_for_a_layer_it_does_not_have():
    assert Carry(deltas={1: torch.zeros(2, 2)}).get(2) is None


def test_a_carry_copies_the_mapping_it_was_given():
    """A carry handed out of a snapshot must not alias the caller's dict."""
    source = {1: torch.zeros(2, 2)}
    carry = Carry(deltas=source)

    source[2] = torch.zeros(2, 2)

    assert carry.layer_indices == (1,)


def test_a_non_positive_perplexity_is_rejected():
    """Perplexity is exp of a loss; zero or negative means a bug upstream."""
    with pytest.raises(ValueError, match="ppl must be positive"):
        PplRow(n_tokens=10, ppl=0.0)


def test_an_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        EvalRow(
            doc_idx=0, source="c4", regime="carry_ish", n_tokens=10, ppl=5.0
        )


def test_every_named_regime_is_accepted():
    from ttt.core.types import REGIMES

    rows = [build.eval_row(regime=regime) for regime in REGIMES]

    assert len(rows) == 6


def _slice_row(**overrides) -> SliceRow:
    defaults = dict(
        doc_idx=0, source="c4", regime="fresh", slice_index=0, n_tokens=10, ppl=5.0
    )
    return SliceRow(**{**defaults, **overrides})


def test_a_slice_row_with_an_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        _slice_row(regime="carry_ish")


def test_a_slice_row_with_a_negative_index_is_rejected():
    with pytest.raises(ValueError, match="slice_index must be >= 0"):
        _slice_row(slice_index=-1)


def test_a_non_positive_slice_row_perplexity_is_rejected():
    with pytest.raises(ValueError, match="ppl must be positive"):
        _slice_row(ppl=0.0)
