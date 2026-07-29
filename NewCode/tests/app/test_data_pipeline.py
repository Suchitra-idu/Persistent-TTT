from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import data_pipeline
from ttt.app.stages import TOKENS_COLUMN
from ttt.core.config.dataset import DatasetSpec
from ttt.ports.table import SOURCE_COLUMN

MIXED = ("alpha", "beta") * 6


def loaded(sources=MIXED, **kwargs):
    defaults = dict(
        source=_builders.data_source(sources),
        spec=_builders.SPEC,
        cfg=_builders.config(),
        rng=ScriptedRng(),
        tokenizer=FakeTokenizer(),
    )
    return data_pipeline.load(**{**defaults, **kwargs})


class TestLoad:
    def test_it_reserves_the_holdout(self):
        assert len(loaded().table) == len(MIXED) - _builders.SPEC.holdout_last_n

    def test_every_document_is_tokenized(self):
        assert all(loaded().table.column(TOKENS_COLUMN))

    def test_every_document_carries_a_source(self):
        assert all(loaded().table.column(SOURCE_COLUMN))

    def test_it_honours_the_document_limit(self):
        assert len(loaded(limit_docs=3).table) == 3

    def test_every_stage_records_a_pass(self):
        assert len(loaded().log) == len(data_pipeline.PIPELINE)

    def test_the_log_names_the_stages_in_order(self):
        assert [entry.stage for entry in loaded().log] == [
            stage.__name__ for stage in data_pipeline.PIPELINE
        ]

    def test_a_preset_reaches_the_balancer_as_data(self):
        data = loaded(weights={"alpha": 1}, limit_docs=2)

        assert set(data.table.column(SOURCE_COLUMN)) == {"alpha"}

    def test_the_same_seed_visits_documents_in_the_same_order(self):
        first = loaded(rng=ScriptedRng(permutations=([3, 0, 1, 2, 4, 5, 6, 7, 8, 9],)))
        second = loaded(rng=ScriptedRng(permutations=([3, 0, 1, 2, 4, 5, 6, 7, 8, 9],)))

        assert first.table.column("text") == second.table.column("text")


class TestDocuments:
    def test_it_yields_one_document_per_row(self):
        data = loaded(limit_docs=3)

        assert len(data_pipeline.documents(data)) == 3

    def test_a_document_carries_its_tokens(self):
        docs = data_pipeline.documents(loaded(limit_docs=1))

        assert docs[0].n_tokens == len(docs[0].token_ids) > 0

    def test_document_indices_are_positions_in_the_pool(self):
        docs = data_pipeline.documents(loaded(limit_docs=3))

        assert [doc.index for doc in docs] == [0, 1, 2]


class TestResolveWeights:
    def test_an_explicit_none_disables_balancing(self):
        cfg = _builders.config(source_preset="none")

        assert data_pipeline.resolve_weights(cfg, _builders.SPEC) is None

    def test_a_spec_without_a_default_balances_nothing(self):
        assert data_pipeline.resolve_weights(_builders.config(), _builders.SPEC) is None

    def test_an_explicit_preset_wins(self):
        cfg = _builders.config(source_preset="slim-research")

        assert "RedPajamaC4" in data_pipeline.resolve_weights(cfg, _builders.SPEC)

    def test_the_spec_default_applies_when_the_config_is_silent(self):
        from dataclasses import replace

        spec = replace(_builders.SPEC, default_source_preset="slim-paper")

        assert data_pipeline.resolve_weights(_builders.config(), spec) == {
            "RedPajamaC4": 62,
            "RedPajamaGithub": 9,
            "RedPajamaBook": 8,
            "RedPajamaArXiv": 7,
            "RedPajamaWikipedia": 7,
            "RedPajamaStackExchange": 6,
        }


class TestHoldout:
    def test_it_takes_the_reserved_tail(self):
        held = data_pipeline.holdout(
            source=_builders.data_source(MIXED),
            spec=_builders.SPEC,
            cfg=_builders.config(),
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )

        assert {doc.index for doc in held.docs} <= {0, 1}

    def test_it_samples_the_configured_count_per_source(self):
        held = data_pipeline.holdout(
            source=_builders.data_source(("alpha", "beta") * 6),
            spec=DatasetSpec(
                name=_builders.SPEC.name,
                source="_fixture",
                source_meta_column="meta",
                source_meta_key="set_name",
                include_sources=_builders.SOURCES,
                holdout_last_n=6,
            ),
            cfg=_builders.config(eval_n_docs_per_source=2),
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )

        assert len(held.docs) == 4

    def test_a_scarce_source_reports_a_shortfall(self):
        held = data_pipeline.holdout(
            source=_builders.data_source(("alpha", "alpha", "beta")),
            spec=DatasetSpec(
                name=_builders.SPEC.name,
                source="_fixture",
                source_meta_column="meta",
                source_meta_key="set_name",
                include_sources=_builders.SOURCES,
                holdout_last_n=3,
            ),
            cfg=_builders.config(eval_n_docs_per_source=2),
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )

        assert [(s.source, s.got, s.wanted) for s in held.shortfalls] == [("beta", 1, 2)]

    def test_it_falls_back_rather_than_evaluating_on_nothing(self):
        held = data_pipeline.holdout(
            source=_builders.data_source(MIXED),
            spec=_builders.SPEC,
            cfg=_builders.config(eval_min_tokens=10_000),
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )

        assert held.docs

    def test_every_held_out_document_is_tokenized(self):
        held = data_pipeline.holdout(
            source=_builders.data_source(MIXED),
            spec=_builders.SPEC,
            cfg=_builders.config(),
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )

        assert all(doc.n_tokens > 0 for doc in held.docs)


def test_the_holdout_and_the_train_pool_do_not_overlap():
    source = _builders.data_source(MIXED)
    train = loaded(source=source)
    held = data_pipeline.holdout(
        source=source,
        spec=_builders.SPEC,
        cfg=_builders.config(),
        rng=ScriptedRng(),
        tokenizer=FakeTokenizer(),
    )

    assert not set(train.table.column("text")) & {
        FakeTokenizer().decode(doc.token_ids) for doc in held.docs
    }


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_the_pipeline_never_returns_more_than_the_limit(limit):
    assert len(loaded(limit_docs=limit).table) <= limit
