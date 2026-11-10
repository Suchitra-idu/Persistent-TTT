from __future__ import annotations


from tests.app import _builders
from ttt.adapters.list_table import ListTable
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import stages
from ttt.app.stages import TOKENS_COLUMN, Data
from ttt.core.config.train import TrainConfig
from ttt.ports.table import SOURCE_COLUMN

MIXED = ("alpha", "beta", "alpha", "beta", "alpha", "beta")


def with_table(data: Data, rows) -> Data:
    return Data(
        table=ListTable(list(rows)),
        spec=data.spec,
        cfg=data.cfg,
        rng=data.rng,
        tokenizer=data.tokenizer,
        limit_docs=data.limit_docs,
        weights=data.weights,
    )


def tokenized(sources, lengths, *, cfg: TrainConfig | None = None) -> Data:
    data = _builders.pipeline_data(sources, cfg=cfg)
    return with_table(
        data,
        [
            {SOURCE_COLUMN: source, TOKENS_COLUMN: list(range(length))}
            for source, length in zip(sources, lengths, strict=True)
        ],
    )


class TestSplitHoldout:
    def test_it_reserves_the_newest_rows(self):
        data = _builders.pipeline_data(MIXED)

        assert len(stages.split_holdout(data).table) == len(MIXED) - 2

    def test_it_keeps_the_oldest_rows(self):
        data = _builders.pipeline_data(MIXED)

        kept = stages.split_holdout(data).table.column(SOURCE_COLUMN)

        assert kept == list(MIXED[:-2])

    def test_a_pool_smaller_than_the_holdout_leaves_nothing_to_train_on(self):
        data = _builders.pipeline_data(MIXED[:1])

        assert len(stages.split_holdout(data).table) == 0


class TestFilterSources:
    def test_it_drops_sources_the_spec_excludes(self):
        data = _builders.pipeline_data(("alpha", "gamma", "beta"))

        kept = stages.filter_sources(data).table.column(SOURCE_COLUMN)

        assert kept == ["alpha", "beta"]


class TestShuffle:
    def test_it_applies_the_permutation_the_rng_drew(self):
        data = _builders.pipeline_data(MIXED[:3])
        scripted = Data(
            table=data.table,
            spec=data.spec,
            cfg=data.cfg,
            rng=ScriptedRng(permutations=([2, 0, 1],)),
            tokenizer=data.tokenizer,
        )

        shuffled = stages.shuffle(scripted).table.column("text")

        assert shuffled == [data.table.column("text")[i] for i in (2, 0, 1)]


class TestPrefilter:
    def test_it_drops_documents_too_short_to_reach_the_minimum(self):
        data = _builders.pipeline_data(MIXED, cfg=_builders.config(min_doc_tokens=20))

        assert len(stages.prefilter(data).table) == 0

    def test_it_keeps_documents_that_clear_the_character_threshold(self):
        data = _builders.pipeline_data(MIXED, cfg=_builders.config(min_doc_tokens=2))

        assert len(stages.prefilter(data).table) == len(MIXED)


class TestBalance:
    def test_no_preset_keeps_everything(self):
        data = _builders.pipeline_data(MIXED, weights=None)

        assert len(stages.balance(data).table) == len(MIXED)

    def test_no_preset_still_caps_when_a_limit_is_given(self):
        """The bug: limit_docs with no source preset was silently ignored
        until the final `cap` stage, so `tokenize` (in between) still ground
        through the whole pool regardless of limit_docs."""
        data = _builders.pipeline_data(MIXED, weights=None, limit_docs=1)

        assert len(stages.balance(data).table) == stages.PRESET_OVERSAMPLE

    def test_a_preset_pulls_the_pool_to_its_ratios(self):
        data = _builders.pipeline_data(
            ("alpha", "beta") * 6, weights={"alpha": 1, "beta": 3}, limit_docs=2
        )

        counts = stages.balance(data).table.column(SOURCE_COLUMN)

        assert (counts.count("alpha"), counts.count("beta")) == (1, 3)

    def test_a_source_the_preset_omits_is_dropped(self):
        data = _builders.pipeline_data(
            ("alpha", "beta") * 3, weights={"alpha": 1}, limit_docs=1
        )

        assert set(stages.balance(data).table.column(SOURCE_COLUMN)) == {"alpha"}

    def test_it_over_fetches_against_a_document_limit(self):
        data = _builders.pipeline_data(
            ("alpha", "beta") * 6, weights=_builders.WEIGHTS, limit_docs=2
        )

        assert len(stages.balance(data).table) == 2 * stages.PRESET_OVERSAMPLE


class TestTokenize:
    def test_it_adds_a_token_column(self):
        data = _builders.pipeline_data(("alpha",))

        assert TOKENS_COLUMN in stages.tokenize(data).table.column_names

    def test_it_truncates_to_the_sequence_length(self):
        data = _builders.pipeline_data(("alpha",), cfg=_builders.config(max_seq_len=5))

        assert len(stages.tokenize(data).table.column(TOKENS_COLUMN)[0]) == 5


    def test_batching_does_not_change_the_result(self, monkeypatch):
        unbatched = stages.tokenize(_builders.pipeline_data(("alpha",) * 5)).table.column(
            TOKENS_COLUMN
        )
        monkeypatch.setattr(stages, "TOKENIZE_BATCH_SIZE", 2)
        batched = stages.tokenize(_builders.pipeline_data(("alpha",) * 5)).table.column(
            TOKENS_COLUMN
        )

        assert batched == unbatched


class TestDropShort:
    def test_it_drops_by_exact_token_count(self):
        data = tokenized(
            ("alpha", "beta"), (10, 2), cfg=_builders.config(min_doc_tokens=5)
        )

        assert stages.drop_short(data).table.column(SOURCE_COLUMN) == ["alpha"]

    def test_a_document_at_the_minimum_survives(self):
        data = tokenized(("alpha",), (5,), cfg=_builders.config(min_doc_tokens=5))

        assert len(stages.drop_short(data).table) == 1


class TestRebalance:
    def test_it_restores_ratios_a_source_biased_drop_wrecked(self):
        skewed = _builders.pipeline_data(
            ("alpha",) * 2 + ("beta",) * 6, weights=_builders.WEIGHTS, limit_docs=4
        )

        counts = stages.rebalance(skewed).table.column(SOURCE_COLUMN)

        assert counts.count("alpha") == counts.count("beta") == 2

    def test_without_a_limit_it_is_bounded_by_the_pool_it_was_given(self):
        skewed = _builders.pipeline_data(
            ("alpha",) * 2 + ("beta",) * 6, weights=_builders.WEIGHTS
        )

        counts = stages.rebalance(skewed).table.column(SOURCE_COLUMN)

        assert (counts.count("alpha"), counts.count("beta")) == (2, 4)

    def test_it_targets_the_document_limit_exactly(self):
        data = _builders.pipeline_data(
            ("alpha", "beta") * 5, weights=_builders.WEIGHTS, limit_docs=4
        )

        assert len(stages.rebalance(data).table) == 4


class TestCap:
    def test_it_truncates_to_the_limit(self):
        data = _builders.pipeline_data(MIXED, limit_docs=2)

        assert len(stages.cap(data).table) == 2

    def test_no_limit_keeps_everything(self):
        data = _builders.pipeline_data(MIXED)

        assert len(stages.cap(data).table) == len(MIXED)


class TestStageLog:


    def test_a_stage_does_not_mutate_the_data_it_was_given(self):
        data = _builders.pipeline_data(MIXED)

        stages.split_holdout(data)

        assert data.log == ()


