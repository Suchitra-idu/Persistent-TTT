"""OLD-vs-NEW string and structure parity (Phase 6.2).

Nothing numeric here — these are the names and regexes a checkpoint and a PEFT
wrap agree on. A silent change to any of them lands LoRA on a fast weight or
drops a tensor from a checkpoint, with no error to say so.
"""

from __future__ import annotations

import math

import pytest
import train_modal
import ttt_wiring

from tests.parity import _builders as build
from ttt.core import metrics, naming
from ttt.core.types import CARRY, COLD_CARRY, COLD_CARRY_OFF, FRESH, EvalRow, PplRow

pytestmark = pytest.mark.parity

LAYER_COUNTS = [4, 28, 36]
STRIDES = [2, 4]

NAMES = [
    "model.layers.1.mlp.w_target",
    "model.layers.1.mlp.target_conv.weight",
    "model.layers.1.mlp.output_gate.weight",
    "model.layers.1.mlp.output_gate.bias",
    "model.layers.1.mlp.v_source_norm.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "model.layers.0.mlp.gate_proj.lora_B.default.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
    "base_model.model.model.layers.3.mlp.w_target",
    "base_model.model.model.layers.3.mlp.down_proj.weight",
]


def old_config(num_layers: int, stride: int):
    cfg, _ = build.configs()
    cfg.layer_indices = tuple(range(1, num_layers, stride))
    return cfg


class TestLoraTargets:
    @pytest.mark.parametrize("num_layers", LAYER_COUNTS)
    @pytest.mark.parametrize("stride", STRIDES)
    def test_the_target_regex_is_byte_identical(self, num_layers, stride):
        cfg = old_config(num_layers, stride)

        assert naming.lora_target_regex(
            num_layers, cfg.layer_indices
        ) == ttt_wiring.build_lora_target_regex(num_layers, cfg)

    @pytest.mark.parametrize("num_layers", LAYER_COUNTS)
    def test_it_still_keeps_lora_off_every_ttt_down_projection(self, num_layers):
        import re

        cfg = old_config(num_layers, 2)
        pattern = re.compile(naming.lora_target_regex(num_layers, cfg.layer_indices))

        assert not any(
            pattern.fullmatch(f"model.layers.{index}.mlp.down_proj")
            for index in cfg.layer_indices
        )


class TestParameterNames:
    @pytest.mark.parametrize("num_layers", LAYER_COUNTS)
    def test_the_down_projection_suffixes_agree(self, num_layers):
        cfg = old_config(num_layers, 2)

        assert naming.ttt_down_suffixes(cfg.layer_indices) == ttt_wiring.ttt_down_suffixes(
            cfg
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_peft_prefix_strips_identically(self, name):
        assert naming.strip_peft_prefix(name) == ttt_wiring.strip_peft_prefix(name)

    @pytest.mark.parametrize("name", NAMES)
    def test_every_name_classifies_into_the_same_group(self, name):
        cfg = old_config(4, 2)
        suffixes = ttt_wiring.ttt_down_suffixes(cfg)

        assert naming.classify_param(name, frozenset(suffixes)) == (
            ttt_wiring._classify_ttt_param(name, suffixes)
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_checkpoint_membership_agrees(self, name):
        cfg = old_config(4, 2)
        suffixes = ttt_wiring.ttt_down_suffixes(cfg)
        key = ttt_wiring.strip_peft_prefix(name)
        old_keeps = any(
            marker in key for marker in ttt_wiring.TTT_PARAM_MARKERS
        ) or any(key.endswith(suffix) for suffix in suffixes)

        assert naming.is_checkpoint_key(name, frozenset(suffixes)) == old_keeps

    def test_the_marker_lists_are_the_same(self):
        assert set(naming.TTT_PARAM_MARKERS) == set(ttt_wiring.TTT_PARAM_MARKERS)

    def test_the_frozen_marker_lists_are_the_same(self):
        assert set(naming.TTT_FROZEN_MARKERS) == set(ttt_wiring.TTT_FROZEN_MARKERS)

    def test_the_name_list_covers_every_group(self):
        """Otherwise the classification parity above would be checking one case."""
        cfg = old_config(4, 2)
        suffixes = frozenset(ttt_wiring.ttt_down_suffixes(cfg))
        found = {naming.classify_param(name, suffixes) for name in NAMES}

        assert found == {naming.GROUP_LORA, naming.GROUP_NEW, naming.GROUP_WDOWN, None}


SLICES = ((100, 2.0), (200, 3.0), (50, 5.0))


class TestMetricArithmetic:
    def test_the_token_weighted_perplexity_agrees(self):
        rows = [(n, ppl, 0.0) for n, ppl in SLICES]

        assert metrics.token_weighted_ppl(
            [PplRow(n_tokens=n, ppl=ppl) for n, ppl in SLICES]
        ) == pytest.approx(train_modal._token_weighted_ppl(rows), rel=1e-12)

    def test_zero_tokens_are_nan_on_both_sides(self):
        assert math.isnan(metrics.token_weighted_ppl([])) and math.isnan(
            train_modal._token_weighted_ppl([])
        )

    @pytest.mark.parametrize("source", ["alpha", "beta"])
    @pytest.mark.parametrize(
        ("regime", "key"),
        [(COLD_CARRY, "carry_ppl"), (COLD_CARRY_OFF, "carry_off_ppl"), (FRESH, "fresh_ppl")],
    )
    def test_the_per_source_perplexity_agrees(self, source, regime, key):
        old, summaries = _both_sides()

        assert summaries[source].ppl_by_regime[regime] == pytest.approx(
            old[f"eval/{source}/{key}"], rel=1e-12
        )

    @pytest.mark.parametrize("source", ["alpha", "beta"])
    @pytest.mark.parametrize("gap", ["within", "between"])
    def test_the_per_source_gap_agrees(self, source, gap):
        old, summaries = _both_sides()

        assert getattr(summaries[source].gaps, gap) == pytest.approx(
            old[f"eval/{source}/gap_{gap}"], rel=1e-12
        )

    @pytest.mark.parametrize("source", ["alpha", "beta"])
    def test_the_per_source_document_count_agrees(self, source):
        old, summaries = _both_sides()

        assert summaries[source].n_docs == old[f"eval/{source}/n_papers"]

    def test_the_document_aggregate_agrees(self):
        """The old loop inlined a geometric mean over documents; core.metrics
        names it, and the numbers have to be the same."""
        ppls = [2.0, 3.5, 4.25]
        inlined = math.exp(sum(math.log(value) for value in ppls) / len(ppls))

        assert metrics.geometric_mean_ppl(ppls) == pytest.approx(inlined, rel=1e-12)

    def test_the_clip_ratio_agrees(self):
        total_norm, maximum = 20.0, 10.0
        inlined = min(1.0, maximum / (total_norm + 1e-12))

        assert metrics.clip_ratio(total_norm, maximum) == pytest.approx(
            inlined, rel=1e-12
        )


def _both_sides():
    """(old flat keys, new summaries by source) over the same fixture."""
    new_rows, old_papers, doc_sources = _eval_fixture()
    return (
        train_modal._per_source_eval_metrics(old_papers, doc_sources),
        {s.source: s for s in metrics.summarise_by_source(new_rows)},
    )


def _eval_fixture():
    """Two sources, two documents each, three slices apiece, all distinct.

    The third return value is one label *per document* — the shape the old
    function zips against, not the distinct set.
    """
    sources = ["alpha", "alpha", "beta", "beta"]
    new_rows: list[EvalRow] = []
    old_papers = []
    for doc_idx, source in enumerate(sources):
        offsets = {COLD_CARRY: 0.0, COLD_CARRY_OFF: 0.4, FRESH: 0.9}
        rows = {
            regime: [
                (n, ppl + offset + doc_idx * 0.11, 0.0) for n, ppl in SLICES
            ]
            for regime, offset in offsets.items()
        }
        old_papers.append(
            {
                "carry_ppl": train_modal._token_weighted_ppl(rows[COLD_CARRY]),
                "carry_off_ppl": train_modal._token_weighted_ppl(rows[COLD_CARRY_OFF]),
                "fresh_ppl": train_modal._token_weighted_ppl(rows[FRESH]),
                "carry_rows": rows[COLD_CARRY],
                "carry_off_rows": rows[COLD_CARRY_OFF],
                "fresh_rows": rows[FRESH],
            }
        )
        new_rows.extend(
            EvalRow(
                doc_idx=doc_idx,
                source=source,
                regime=regime,
                n_tokens=sum(n for n, _, _ in rows[regime]),
                ppl=metrics.token_weighted_ppl(
                    [PplRow(n_tokens=n, ppl=ppl) for n, ppl, _ in rows[regime]]
                ),
            )
            for regime in offsets
        )
    return new_rows, old_papers, sources


def test_the_fixture_gives_each_source_a_different_number():
    """Otherwise the per-source parity assertions would all compare one value."""
    new_rows, _, _ = _eval_fixture()
    by_source = {
        row.source: row.ppl for row in new_rows if row.regime == COLD_CARRY
    }

    assert len(set(by_source.values())) == len(by_source)


def test_the_seeded_regime_has_no_counterpart_in_the_old_flat_keys():
    """`carry` is the seeded regime; the old code called the cold one `carry`
    and had no name for this. The rename is recorded in docs/app-map.md."""
    assert CARRY not in (COLD_CARRY, COLD_CARRY_OFF, FRESH)


@pytest.mark.parametrize("field", build.SHARED_FIELDS)
def test_the_two_config_types_still_agree_on_every_shared_field(field):
    old_cfg, new_cfg = build.configs()

    assert getattr(old_cfg, field) == getattr(new_cfg, field)


@pytest.mark.parametrize("field", ["v_source", "gate_reg_weight"])
def test_the_fields_d11_cut_are_gone_from_the_new_config(field):
    _, new_cfg = build.configs()

    assert not hasattr(new_cfg, field)
