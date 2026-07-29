"""Tests for pure helpers inside train_modal.py.

Covers:
  - _resolve_resume path resolution + missing-file guard
  - _token_weighted_ppl math
  - _stratified_sample_indices round-robin behavior
  - _n_per_source_indices per-source sampler
  - _per_source_eval_metrics token-weighted aggregation
  - _apply_cli_overrides mode dispatch and knob merging
"""

import math
import random

import pytest


# ---------- _apply_cli_overrides ----------

def _defaults():
    """Everything-unset kwargs: 0 for int knobs, "" for str knobs,
    _UNSET (-1) for session."""
    from train_modal import _UNSET
    return dict(num_epochs=0, grad_accum=0, session=_UNSET, mode="",
                min_doc_tokens=0, hybrid_carry_min=0,
                hybrid_slice_min=0, hybrid_slices_min=0,
                hybrid_slices_max=0, eval_n_papers=0,
                eval_n_papers_per_source=0, eval_min_tokens=0,
                eval_every=0, source_preset="")


def test_apply_overrides_all_unset_returns_train_cfg_identity():
    from train_modal import TRAIN_CFG, _apply_cli_overrides
    assert _apply_cli_overrides(**_defaults()) is TRAIN_CFG


def test_apply_overrides_mode_multi_forces_both_flags_false():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["mode"] = "multi"
    cfg = _apply_cli_overrides(**kw)
    assert cfg.single_paper_sessions is False
    assert cfg.hybrid_sessions is False


def test_apply_overrides_mode_single_enables_single_only():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["mode"] = "single"
    cfg = _apply_cli_overrides(**kw)
    assert cfg.single_paper_sessions is True
    assert cfg.hybrid_sessions is False


def test_apply_overrides_mode_hybrid_enables_hybrid_only():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["mode"] = "hybrid"
    cfg = _apply_cli_overrides(**kw)
    assert cfg.single_paper_sessions is False
    assert cfg.hybrid_sessions is True


def test_apply_overrides_unknown_mode_raises():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["mode"] = "bogus"
    with pytest.raises(ValueError, match="unknown --mode"):
        _apply_cli_overrides(**kw)


def test_apply_overrides_hybrid_knobs_flow_through():
    from train_modal import _apply_cli_overrides
    kw = _defaults()
    kw.update(mode="hybrid", min_doc_tokens=256,
              hybrid_carry_min=5000, hybrid_slice_min=1000,
              hybrid_slices_min=3, hybrid_slices_max=8)
    cfg = _apply_cli_overrides(**kw)
    assert cfg.min_doc_tokens == 256
    assert cfg.hybrid_carry_min_tokens == 5000
    assert cfg.hybrid_slice_min_tokens == 1000
    assert cfg.hybrid_slices_min == 3
    assert cfg.hybrid_slices_max == 8


def test_apply_overrides_eval_knobs_flow_through():
    from train_modal import _apply_cli_overrides
    kw = _defaults()
    kw.update(eval_n_papers=12, eval_n_papers_per_source=2,
              eval_min_tokens=4096)
    cfg = _apply_cli_overrides(**kw)
    assert cfg.eval_n_papers == 12
    assert cfg.eval_n_papers_per_source == 2
    assert cfg.eval_min_tokens == 4096


def test_apply_overrides_session_flag_toggles_training():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["session"] = 1
    assert _apply_cli_overrides(**kw).session_training is True
    kw["session"] = 0
    assert _apply_cli_overrides(**kw).session_training is False


def test_apply_overrides_source_preset_flows_through():
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["source_preset"] = "slim-paper"
    assert _apply_cli_overrides(**kw).source_preset == "slim-paper"


def test_apply_overrides_unknown_source_preset_fails_fast():
    """Bad preset name should fail in the local entrypoint, before
    the container spins up."""
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["source_preset"] = "does-not-exist"
    with pytest.raises(KeyError, match="Unknown source preset"):
        _apply_cli_overrides(**kw)


def test_apply_overrides_source_preset_none_bypasses_validation():
    """'none' is an explicit opt-out that skips preset-name validation
    (so users can disable the spec's default without knowing what's
    registered)."""
    from train_modal import _apply_cli_overrides
    kw = _defaults(); kw["source_preset"] = "none"
    assert _apply_cli_overrides(**kw).source_preset == "none"


def test_resolve_resume_none_returns_none_pair(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    assert _resolve_resume("", "any-run") == (None, None)


def test_resolve_resume_same_run_form(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    step_dir = tmp_path / "myrun" / "step_100"
    (step_dir / "adapter").mkdir(parents=True)
    (step_dir / "ttt_params.pt").write_bytes(b"")

    adapter, ttt = _resolve_resume("step_100", "myrun")
    assert adapter == str(step_dir / "adapter")
    assert ttt == str(step_dir / "ttt_params.pt")


def test_resolve_resume_cross_run_form(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    step_dir = tmp_path / "otherrun" / "step_50"
    (step_dir / "adapter").mkdir(parents=True)
    (step_dir / "ttt_params.pt").write_bytes(b"")

    adapter, ttt = _resolve_resume("otherrun/step_50", "currentrun")
    assert adapter.startswith(str(tmp_path / "otherrun" / "step_50"))
    assert not adapter.startswith(str(tmp_path / "currentrun"))


def test_resolve_resume_missing_files_raises(monkeypatch, tmp_path):
    from train_modal import _resolve_resume

    monkeypatch.setattr("train_modal.CKPT_MOUNT", str(tmp_path))
    (tmp_path / "myrun" / "step_100" / "adapter").mkdir(parents=True)
    # ttt_params.pt intentionally missing.
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        _resolve_resume("step_100", "myrun")


def test_token_weighted_ppl_matches_log_mean():
    from train_modal import _token_weighted_ppl

    rows = [(100, 10.0, 0.0), (300, 20.0, 0.0)]
    expected = math.exp((100 * math.log(10) + 300 * math.log(20)) / 400)
    assert abs(_token_weighted_ppl(rows) - expected) < 1e-12


def test_token_weighted_ppl_empty_returns_nan():
    from train_modal import _token_weighted_ppl

    assert math.isnan(_token_weighted_ppl([]))


# ---------- _stratified_sample_indices ----------

def test_stratified_sample_covers_every_source_when_possible():
    from train_modal import _stratified_sample_indices

    labels = (["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    picked = _stratified_sample_indices(labels, n_target=6, rng=random.Random(0))
    picked_labels = {labels[i] for i in picked}
    # Round-robin picks A, B, C, A, B, C so every source represented.
    assert picked_labels == {"A", "B", "C"}
    assert len(picked) == 6


def test_stratified_sample_undersized_bucket_still_yields_max_target():
    from train_modal import _stratified_sample_indices

    labels = ["A"] * 20 + ["B"] * 1
    picked = _stratified_sample_indices(labels, n_target=10, rng=random.Random(0))
    assert len(picked) == 10
    labels_seen = [labels[i] for i in picked]
    # B is picked at least once (round-robin) but at most once (only one available).
    assert labels_seen.count("B") == 1


def test_stratified_sample_no_labels_falls_back_to_random():
    from train_modal import _stratified_sample_indices

    labels = [""] * 20
    picked = _stratified_sample_indices(labels, n_target=5, rng=random.Random(0))
    assert len(picked) == 5
    assert len(set(picked)) == 5     # no duplicates


def test_stratified_sample_is_deterministic():
    from train_modal import _stratified_sample_indices

    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    a = _stratified_sample_indices(labels, 6, random.Random(42))
    b = _stratified_sample_indices(labels, 6, random.Random(42))
    assert a == b


# ---------- _per_source_eval_metrics ----------

def _row(n_tok, ppl):
    return (n_tok, ppl, 0.0)


def _paper(carry_ppl, carry_off_ppl, fresh_ppl, n_tok):
    return {
        "carry_ppl": carry_ppl,
        "carry_off_ppl": carry_off_ppl,
        "fresh_ppl": fresh_ppl,
        "state_ratio_final": 0.0,
        "carry_rows": [_row(n_tok, carry_ppl)],
        "carry_off_rows": [_row(n_tok, carry_off_ppl)],
        "fresh_rows": [_row(n_tok, fresh_ppl)],
    }


def test_per_source_eval_groups_by_source():
    from train_modal import _per_source_eval_metrics

    per_paper = [
        _paper(carry_ppl=10.0, carry_off_ppl=11.0, fresh_ppl=12.0, n_tok=100),
        _paper(carry_ppl=20.0, carry_off_ppl=22.0, fresh_ppl=25.0, n_tok=200),
        _paper(carry_ppl=8.0,  carry_off_ppl=8.2,  fresh_ppl=8.5,  n_tok=100),
    ]
    sources = ["C4", "Github", "C4"]
    out = _per_source_eval_metrics(per_paper, sources)

    assert "eval/C4/carry_ppl" in out
    assert "eval/C4/carry_off_ppl" in out
    assert "eval/Github/fresh_ppl" in out
    assert out["eval/C4/n_papers"] == 2
    assert out["eval/Github/n_papers"] == 1
    # single-paper Github: within = fresh - carry_off; between = carry_off - carry
    assert out["eval/Github/gap_within"] == pytest.approx(25.0 - 22.0)
    assert out["eval/Github/gap_between"] == pytest.approx(22.0 - 20.0)
    assert out["eval/Github/gap_total"] == pytest.approx(25.0 - 20.0)


def test_per_source_eval_token_weighted():
    """C4: two papers, equal weight 100. carry ppls 10 and 8 -> geo mean sqrt(80).
    All three modes are token-weighted geo means over the same token counts."""
    from train_modal import _per_source_eval_metrics

    per_paper = [
        _paper(carry_ppl=10.0, carry_off_ppl=11.0, fresh_ppl=12.0, n_tok=100),
        _paper(carry_ppl=8.0,  carry_off_ppl=8.5,  fresh_ppl=9.0,  n_tok=100),
    ]
    out = _per_source_eval_metrics(per_paper, ["C4", "C4"])
    expected_carry = math.exp((math.log(10.0) + math.log(8.0)) / 2)
    expected_carry_off = math.exp((math.log(11.0) + math.log(8.5)) / 2)
    expected_fresh = math.exp((math.log(12.0) + math.log(9.0)) / 2)
    assert out["eval/C4/carry_ppl"] == pytest.approx(expected_carry)
    assert out["eval/C4/carry_off_ppl"] == pytest.approx(expected_carry_off)
    assert out["eval/C4/fresh_ppl"] == pytest.approx(expected_fresh)


def test_per_source_eval_skips_empty_sources():
    from train_modal import _per_source_eval_metrics

    per_paper = [_paper(10.0, 11.0, 12.0, 100),
                 _paper(20.0, 22.0, 25.0, 200)]
    out = _per_source_eval_metrics(per_paper, ["", ""])
    assert out == {}


def test_per_source_eval_gap_decomposition_sums():
    """gap_within + gap_between == gap_total for every source (by construction)."""
    from train_modal import _per_source_eval_metrics

    per_paper = [_paper(carry_ppl=5.0, carry_off_ppl=6.0, fresh_ppl=8.0,
                        n_tok=250)]
    out = _per_source_eval_metrics(per_paper, ["Wiki"])
    assert out["eval/Wiki/gap_within"] + out["eval/Wiki/gap_between"] == \
        pytest.approx(out["eval/Wiki/gap_total"])


# ---------- _n_per_source_indices ----------

def test_n_per_source_gives_exactly_n_per_source():
    from train_modal import _n_per_source_indices

    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    picked = _n_per_source_indices(labels, 2, random.Random(0))
    counts = {"A": 0, "B": 0, "C": 0}
    for i in picked:
        counts[labels[i]] += 1
    assert counts == {"A": 2, "B": 2, "C": 2}
    assert len(picked) == 6


def test_n_per_source_undersized_bucket_uses_what_it_has():
    from train_modal import _n_per_source_indices

    labels = ["A"] * 5 + ["B"] * 1
    picked = _n_per_source_indices(labels, 3, random.Random(0))
    counts = {"A": 0, "B": 0}
    for i in picked:
        counts[labels[i]] += 1
    assert counts["A"] == 3
    assert counts["B"] == 1     # only 1 B available


def test_n_per_source_no_duplicates():
    from train_modal import _n_per_source_indices

    labels = ["A"] * 10 + ["B"] * 10
    picked = _n_per_source_indices(labels, 3, random.Random(0))
    assert len(set(picked)) == len(picked)


def test_n_per_source_deterministic_per_rng():
    from train_modal import _n_per_source_indices

    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    a = _n_per_source_indices(labels, 2, random.Random(42))
    b = _n_per_source_indices(labels, 2, random.Random(42))
    assert a == b


# ---------- _balance_by_source_preset ----------

class _FakeDS:
    """Minimal duck-typed stand-in for a HF datasets.Dataset.

    Supports the two operations the balancer uses: `len(ds)`,
    `ds["source"]` (returns the source column as a list), and
    `ds.select(indices)` (returns a new _FakeDS restricted to those
    indices)."""

    def __init__(self, sources):
        self._sources = list(sources)

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, key):
        if key == "source":
            return list(self._sources)
        raise KeyError(key)

    @property
    def column_names(self):
        return ["source"]

    def select(self, indices):
        return _FakeDS([self._sources[i] for i in indices])


def _slim_ds(counts: dict):
    """Build a fake ds with the given per-source counts."""
    sources = []
    for src, n in counts.items():
        sources.extend([src] * n)
    return _FakeDS(sources)


def test_balance_by_preset_hits_ratios_when_pool_is_large():
    from collections import Counter
    from train_modal import _balance_by_source_preset

    # Big pool: 6 sources, 10000 rows each. Any preset can be honored.
    ds = _slim_ds({
        "RedPajamaC4": 10000, "RedPajamaGithub": 10000,
        "RedPajamaBook": 10000, "RedPajamaArXiv": 10000,
        "RedPajamaWikipedia": 10000, "RedPajamaStackExchange": 10000,
    })
    out = _balance_by_source_preset(ds, "slim-paper", target_total=10000)
    counts = Counter(out["source"])

    # slim-paper: C4 62/99, GH 9/99, Book 8/99, ArXiv 7/99, Wiki 7/99, SE 6/99
    total_w = 62 + 9 + 8 + 7 + 7 + 6
    assert counts["RedPajamaC4"] == int(10000 * 62 / total_w)
    assert counts["RedPajamaGithub"] == int(10000 * 9 / total_w)
    assert counts["RedPajamaBook"] == int(10000 * 8 / total_w)


def test_balance_by_preset_undersized_bucket_takes_all_available():
    from collections import Counter
    from train_modal import _balance_by_source_preset

    # Books has only 3 rows -- should take all 3 rather than pad with
    # something else.
    ds = _slim_ds({
        "RedPajamaC4": 10000, "RedPajamaGithub": 10000,
        "RedPajamaBook": 3,   "RedPajamaArXiv": 10000,
        "RedPajamaWikipedia": 10000, "RedPajamaStackExchange": 10000,
    })
    out = _balance_by_source_preset(ds, "slim-paper", target_total=10000)
    counts = Counter(out["source"])
    assert counts["RedPajamaBook"] == 3


def test_balance_by_preset_raises_when_no_source_matches():
    from train_modal import _balance_by_source_preset

    # Dataset labels don't match preset -- fail loudly.
    ds = _slim_ds({"totally_different_source": 100})
    with pytest.raises(ValueError, match="none match"):
        _balance_by_source_preset(ds, "slim-paper", target_total=100)


def test_balance_by_preset_preserves_shuffled_order():
    """The returned ds should keep the pre-shuffled order of picked
    indices, not group by source. Verified by checking that adjacent
    positions in the output span multiple sources."""
    from train_modal import _balance_by_source_preset

    # Interleave sources so any window has multiple.
    sources = []
    for _ in range(200):
        for src in ("RedPajamaC4", "RedPajamaGithub", "RedPajamaBook",
                    "RedPajamaArXiv", "RedPajamaWikipedia",
                    "RedPajamaStackExchange"):
            sources.append(src)
    ds = _FakeDS(sources)
    out = _balance_by_source_preset(ds, "slim-paper", target_total=600)

    # The output should NOT be all one source in a block; a window of
    # 20 near the start should have at least 3 distinct sources.
    window = out["source"][:20]
    assert len(set(window)) >= 3


def test_balance_by_preset_target_scaling():
    """Doubling target_total should roughly double per-source picks
    (up to available-pool bottlenecks)."""
    from collections import Counter
    from train_modal import _balance_by_source_preset

    ds = _slim_ds({
        "RedPajamaC4": 100000, "RedPajamaGithub": 100000,
        "RedPajamaBook": 100000, "RedPajamaArXiv": 100000,
        "RedPajamaWikipedia": 100000, "RedPajamaStackExchange": 100000,
    })
    a = Counter(_balance_by_source_preset(ds, "slim-research",
                                          target_total=1000)["source"])
    b = Counter(_balance_by_source_preset(ds, "slim-research",
                                          target_total=2000)["source"])
    for src in a:
        # Allow a small off-by-one from int truncation of the ratio.
        assert abs(b[src] - 2 * a[src]) <= 2
