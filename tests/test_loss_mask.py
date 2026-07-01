"""Content-token loss masking: determinism, no input mutation, disable
switch, and kept-token coverage vs keep_fraction."""

import pytest
import torch

from train_utils import (
    apply_loss_mask, apply_protect_passes, common_mask_from_counts,
    count_unigrams, load_reference_counts, protect_by_predicate,
    protect_numeric_tokens, protect_symbol_tokens, protect_token_ids,
)


def build_common_token_mask(input_ids_iter, vocab_size, keep_fraction):
    """Test helper composing the two public primitives."""
    return common_mask_from_counts(
        count_unigrams(input_ids_iter, vocab_size), keep_fraction,
    )


class FakeTokenizer:
    """Minimal stub satisfying .encode/.decode for protect_token_ids.

    decode_map defaults to {} -- missing ids decode to '' which conservatively
    fails the length gate, mirroring 'don't unmask what we can't characterize'.
    """

    def __init__(self, encode_map, decode_map=None):
        self._encode = encode_map
        self._decode = decode_map or {}

    def encode(self, text, add_special_tokens=False):
        return list(self._encode.get(text, []))

    def decode(self, ids):
        return "".join(self._decode.get(int(tid), "") for tid in ids)


def test_apply_loss_mask_disabled_returns_ids_unchanged():
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    out = apply_loss_mask(ids, None)
    assert out is ids


def test_apply_loss_mask_does_not_mutate_input():
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    original = ids.clone()
    mask = torch.zeros(256, dtype=torch.bool)
    mask[5] = mask[10] = True
    apply_loss_mask(ids, mask)
    assert torch.equal(ids, original)


def test_apply_loss_mask_masks_only_marked_token_ids():
    ids = torch.tensor([[5, 10, 200, 300]], dtype=torch.long)
    mask = torch.zeros(512, dtype=torch.bool)
    mask[10] = mask[300] = True
    out = apply_loss_mask(ids, mask)
    assert out.tolist() == [[5, -100, 200, -100]]


def test_apply_loss_mask_preserves_shape_and_dtype():
    ids = torch.randint(0, 100, (2, 3, 7), dtype=torch.long)
    mask = torch.zeros(100, dtype=torch.bool)
    out = apply_loss_mask(ids, mask)
    assert out.shape == ids.shape
    assert out.dtype == ids.dtype


def test_apply_loss_mask_empty_mask_returns_clone_equal_to_ids():
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    mask = torch.zeros(512, dtype=torch.bool)
    out = apply_loss_mask(ids, mask)
    assert torch.equal(out, ids)
    assert out.data_ptr() != ids.data_ptr()


def test_apply_loss_mask_all_masked_returns_all_ignore_index():
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    mask = torch.zeros(512, dtype=torch.bool)
    mask[[5, 10, 200]] = True
    out = apply_loss_mask(ids, mask)
    assert (out == -100).all()


def test_apply_loss_mask_first_tokens_only_masks_leading_positions():
    ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=2)
    assert out.tolist() == [[-100, -100, 30, 40, 50]]


def test_apply_loss_mask_first_tokens_clamps_to_seq_length():
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=99)
    assert (out == -100).all()


def test_apply_loss_mask_first_tokens_zero_is_no_op():
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=0)
    assert out is ids


def test_apply_loss_mask_first_tokens_composes_with_common_mask():
    ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[40] = True
    out = apply_loss_mask(ids, common_mask=mask, first_tokens=2)
    assert out.tolist() == [[-100, -100, 30, -100, 50]]


def test_apply_loss_mask_first_tokens_supports_batched_input():
    ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=1)
    assert out.tolist() == [[-100, 20, 30], [-100, 50, 60]]


def test_apply_loss_mask_first_tokens_does_not_mutate_input():
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    original = ids.clone()
    apply_loss_mask(ids, common_mask=None, first_tokens=2)
    assert torch.equal(ids, original)


def test_protect_token_ids_unmasks_single_token_variants():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[42, 50, 60]] = True
    tok = FakeTokenizer(
        {" transformer": [42], "transformer": [50]},
        {42: " transformer", 50: "transformer"},
    )
    touched = protect_token_ids(mask, tok, ["transformer"])
    assert not bool(mask[42])
    assert not bool(mask[50])
    assert bool(mask[60])
    assert touched["transformer"] == [42, 50]


def test_protect_token_ids_unmasks_long_multi_piece_variants():
    """Multi-piece variant where EVERY piece clears min_piece_chars (default 3)."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[10, 20]] = True
    tok = FakeTokenizer(
        {" diffusion": [10, 20]},
        {10: " diff", 20: "usion"},
    )
    touched = protect_token_ids(mask, tok, ["diffusion"])
    assert not bool(mask[10])
    assert not bool(mask[20])
    assert touched["diffusion"] == [10, 20]


def test_protect_token_ids_rejects_variant_with_any_short_piece():
    """ANY piece below min_piece_chars => skip the WHOLE variant (do not partially unmask)."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[1, 7]] = True
    tok = FakeTokenizer(
        {" VAE": [1, 7]},
        {1: " V", 7: "AE"},
    )
    touched = protect_token_ids(mask, tok, ["VAE"])
    assert bool(mask[1]) and bool(mask[7])
    assert touched["VAE"] == []


def test_protect_token_ids_min_piece_chars_is_tunable():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[1, 7]] = True
    tok = FakeTokenizer(
        {" VAE": [1, 7]},
        {1: " V", 7: "AE"},
    )
    protect_token_ids(mask, tok, ["VAE"])
    assert bool(mask[1]) and bool(mask[7])
    touched = protect_token_ids(mask, tok, ["VAE"], min_piece_chars=1)
    assert not bool(mask[1]) and not bool(mask[7])
    assert touched["VAE"] == [1, 7]


def test_protect_token_ids_length_gate_strips_leading_space():
    """Length gate counts visible characters, not raw len() including the leading space."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer(
        {" transformer": [5]},
        {5: " transformer"},
    )
    touched = protect_token_ids(mask, tok, ["transformer"])
    assert not bool(mask[5])
    assert touched["transformer"] == [5]


def test_protect_token_ids_reports_already_unmasked_as_empty():
    mask = torch.zeros(100, dtype=torch.bool)
    tok = FakeTokenizer(
        {" gradient": [55]},
        {55: " gradient"},
    )
    touched = protect_token_ids(mask, tok, ["gradient"])
    assert touched["gradient"] == []


def test_protect_token_ids_mutates_mask_in_place():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[3] = True
    tok = FakeTokenizer({" xyz": [3]}, {3: " xyz"})
    before_ptr = mask.data_ptr()
    protect_token_ids(mask, tok, ["xyz"])
    assert mask.data_ptr() == before_ptr
    assert not bool(mask[3])


def test_protect_token_ids_ignores_out_of_range_ids():
    mask = torch.zeros(10, dtype=torch.bool)
    tok = FakeTokenizer({" xyz": [99]}, {99: " xyz"})
    touched = protect_token_ids(mask, tok, ["xyz"])
    assert touched["xyz"] == []


def test_protect_token_ids_handles_empty_terms_iterable():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({})
    touched = protect_token_ids(mask, tok, [])
    assert bool(mask[5])
    assert touched == {}


def test_protect_token_ids_skips_empty_strings_in_terms():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0] = True
    tok = FakeTokenizer({"": [0], " ": [0]}, {0: ""})
    touched = protect_token_ids(mask, tok, ["", "valid"])
    assert bool(mask[0])
    assert "" not in touched
    assert "valid" in touched and touched["valid"] == []


def test_protect_token_ids_tries_all_case_and_space_variants():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[77] = True
    # Only the capitalized leading-space variant tokenizes as 1 piece clearing the gate.
    tok = FakeTokenizer({" Markov": [77]}, {77: " Markov"})
    touched = protect_token_ids(mask, tok, ["markov"])
    assert not bool(mask[77])
    assert 77 in touched["markov"]


def test_load_reference_counts_empty_path_returns_none():
    counts, meta = load_reference_counts("", expected_vocab_size=100)
    assert counts is None
    assert meta is None


def test_load_reference_counts_missing_file_returns_none(tmp_path):
    counts, meta = load_reference_counts(
        str(tmp_path / "does-not-exist.pt"), expected_vocab_size=100,
    )
    assert counts is None
    assert meta is None


def test_load_reference_counts_loads_valid_file(tmp_path):
    import torch
    f = tmp_path / "ref.pt"
    counts = torch.tensor([10, 20, 30, 40, 0], dtype=torch.int64)
    torch.save({
        "counts": counts,
        "dataset_id": "wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "n_tokens": 100,
        "n_docs": 5,
        "vocab_size": 5,
    }, str(f))
    loaded_counts, meta = load_reference_counts(
        str(f), expected_vocab_size=5,
    )
    assert torch.equal(loaded_counts, counts)
    assert meta["dataset_id"] == "wikitext"
    assert meta["n_tokens"] == 100


def test_load_reference_counts_vocab_mismatch_raises(tmp_path):
    import torch
    f = tmp_path / "ref.pt"
    counts = torch.tensor([1, 2, 3], dtype=torch.int64)
    torch.save({"counts": counts}, str(f))
    with pytest.raises(RuntimeError, match="vocab_size"):
        load_reference_counts(str(f), expected_vocab_size=5)


def test_protect_numeric_tokens_unmasks_single_digits():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[15, 16, 17, 18]] = True
    tok = FakeTokenizer({}, {15: '0', 16: '1', 17: '2', 18: '3'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [15, 16, 17, 18]
    assert not any(bool(mask[t]) for t in [15, 16, 17, 18])


def test_protect_numeric_tokens_unmasks_multi_digit():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[20, 25]] = True
    tok = FakeTokenizer({}, {20: ' 100', 25: ' 1024'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [20, 25]


def test_protect_numeric_tokens_skips_punctuation_attached():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[30, 31, 32, 33]] = True
    tok = FakeTokenizer({}, {30: '1.', 31: '0)', 32: 'e-4', 33: '1st'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []
    assert all(bool(mask[t]) for t in [30, 31, 32, 33])


def test_protect_numeric_tokens_skips_already_unmasked():
    mask = torch.zeros(100, dtype=torch.bool)
    mask[15] = True
    tok = FakeTokenizer({}, {15: '0', 16: '1'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [15]


def test_protect_numeric_tokens_handles_empty_masked_set():
    mask = torch.zeros(20, dtype=torch.bool)
    tok = FakeTokenizer({}, {})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []


def test_protect_numeric_tokens_handles_whitespace_only_tokens():
    """Stripped form must be truthy, not just isdigit(): '' is not a digit string."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({}, {5: '   '})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []
    assert bool(mask[5])


def test_protect_numeric_tokens_mutates_in_place():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[3] = True
    before_ptr = mask.data_ptr()
    tok = FakeTokenizer({}, {3: '5'})
    protect_numeric_tokens(mask, tok)
    assert mask.data_ptr() == before_ptr
    assert not bool(mask[3])


def test_protect_symbol_tokens_unmasks_listed_symbols():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[5, 6, 7]] = True
    tok = FakeTokenizer({}, {5: '=', 6: '@', 7: '^'})
    flipped = protect_symbol_tokens(mask, tok, {"=", "@", "^"})
    assert flipped == [5, 6, 7]
    assert not any(bool(mask[t]) for t in [5, 6, 7])


def test_protect_symbol_tokens_handles_leading_space_form():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[5, 9]] = True
    tok = FakeTokenizer({}, {5: '=', 9: ' ='})
    flipped = protect_symbol_tokens(mask, tok, {"="})
    assert flipped == [5, 9]


def test_protect_symbol_tokens_skips_non_listed():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[5, 6]] = True
    tok = FakeTokenizer({}, {5: '=', 6: '%'})
    flipped = protect_symbol_tokens(mask, tok, {"="})
    assert flipped == [5]
    assert bool(mask[6])


def test_protect_symbol_tokens_accepts_tuple_or_set():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({}, {5: '='})
    assert protect_symbol_tokens(mask, tok, ("=",)) == [5]
    mask[5] = True
    assert protect_symbol_tokens(mask, tok, {"="}) == [5]


def test_protect_symbol_tokens_empty_symbols_is_noop():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({}, {5: '='})
    flipped = protect_symbol_tokens(mask, tok, set())
    assert flipped == []
    assert bool(mask[5])


def test_protect_symbol_tokens_default_list_includes_critical_ops():
    from ttt_config import TRAIN_CFG
    defaults = set(TRAIN_CFG.loss_mask_protect_symbols)
    for required in ("=", "@", "^", "_", "\\", "|"):
        assert required in defaults, required


def test_protect_by_predicate_unmasks_only_matching_ids():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[1, 2, 3]] = True
    tok = FakeTokenizer({}, {1: "abc", 2: "xyz", 3: "abc"})
    flipped = protect_by_predicate(mask, tok, lambda s: s == "abc")
    assert flipped == [1, 3]
    assert bool(mask[2])


def test_protect_by_predicate_skips_already_unmasked():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[1] = True
    tok = FakeTokenizer({}, {1: "abc", 2: "abc"})
    flipped = protect_by_predicate(mask, tok, lambda s: s == "abc")
    assert flipped == [1]


def test_protect_by_predicate_handles_no_masked_ids():
    mask = torch.zeros(10, dtype=torch.bool)
    tok = FakeTokenizer({}, {})
    assert protect_by_predicate(mask, tok, lambda s: True) == []


def test_apply_protect_passes_aggregates_all_three_freed_counts():
    mask = torch.zeros(30, dtype=torch.bool)
    mask[[10, 20, 25]] = True
    tok = FakeTokenizer(
        {" diff": [10], "diff": [10]},
        {10: " diff", 20: "0", 25: "="},
    )
    n_t, n_n, n_s = apply_protect_passes(
        mask, tok, protect_terms=("diff",),
        protect_numeric=True, protect_symbols=("=",),
    )
    assert n_t == 1
    assert n_n == 1
    assert n_s == 1
    assert not any(bool(mask[t]) for t in [10, 20, 25])


def test_apply_protect_passes_all_disabled_is_noop():
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({}, {5: '='})
    n_t, n_n, n_s = apply_protect_passes(
        mask, tok, protect_terms=None,
        protect_numeric=False, protect_symbols=(),
    )
    assert (n_t, n_n, n_s) == (0, 0, 0)
    assert bool(mask[5])


def test_protect_token_ids_default_list_is_non_empty():
    """Default list must be a non-empty tuple of stripped strings (catches the `("foo")` typo)."""
    from ttt_config import LOSS_MASK_DEFAULT_PROTECT_TERMS
    assert isinstance(LOSS_MASK_DEFAULT_PROTECT_TERMS, tuple)
    assert len(LOSS_MASK_DEFAULT_PROTECT_TERMS) > 30
    assert all(isinstance(t, str) and t for t in LOSS_MASK_DEFAULT_PROTECT_TERMS)
    assert all(t == t.strip() for t in LOSS_MASK_DEFAULT_PROTECT_TERMS)


def test_build_rejects_keep_fraction_outside_unit_interval():
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 3]], vocab_size=10, keep_fraction=0.0)
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 3]], vocab_size=10, keep_fraction=-0.1)
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 3]], vocab_size=10, keep_fraction=1.1)


def test_build_rejects_nonpositive_vocab_size():
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 3]], vocab_size=0, keep_fraction=0.5)


def test_build_keep_fraction_one_returns_all_false():
    mask = build_common_token_mask(
        [[1, 2, 3, 4]] * 5, vocab_size=10, keep_fraction=1.0,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (10,)
    assert not mask.any()


def test_build_empty_corpus_returns_all_false():
    mask = build_common_token_mask([], vocab_size=10, keep_fraction=0.5)
    assert not mask.any()


def test_build_zero_length_sequences_returns_all_false():
    mask = build_common_token_mask([[], []], vocab_size=10, keep_fraction=0.5)
    assert not mask.any()


def test_build_rejects_out_of_range_token_id():
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 99]], vocab_size=10, keep_fraction=0.5)


def test_build_masks_the_dominant_token_first():
    corpus = [[7] * 90 + [1, 2, 3, 4, 5, 6, 8, 9, 0, 4]]
    mask = build_common_token_mask(corpus, vocab_size=10, keep_fraction=0.5)
    assert bool(mask[7])
    # 7 alone covers 90 of 100 positions; exactly one id is masked.
    assert int(mask.sum()) == 1


def test_build_kept_position_coverage_tracks_keep_fraction():
    g = torch.Generator().manual_seed(0)
    corpus = [torch.randint(0, 200, (5000,), generator=g).tolist()]
    vocab = 200

    counts = torch.bincount(torch.tensor(corpus[0]), minlength=vocab).double()
    total = counts.sum()

    for kf in (0.2, 0.5, 0.8):
        mask = build_common_token_mask(corpus, vocab_size=vocab,
                                       keep_fraction=kf)
        kept_share = float((counts * (~mask).double()).sum() / total)
        # Granularity bound: at worst under-cover the drop by one token-id's
        # frequency, ~max_count/total. Loose 5% slack.
        slack = float(counts.max() / total) + 0.05
        assert abs(kept_share - kf) <= slack, (kf, kept_share, slack)


def test_build_is_deterministic():
    corpus = [[i % 17 for i in range(1000)], [j % 23 for j in range(800)]]
    a = build_common_token_mask(corpus, vocab_size=50, keep_fraction=0.4)
    b = build_common_token_mask(corpus, vocab_size=50, keep_fraction=0.4)
    assert torch.equal(a, b)


def test_build_iterator_input_not_just_lists():
    corpus = [[0, 0, 0, 1, 2, 3]] * 10
    mask = build_common_token_mask(
        (row for row in corpus), vocab_size=10, keep_fraction=0.5,
    )
    assert bool(mask[0])


def test_apply_with_built_mask_keeps_only_rare_token_positions():
    corpus = [[7] * 90 + list(range(10))]
    mask = build_common_token_mask(corpus, vocab_size=10, keep_fraction=0.3)

    ids = torch.tensor([[7, 1, 7, 2, 7, 3]], dtype=torch.long)
    labels = apply_loss_mask(ids, mask)

    kept = labels != -100
    for keep, tok in zip(kept.view(-1).tolist(), ids.view(-1).tolist()):
        assert keep == (not bool(mask[tok]))
