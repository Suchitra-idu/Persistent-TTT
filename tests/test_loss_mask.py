"""
Content-token loss masking. The mask must be deterministic, leave the
input tensor untouched, behave as a clean no-op when the feature is
toggled off, and produce a kept-token set whose corpus coverage tracks
keep_fraction within the discretization error of "drop whole token ids."
"""

import pytest
import torch

from train_utils import (
    apply_loss_mask, build_common_token_mask, load_reference_counts,
    protect_numeric_tokens, protect_token_ids,
)


class FakeTokenizer:
    """Minimal stub satisfying .encode/.decode for protect_token_ids.

    encode_map: {text -> [ids]} for forward tokenization (missing keys
        return []).
    decode_map: {id -> string} for reverse decoding, used by the
        length-floor gate that decides whether a multi-piece variant
        clears min_piece_chars. Defaults to {} -- missing ids decode
        to '' which conservatively fails the length gate, mirroring
        the strict 'don't unmask what we can't characterize' behavior.
    """

    def __init__(self, encode_map, decode_map=None):
        self._encode = encode_map
        self._decode = decode_map or {}

    def encode(self, text, add_special_tokens=False):
        return list(self._encode.get(text, []))

    def decode(self, ids):
        return "".join(self._decode.get(int(tid), "") for tid in ids)


# --------------------------------------------------------------------- apply --
def test_apply_loss_mask_disabled_returns_ids_unchanged():
    """common_mask=None is the at-runtime kill switch -- ids must come
    back identical and untouched (not even cloned), so wiring the helper
    into the loss path unconditionally costs nothing when disabled."""
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    out = apply_loss_mask(ids, None)
    assert out is ids


def test_apply_loss_mask_does_not_mutate_input():
    """The hot loop reuses `ids` for the forward pass; the mask helper
    must produce a separate labels tensor instead of corrupting the
    input. A regression here would silently break the forward."""
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
    """Same shape + dtype is what makes labels= a drop-in replacement
    for ids= in the HF causal-LM API. Mismatch would surface as a noisy
    runtime error inside the model."""
    ids = torch.randint(0, 100, (2, 3, 7), dtype=torch.long)
    mask = torch.zeros(100, dtype=torch.bool)
    out = apply_loss_mask(ids, mask)
    assert out.shape == ids.shape
    assert out.dtype == ids.dtype


def test_apply_loss_mask_empty_mask_returns_clone_equal_to_ids():
    """Mask with no marked tokens => labels equal ids in value (but a
    distinct tensor, since we always clone when masking is active)."""
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    mask = torch.zeros(512, dtype=torch.bool)
    out = apply_loss_mask(ids, mask)
    assert torch.equal(out, ids)
    assert out.data_ptr() != ids.data_ptr()


def test_apply_loss_mask_all_masked_returns_all_ignore_index():
    """Degenerate but valid: every position becomes -100. The training
    loop's nonfinite-loss guard catches the NaN this produces in HF's
    CrossEntropyLoss; the helper itself does not have to special-case."""
    ids = torch.tensor([[5, 10, 200]], dtype=torch.long)
    mask = torch.zeros(512, dtype=torch.bool)
    mask[[5, 10, 200]] = True
    out = apply_loss_mask(ids, mask)
    assert (out == -100).all()


# ------------------------------------------------- first-token masking --
def test_apply_loss_mask_first_tokens_only_masks_leading_positions():
    """first_tokens=N with common_mask=None must mask exactly positions
    0..N-1 on every row and leave the rest as ids. This is the 'skip
    the shared paper opening' path."""
    ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=2)
    assert out.tolist() == [[-100, -100, 30, 40, 50]]


def test_apply_loss_mask_first_tokens_clamps_to_seq_length():
    """first_tokens larger than the sequence must clamp, not crash.
    Mid-paper slices shorter than the boilerplate-prefix length are
    a real possibility under aggressive slicing."""
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=99)
    assert (out == -100).all()


def test_apply_loss_mask_first_tokens_zero_is_no_op():
    """first_tokens=0 must be a clean no-op even with no common_mask.
    Combined with common_mask=None this is the at-runtime kill switch
    for the entire feature."""
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=0)
    assert out is ids


def test_apply_loss_mask_first_tokens_composes_with_common_mask():
    """Both masking sources active: positions 0..N-1 AND any positions
    whose token is in common_mask must be -100. The two paths are
    independent and union together."""
    ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[40] = True
    out = apply_loss_mask(ids, common_mask=mask, first_tokens=2)
    assert out.tolist() == [[-100, -100, 30, -100, 50]]


def test_apply_loss_mask_first_tokens_supports_batched_input():
    """Leading-position mask must work on rank >= 2 tensors; the hot
    loop uses [B=1, N] but session_training=True with B>1 (hypothetical
    future) shouldn't surprise the helper."""
    ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long)
    out = apply_loss_mask(ids, common_mask=None, first_tokens=1)
    assert out.tolist() == [[-100, 20, 30], [-100, 50, 60]]


def test_apply_loss_mask_first_tokens_does_not_mutate_input():
    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    original = ids.clone()
    apply_loss_mask(ids, common_mask=None, first_tokens=2)
    assert torch.equal(ids, original)


# ----------------------------------------------------- protect-list --
def test_protect_token_ids_unmasks_single_token_variants():
    """The standard case: term tokenizes to one BPE piece in at least
    one variant; that piece must be force-unmasked even if it's a
    high-frequency token."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[42, 50, 60]] = True
    tok = FakeTokenizer(
        {" transformer": [42], "transformer": [50]},
        {42: " transformer", 50: "transformer"},
    )
    touched = protect_token_ids(mask, tok, ["transformer"])
    assert not bool(mask[42])
    assert not bool(mask[50])
    assert bool(mask[60])               # untouched id stays masked
    assert touched["transformer"] == [42, 50]


def test_protect_token_ids_unmasks_long_multi_piece_variants():
    """Multi-piece variant where EVERY piece clears min_piece_chars
    (default 3 stripped chars). Realistic example: ' diffusion' ->
    [' diff', 'usion'] -- both pieces are 4-5 chars, both unmask. The
    user gets the full word's gradient signal even when BPE splits."""
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
    """Multi-piece variant where ANY piece falls below min_piece_chars:
    skip the WHOLE variant (do not partially unmask). The classic
    failure this prevents is acronym splits like ' VAE' -> [' V', 'AE']
    where unmasking ' V' would fire on every capital-V mid-sentence
    word, silently shrinking the mask far beyond what the user asked
    for. Better to leave the acronym unprotected than to leak."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[1, 7]] = True
    tok = FakeTokenizer(
        {" VAE": [1, 7]},
        {1: " V", 7: "AE"},               # 1 char and 2 chars stripped
    )
    touched = protect_token_ids(mask, tok, ["VAE"])
    assert bool(mask[1]) and bool(mask[7])
    assert touched["VAE"] == []


def test_protect_token_ids_min_piece_chars_is_tunable():
    """Lowering min_piece_chars admits acronym splits that the default
    rejects. Tests both that the parameter is plumbed and that the
    default value is the strict (=3) behavior."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[1, 7]] = True
    tok = FakeTokenizer(
        {" VAE": [1, 7]},
        {1: " V", 7: "AE"},
    )
    # Default: rejected.
    protect_token_ids(mask, tok, ["VAE"])
    assert bool(mask[1]) and bool(mask[7])
    # Lowered floor: 'AE' is 2 chars stripped, 'V' is 1 char; relax
    # the gate to 1 and both clear.
    touched = protect_token_ids(mask, tok, ["VAE"], min_piece_chars=1)
    assert not bool(mask[1]) and not bool(mask[7])
    assert touched["VAE"] == [1, 7]


def test_protect_token_ids_length_gate_strips_leading_space():
    """' transformer' decodes with a leading space, but the gate must
    count the visible characters -- 'transformer' is 11 chars, gate
    passes. A buggy implementation that used raw len() would count
    the space and still pass here, but would also wrongly pass ' V'
    (2 raw chars)."""
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
    """A term whose every passing variant maps to an already-unmasked
    id reports empty in the touched dict. Distinguishes 'protect-list
    redundant for this term' from 'no variant cleared the gate.' Both
    look like empty -- the diagnostic surfaces this so the user can
    audit."""
    mask = torch.zeros(100, dtype=torch.bool)   # nothing masked
    tok = FakeTokenizer(
        {" gradient": [55]},
        {55: " gradient"},
    )
    touched = protect_token_ids(mask, tok, ["gradient"])
    assert touched["gradient"] == []


def test_protect_token_ids_mutates_mask_in_place():
    """Caller relies on in-place mutation -- the mask was already
    moved/copied with its final shape, returning a new tensor would
    silently desync the GPU copy."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[3] = True
    tok = FakeTokenizer({" xyz": [3]}, {3: " xyz"})
    before_ptr = mask.data_ptr()
    protect_token_ids(mask, tok, ["xyz"])
    assert mask.data_ptr() == before_ptr
    assert not bool(mask[3])


def test_protect_token_ids_ignores_out_of_range_ids():
    """A tokenizer that ever emits an id >= vocab_size is malformed for
    our mask, but the helper must skip those entries rather than
    crash with an IndexError -- the build pass already validates ids."""
    mask = torch.zeros(10, dtype=torch.bool)
    tok = FakeTokenizer({" xyz": [99]}, {99: " xyz"})
    touched = protect_token_ids(mask, tok, ["xyz"])
    # No crash; oversized id just doesn't protect anything.
    assert touched["xyz"] == []


def test_protect_token_ids_handles_empty_terms_iterable():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({})
    touched = protect_token_ids(mask, tok, [])
    assert bool(mask[5])                 # unchanged
    assert touched == {}


def test_protect_token_ids_skips_empty_strings_in_terms():
    """A stray empty string in the protect-list (e.g. from a trailing
    comma in config) must be silently skipped, not encoded into
    arbitrary special tokens that happen to live at id 0."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0] = True
    tok = FakeTokenizer({"": [0], " ": [0]}, {0: ""})
    touched = protect_token_ids(mask, tok, ["", "valid"])
    assert bool(mask[0])                 # id 0 stayed masked
    assert "" not in touched
    assert "valid" in touched and touched["valid"] == []


def test_protect_token_ids_tries_all_case_and_space_variants():
    """The contract is 'we try 8 variants per term'; spot-check that
    a term protected only via its capitalized leading-space variant
    still gets caught. Models that store a Title-case proper noun as
    one token can rely on this."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[77] = True
    # Only the capitalized leading-space variant tokenizes as 1 piece
    # that clears the gate.
    tok = FakeTokenizer({" Markov": [77]}, {77: " Markov"})
    touched = protect_token_ids(mask, tok, ["markov"])
    assert not bool(mask[77])
    assert 77 in touched["markov"]


# --------------------------------------------- reference-counts load --
def test_load_reference_counts_empty_path_returns_none():
    """The fallback signal: an empty config path means 'no reference,
    use in-corpus.' Must return (None, None) without touching disk."""
    counts, meta = load_reference_counts("", expected_vocab_size=100)
    assert counts is None
    assert meta is None


def test_load_reference_counts_missing_file_returns_none(tmp_path):
    """A configured-but-missing path is the first-run UX (config
    points at a file that build_reference_counts hasn't produced yet).
    Must NOT raise -- training has to fall back gracefully."""
    counts, meta = load_reference_counts(
        str(tmp_path / "does-not-exist.pt"), expected_vocab_size=100,
    )
    assert counts is None
    assert meta is None


def test_load_reference_counts_loads_valid_file(tmp_path):
    """The happy path. Confirm the helper round-trips a torch.save'd
    dict and returns the counts tensor + the full metadata dict."""
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
    """A vocab-size mismatch is a tokenizer-change failure (e.g. the
    reference was built with Qwen3 but the run uses Qwen2). Must
    fail loud at startup with a clear message, not silently corrupt
    the mask via a shape-mismatched bincount later."""
    import torch
    f = tmp_path / "ref.pt"
    counts = torch.tensor([1, 2, 3], dtype=torch.int64)
    torch.save({"counts": counts}, str(f))
    with pytest.raises(RuntimeError, match="vocab_size"):
        load_reference_counts(str(f), expected_vocab_size=5)


# --------------------------------------------- numeric-token protect --
def test_protect_numeric_tokens_unmasks_single_digits():
    """Pure-digit single-char tokens (' 0', ' 1', ...) sit near the
    top of the frequency table in scientific corpora and would
    otherwise always be masked; this is the predicate path that
    rescues them since the term-based protect can't (its length gate
    rejects single-char pieces)."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[15, 16, 17, 18]] = True
    tok = FakeTokenizer({}, {15: '0', 16: '1', 17: '2', 18: '3'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [15, 16, 17, 18]
    assert not any(bool(mask[t]) for t in [15, 16, 17, 18])


def test_protect_numeric_tokens_unmasks_multi_digit():
    """BPE sometimes emits multi-digit pieces (' 100', ' 1024'). Those
    decode to a pure-digit string after stripping the leading space,
    so the same predicate catches them."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[20, 25]] = True
    tok = FakeTokenizer({}, {20: ' 100', 25: ' 1024'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [20, 25]


def test_protect_numeric_tokens_skips_punctuation_attached():
    """Forms like '1.' (section enumeration), '0)' (figure ref), 'e-4'
    (scientific notation piece) are not pure-digit and must NOT be
    unmasked -- they sit at the boundary between content and glue,
    and the predicate is explicitly the strict 'all digits' rule."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[30, 31, 32, 33]] = True
    tok = FakeTokenizer({}, {30: '1.', 31: '0)', 32: 'e-4', 33: '1st'})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []
    assert all(bool(mask[t]) for t in [30, 31, 32, 33])


def test_protect_numeric_tokens_skips_already_unmasked():
    """An already-unmasked numeric id must not appear in the flipped
    list -- 'flipped' means 'we flipped it from True to False this
    call,' not 'this id is now False.'"""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[15] = True
    tok = FakeTokenizer({}, {15: '0', 16: '1'})    # id 16 not masked
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == [15]


def test_protect_numeric_tokens_handles_empty_masked_set():
    """No masked ids => no decode calls, empty flipped list. Important
    edge case at very high keep_fraction or after the term protect
    has already stripped everything."""
    mask = torch.zeros(20, dtype=torch.bool)
    tok = FakeTokenizer({}, {})
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []


def test_protect_numeric_tokens_handles_whitespace_only_tokens():
    """Tokens that decode to pure whitespace strip to '' (falsy).
    The predicate must require the stripped form be truthy, not just
    isdigit() -- otherwise '' would be treated as a 'digit string'
    (Python: ''.isdigit() is False, so this is belt-and-suspenders,
    but worth pinning so a refactor can't silently change it)."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[5] = True
    tok = FakeTokenizer({}, {5: '   '})           # all whitespace
    flipped = protect_numeric_tokens(mask, tok)
    assert flipped == []
    assert bool(mask[5])                           # untouched


def test_protect_numeric_tokens_mutates_in_place():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[3] = True
    before_ptr = mask.data_ptr()
    tok = FakeTokenizer({}, {3: '5'})
    protect_numeric_tokens(mask, tok)
    assert mask.data_ptr() == before_ptr
    assert not bool(mask[3])


def test_protect_token_ids_default_list_is_non_empty():
    """The default list from ttt_config must be non-empty AND contain
    string entries, not accidentally a single-string-as-tuple. Catches
    the classic `("foo")` typo where the parentheses don't make a tuple."""
    from ttt_config import LOSS_MASK_DEFAULT_PROTECT_TERMS
    assert isinstance(LOSS_MASK_DEFAULT_PROTECT_TERMS, tuple)
    assert len(LOSS_MASK_DEFAULT_PROTECT_TERMS) > 30
    assert all(isinstance(t, str) and t for t in LOSS_MASK_DEFAULT_PROTECT_TERMS)
    # No accidental whitespace baked into terms -- protect_token_ids
    # already prepends leading spaces internally.
    assert all(t == t.strip() for t in LOSS_MASK_DEFAULT_PROTECT_TERMS)


# --------------------------------------------------------------------- build --
def test_build_rejects_keep_fraction_outside_unit_interval():
    """0 and >1 are nonsense (drop everything / drop more than exists)
    so they must fail loud at the boundary, not silently produce a
    surprising mask the user can't reason about."""
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
    """keep_fraction=1.0 is the config-level disable switch even when
    loss_mask_enabled=True: it must produce an all-False mask whose
    apply_loss_mask is a no-op (modulo cloning)."""
    mask = build_common_token_mask(
        [[1, 2, 3, 4]] * 5, vocab_size=10, keep_fraction=1.0,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (10,)
    assert not mask.any()


def test_build_empty_corpus_returns_all_false():
    """Empty corpus means no frequency information; safest default is
    'mask nothing' rather than crash."""
    mask = build_common_token_mask([], vocab_size=10, keep_fraction=0.5)
    assert not mask.any()


def test_build_zero_length_sequences_returns_all_false():
    mask = build_common_token_mask([[], []], vocab_size=10, keep_fraction=0.5)
    assert not mask.any()


def test_build_rejects_out_of_range_token_id():
    """A token id >= vocab_size would silently extend the bincount
    output and mismatch the mask shape callers index with. Catch at
    build time, where the error message can point at the wrong vocab."""
    with pytest.raises(ValueError):
        build_common_token_mask([[1, 2, 99]], vocab_size=10, keep_fraction=0.5)


def test_build_masks_the_dominant_token_first():
    """In a corpus that's 90% token 7, dropping ~50% of positions can
    only be done by masking token 7."""
    corpus = [[7] * 90 + [1, 2, 3, 4, 5, 6, 8, 9, 0, 4]]
    mask = build_common_token_mask(corpus, vocab_size=10, keep_fraction=0.5)
    assert bool(mask[7])
    # 7 alone covers 90 of 100 positions; no other id needed to hit the
    # ~50% drop budget, so exactly one id is masked.
    assert int(mask.sum()) == 1


def test_build_kept_position_coverage_tracks_keep_fraction():
    """The corpus-position coverage of the kept (un-masked) set should
    sit at keep_fraction, plus or minus the granularity of dropping a
    whole token id. With a high-cardinality corpus the granularity is
    small."""
    g = torch.Generator().manual_seed(0)
    corpus = [torch.randint(0, 200, (5000,), generator=g).tolist()]
    vocab = 200

    counts = torch.bincount(torch.tensor(corpus[0]), minlength=vocab).double()
    total = counts.sum()

    for kf in (0.2, 0.5, 0.8):
        mask = build_common_token_mask(corpus, vocab_size=vocab,
                                       keep_fraction=kf)
        kept_share = float((counts * (~mask).double()).sum() / total)
        # Granularity bound: at worst we under-cover the drop by exactly
        # one token-id's frequency, ~max_count/total. Loose 5% slack.
        slack = float(counts.max() / total) + 0.05
        assert abs(kept_share - kf) <= slack, (kf, kept_share, slack)


def test_build_is_deterministic():
    corpus = [[i % 17 for i in range(1000)], [j % 23 for j in range(800)]]
    a = build_common_token_mask(corpus, vocab_size=50, keep_fraction=0.4)
    b = build_common_token_mask(corpus, vocab_size=50, keep_fraction=0.4)
    assert torch.equal(a, b)


def test_build_iterator_input_not_just_lists():
    """The training-loop call site passes a generator over
    ds[i]['input_ids']; verify the helper does not require a list."""
    corpus = [[0, 0, 0, 1, 2, 3]] * 10
    mask = build_common_token_mask(
        (row for row in corpus), vocab_size=10, keep_fraction=0.5,
    )
    # Token 0 dominates and must be in the masked set.
    assert bool(mask[0])


# -------------------------------------------------------------- integration --
def test_apply_with_built_mask_keeps_only_rare_token_positions():
    """End-to-end: build the mask from a corpus, apply it to a sequence
    drawn from the same distribution, verify the kept (label != -100)
    positions are exactly those whose token id is not in the masked set.
    This is what the training loop actually does."""
    corpus = [[7] * 90 + list(range(10))]
    mask = build_common_token_mask(corpus, vocab_size=10, keep_fraction=0.3)

    ids = torch.tensor([[7, 1, 7, 2, 7, 3]], dtype=torch.long)
    labels = apply_loss_mask(ids, mask)

    kept = labels != -100
    # Every kept position holds a token not in the mask.
    for keep, tok in zip(kept.view(-1).tolist(), ids.view(-1).tolist()):
        assert keep == (not bool(mask[tok]))
