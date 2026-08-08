from __future__ import annotations

import math

from ttt.adapters.fake_compute import FakeCompute
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.fake_wiki_lang_source import FakeWikiLangSource
from ttt.app import lang_scan

LANGUAGES = (("aa", "Language A"), ("bb", "Language B"))


def _scan(*, wiki_rows, losses=(1.0,), languages=LANGUAGES, target_docs=2, max_chars=100):
    wiki = FakeWikiLangSource(wiki_rows)
    compute = FakeCompute(losses=losses)
    fast_weights = FakeFastWeights(layer_indices=(0,))
    results = lang_scan.scan(
        languages=languages,
        snapshot="20231101",
        target_docs=target_docs,
        max_chars=max_chars,
        wiki_source=wiki,
        tokenizer=FakeTokenizer(),
        compute=compute,
        fast_weights=fast_weights,
        announce=lambda _: None,
    )
    return results, compute, wiki, fast_weights


def test_it_measures_every_language_with_rows():
    results, _, _, _ = _scan(
        wiki_rows={"aa": [{"text": "x" * 250}], "bb": [{"text": "y" * 250}]}
    )

    assert {r.code for r in results} == {"aa", "bb"}


def test_a_language_with_no_rows_is_skipped_not_crashed():
    results, _, _, _ = _scan(wiki_rows={"aa": [{"text": "x" * 250}]})

    assert {r.code for r in results} == {"aa"}


def test_eval_loss_is_always_called_without_lora():
    _, compute, _, _ = _scan(wiki_rows={"aa": [{"text": "x" * 250}], "bb": [{"text": "y" * 250}]})

    assert compute.eval_lora_flags and all(flag is False for flag in compute.eval_lora_flags)


def test_carry_is_reset_once_per_document():
    wiki_rows = {"aa": [{"text": "x" * 250}, {"text": "x" * 260}]}
    _, _, _, fast_weights = _scan(
        wiki_rows=wiki_rows, languages=(("aa", "Language A"),), target_docs=2
    )

    assert fast_weights.events.count("reset_carry") == 2


def test_ppl_matches_the_scripted_loss():
    results, _, _, _ = _scan(wiki_rows={"aa": [{"text": "x" * 250}]}, losses=(2.0,))

    assert math.isclose(results[0].ppl, math.exp(2.0), rel_tol=1e-9)


def test_bpb_is_lower_for_a_more_efficiently_tokenized_language():
    """FakeTokenizer is byte-level, so `aa`'s longer text yields more tokens
    per byte than `bb`'s shorter one at the same scripted per-token loss —
    bpb should track that, not disagree with it."""
    results, _, _, _ = _scan(
        wiki_rows={"aa": [{"text": "x" * 290}], "bb": [{"text": "y" * 210}]},
        losses=(1.0,),
        max_chars=1000,
    )
    by_code = {r.code: r for r in results}

    assert by_code["aa"].n_tokens > by_code["bb"].n_tokens


def test_max_chars_truncates_before_tokenizing():
    results, _, _, _ = _scan(
        wiki_rows={"aa": [{"text": "x" * 500}]}, languages=(("aa", "Language A"),), max_chars=10
    )

    assert results[0].n_bytes == 10
