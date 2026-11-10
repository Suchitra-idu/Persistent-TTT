from __future__ import annotations

import pytest

from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.hf_tokenizer import HfTokenizer
from ttt.ports.tokenizer import Tokenizer

TEXT = "the carry is the research"
MESSAGES = ({"role": "user", "content": "hello"},)


class TokenizerConformance:
    @pytest.fixture
    def tokenizer(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, tokenizer):
        assert isinstance(tokenizer, Tokenizer)

    def test_encode_then_decode_round_trips(self, tokenizer):
        assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT

    def test_encoding_is_deterministic(self, tokenizer):
        assert tokenizer.encode(TEXT) == tokenizer.encode(TEXT)

    def test_empty_text_encodes_to_no_tokens(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_max_length_truncates(self, tokenizer):
        assert len(tokenizer.encode(TEXT, max_length=3)) == 3

    def test_a_batch_matches_encoding_one_by_one(self, tokenizer):
        texts = [TEXT, "another", ""]

        assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]

    def test_an_empty_batch_encodes_to_no_rows(self, tokenizer):
        """A holdout pool a length filter has emptied out still calls this
        (data_pipeline._encode) — must return [], not raise."""
        assert tokenizer.encode_batch([]) == []

class TestFakeTokenizer(TokenizerConformance):
    @pytest.fixture
    def tokenizer(self):
        return FakeTokenizer()

    def test_it_round_trips_non_ascii(self, tokenizer):
        text = "carré — δ"

        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_specials_survive_a_round_trip(self, tokenizer):
        text = "a<|im_end|>b"

        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_specials_can_be_skipped_on_decode(self, tokenizer):
        ids = tokenizer.encode("a<|im_end|>b")

        assert tokenizer.decode(ids, skip_special_tokens=True) == "ab"

    def test_a_special_token_is_one_token(self, tokenizer):
        assert len(tokenizer.encode("<|im_end|>")) == 1

@pytest.mark.integration
class TestHfTokenizer(TokenizerConformance):
    @pytest.fixture
    def tokenizer(self):
        pytest.importorskip("transformers")
        return HfTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
