from __future__ import annotations

import pytest

from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.hf_tokenizer import HfTokenizer
from ttt.core.sampling_text import CHAT_SPECIAL_TOKENS, stop_token_ids
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

    def test_the_eos_token_has_an_id(self, tokenizer):
        assert isinstance(tokenizer.eos_token_id, int)

    def test_an_unknown_token_has_no_id(self, tokenizer):
        assert tokenizer.token_id("<|not-a-real-token|>") is None

    def test_the_chat_specials_all_have_ids(self, tokenizer):
        assert all(tokenizer.token_id(t) is not None for t in CHAT_SPECIAL_TOKENS)

    def test_the_stop_set_is_built_from_real_ids(self, tokenizer):
        assert tokenizer.eos_token_id in stop_token_ids(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            special_token_ids={t: tokenizer.token_id(t) for t in CHAT_SPECIAL_TOKENS},
            unk_token_id=tokenizer.unk_token_id,
        )

    def test_the_chat_template_wraps_the_message(self, tokenizer):
        rendered = tokenizer.apply_chat_template(MESSAGES)

        assert "hello" in rendered and rendered.endswith("\n")

    def test_the_generation_prompt_is_optional(self, tokenizer):
        with_prompt = tokenizer.apply_chat_template(
            MESSAGES, add_generation_prompt=True
        )
        without = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=False)

        assert len(with_prompt) > len(without)


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

    def test_no_pad_token_unless_asked_for(self, tokenizer):
        assert tokenizer.pad_token_id is None

    def test_disabling_thinking_closes_the_think_block(self, tokenizer):
        rendered = tokenizer.apply_chat_template(MESSAGES, enable_thinking=False)

        assert rendered.endswith("<think>\n\n</think>\n\n")


@pytest.mark.integration
class TestHfTokenizer(TokenizerConformance):
    @pytest.fixture
    def tokenizer(self):
        pytest.importorskip("transformers")
        return HfTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
