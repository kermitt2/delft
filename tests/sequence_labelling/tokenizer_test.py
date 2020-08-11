import logging

from delft.utilities.Tokenizer import tokenizeAndFilter


class TestTokenizer:
    LOGGER = logging.getLogger(__name__)

    def test_tokenize_and_filter(self):
        tokens, offsets = tokenizeAndFilter("this is a simple text")

        assert len(tokens) == len(offsets)
        assert len(tokens) == 5
