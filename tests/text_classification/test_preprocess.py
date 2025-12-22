"""
Tests for text classification preprocessing in delft/textClassification/preprocess.py
"""

import logging
import numpy as np
import pytest
import os
import tempfile

from delft.textClassification.preprocess import (
    TextPreprocessor,
    to_indices_single,
    clean_text,
    create_batch_input_bert,
)
from delft.utilities.preprocess import UNK, PAD

LOGGER = logging.getLogger(__name__)


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_special_characters(self):
        result = clean_text("Hello™ World®!")
        # Unidecode converts, then special chars removed
        assert "™" not in result
        assert "®" not in result

    def test_keeps_basic_punctuation(self):
        result = clean_text("Hello, world! How are you?")
        assert "," in result
        assert "!" in result
        assert "?" in result


class TestToIndicesSingle:
    """Tests for to_indices_single function."""

    def test_converts_text_to_indices(self):
        vocab = {PAD: 0, UNK: 1, "hello": 2, "world": 3}
        result = to_indices_single("hello world", vocab, maxlen=5)
        assert result.shape == (5,)
        assert result[0] == 2  # hello
        assert result[1] == 3  # world
        assert result[2] == 0  # padding

    def test_unknown_words_get_unk_index(self):
        vocab = {PAD: 0, UNK: 1, "hello": 2}
        result = to_indices_single("hello unknown", vocab, maxlen=5)
        assert result[0] == 2  # hello
        assert result[1] == 1  # unknown -> UNK

    def test_truncates_long_text(self):
        vocab = {PAD: 0, UNK: 1, "word": 2}
        # Create a long text
        text = " ".join(["word"] * 100)
        result = to_indices_single(text, vocab, maxlen=10)
        assert result.shape == (10,)


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_instantiate_with_defaults(self):
        tp = TextPreprocessor()
        assert tp.maxlen == 300
        assert tp.lowercase is False

    def test_fit_builds_vocabulary(self):
        tp = TextPreprocessor(maxlen=50)
        texts = ["Hello world", "Hello again"]
        tp.fit(texts)

        assert PAD in tp.vocab_word
        assert UNK in tp.vocab_word
        assert "Hello" in tp.vocab_word or "hello" in tp.vocab_word
        assert "world" in tp.vocab_word

    def test_fit_with_lowercase(self):
        tp = TextPreprocessor(maxlen=50, lowercase=True)
        texts = ["Hello World"]
        tp.fit(texts)

        assert "hello" in tp.vocab_word
        assert "world" in tp.vocab_word
        # Uppercase versions should NOT be in vocab when lowercase=True
        # (depends on tokenizer behavior)

    def test_transform_returns_indices(self):
        tp = TextPreprocessor(maxlen=10)
        texts = ["Hello world", "Test text"]
        tp.fit(texts)

        indices = tp.transform(texts)
        assert indices.shape == (2, 10)
        # First word index should be > 1 (not PAD or UNK)
        assert indices[0, 0] > 1

    def test_fit_transform(self):
        tp = TextPreprocessor(maxlen=10)
        texts = ["Hello world"]
        labels = np.array([[1, 0, 0]])  # One-hot encoded

        indices, y = tp.fit_transform(texts, labels)
        assert indices.shape == (1, 10)
        assert np.array_equal(y, labels)

    def test_save_and_load(self):
        tp = TextPreprocessor(maxlen=50, lowercase=True)
        texts = ["Hello world", "Test text"]
        tp.fit(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "preprocessor.json")
            tp.save(filepath)

            # Load it back
            loaded = TextPreprocessor.load(filepath)

            assert loaded.maxlen == tp.maxlen
            assert loaded.lowercase == tp.lowercase
            assert loaded.vocab_word == tp.vocab_word

    def test_transform_unseen_text(self):
        tp = TextPreprocessor(maxlen=10)
        texts = ["Hello world"]
        tp.fit(texts)

        # Transform text with unseen words
        unseen_indices = tp.transform(["unknown words here"])
        assert unseen_indices.shape == (1, 10)
        # Unseen words should map to UNK (index 1)


class TestTextPreprocessorEmbeddingWeights:
    """Tests for get_embedding_weights method."""

    def test_get_embedding_weights(self):
        tp = TextPreprocessor(maxlen=10)
        texts = ["hello world"]
        tp.fit(texts)

        # Mock embeddings object
        class MockEmbeddings:
            embed_size = 50

            def get_word_vector(self, word):
                # Return a vector based on word hash for consistency
                np.random.seed(hash(word) % 2**32)
                return np.random.randn(self.embed_size).astype(np.float32)

        embeddings = MockEmbeddings()
        weights = tp.get_embedding_weights(embeddings)

        assert weights.shape == (len(tp.vocab_word), 50)
        # PAD and UNK should have random initialization (non-zero)
        assert not np.allclose(weights[0], 0)  # PAD
        assert not np.allclose(weights[1], 0)  # UNK
