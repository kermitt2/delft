"""
Tests for shared preprocessing utilities in delft/utilities/preprocess.py
"""

import logging
import numpy as np

from delft.utilities.preprocess import (
    UNK,
    PAD,
    case_index,
    pad_sequences,
    dense_to_one_hot,
    get_casing,
    lower,
    normalize_num,
    FeaturesPreprocessor,
    BERTPreprocessor,
)

LOGGER = logging.getLogger(__name__)


class TestConstants:
    """Test that constants are properly defined."""

    def test_unk_constant(self):
        assert UNK == "<UNK>"

    def test_pad_constant(self):
        assert PAD == "<PAD>"

    def test_case_index_contains_expected_keys(self):
        expected_keys = {
            "<PAD>",
            "numeric",
            "allLower",
            "allUpper",
            "initialUpper",
            "other",
            "mainly_numeric",
            "contains_digit",
        }
        assert set(case_index.keys()) == expected_keys


class TestPadSequences:
    """Tests for pad_sequences function."""

    def test_pad_sequences_single_level(self):
        sequences = [[1, 2], [3, 4, 5]]
        padded, lengths = pad_sequences(sequences, pad_tok=0, nlevels=1)
        assert len(padded) == 2
        assert len(padded[0]) == 3  # Max length is 3
        assert len(padded[1]) == 3
        assert padded[0] == [1, 2, 0]
        assert padded[1] == [3, 4, 5]
        assert lengths == [2, 3]

    def test_pad_sequences_two_level(self):
        # Char-level sequences: [[chars for word1], [chars for word2]]
        sequences = [[[1, 2], [3]], [[4, 5, 6]]]
        padded, lengths = pad_sequences(
            sequences, pad_tok=0, nlevels=2, max_char_length=4
        )
        assert len(padded) == 2
        # Check padding is applied
        assert len(padded[0]) == 2  # Two words in first sentence (max sentence length)
        assert len(padded[0][0]) == 4  # Max char length


class TestDenseToOneHot:
    """Tests for dense_to_one_hot function."""

    def test_one_hot_single_level(self):
        labels = np.array([0, 1, 2])
        one_hot = dense_to_one_hot(labels, num_classes=3, nlevels=1)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.array_equal(one_hot, expected)

    def test_one_hot_two_level(self):
        labels = np.array([[0, 1], [2, 0]])
        one_hot = dense_to_one_hot(labels, num_classes=3, nlevels=2)
        assert one_hot.shape == (2, 2, 3)
        # First sample, first label = 0
        assert one_hot[0, 0, 0] == 1
        # First sample, second label = 1
        assert one_hot[0, 1, 1] == 1


class TestGetCasing:
    """Tests for get_casing function."""

    def test_numeric(self):
        assert get_casing("12345") == case_index["numeric"]

    def test_all_lower(self):
        assert get_casing("hello") == case_index["allLower"]

    def test_all_upper(self):
        assert get_casing("HELLO") == case_index["allUpper"]

    def test_initial_upper(self):
        assert get_casing("Hello") == case_index["initialUpper"]

    def test_mainly_numeric(self):
        # More than 50% digits
        assert get_casing("12a") == case_index["mainly_numeric"]

    def test_contains_digit(self):
        # Words with some digits but doesn't match other patterns
        # Must not be: numeric, allLower, allUpper, initialUpper, mainly_numeric
        # "aB1c" is mixed case with digit - should be "contains_digit"
        result = get_casing("aB1c")
        assert result == case_index["contains_digit"]

    def test_other(self):
        assert get_casing("---") == case_index["other"]


class TestLowerAndNormalizeNum:
    """Tests for lowercase and number normalization."""

    def test_lower(self):
        assert lower("HeLLo") == "hello"

    def test_normalize_num(self):
        assert normalize_num("abc123") == "abc000"
        assert normalize_num("２０２３") == "0000"  # Full-width digits


class TestFeaturesPreprocessor:
    """Tests for FeaturesPreprocessor class."""

    def test_instantiate_with_defaults(self):
        fp = FeaturesPreprocessor()
        assert fp.features_vocabulary_size == 12

    def test_fit_empty(self):
        fp = FeaturesPreprocessor()
        fp.fit([])
        # Should not raise

    def test_fit_and_transform(self):
        fp = FeaturesPreprocessor()
        features = [[["a", "b"], ["c", "d"]]]
        transformed = fp.fit_transform(features)
        assert transformed is not None
        assert len(transformed) == 1

    def test_empty_features_vector(self):
        fp = FeaturesPreprocessor(features_indices=[0, 1])
        fp.features_indices = [0, 1]
        empty = fp.empty_features_vector()
        assert empty == [0, 0]


class TestBERTPreprocessor:
    """Tests for BERTPreprocessor class."""

    def test_infer_bpe_sp_roberta(self):
        class MockTokenizer:
            pass

        mock = MockTokenizer()
        mock.__class__.__name__ = "RobertaTokenizer"

        bp = BERTPreprocessor(mock)
        assert bp.is_BPE_SP is True

    def test_infer_bpe_sp_bert(self):
        class MockTokenizer:
            pass

        mock = MockTokenizer()
        mock.__class__.__name__ = "BertTokenizer"

        bp = BERTPreprocessor(mock)
        assert bp.is_BPE_SP is False
