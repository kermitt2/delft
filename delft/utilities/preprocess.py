"""
Shared preprocessing utilities for DeLFT.

This module contains common preprocessing classes and functions used by both
sequenceLabelling and textClassification modules.
"""

import logging
import re
from typing import Iterable, Dict
from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = logging.getLogger(__name__)

# Special tokens
UNK = "<UNK>"
PAD = "<PAD>"

# Casing index for character casing features
case_index = {
    "<PAD>": 0,
    "numeric": 1,
    "allLower": 2,
    "allUpper": 3,
    "initialUpper": 4,
    "other": 5,
    "mainly_numeric": 6,
    "contains_digit": 7,
}


# =============================================================================
# Utility Functions
# =============================================================================


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Pad sequences to a fixed length.

    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.
        max_length: the maximum length to pad to.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok=0, nlevels=1, max_char_length=30):
    """
    Pad sequences to a fixed length.

    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.
        nlevels: 1 for word-level, 2 for character-level (nested).
        max_char_length: maximum character length per word.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(
            sequences, pad_tok, max_length
        )
    elif nlevels == 2:
        max_length_word = max_char_length
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(
            sequence_padded, [pad_tok] * max_length_word, max_length_sentence
        )
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError("nlevels can take 1 or 2, not take {}.".format(nlevels))

    return sequence_padded, sequence_length


def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """
    Convert class labels from scalars to one-hot vectors.

    Args:
        labels_dense: a dense set of values
        num_classes: the number of classes in output
        nlevels: the number of levels (1 for flat, 2 for sequences)
    """
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError("nlevels can take 1 or 2, not take {}.".format(nlevels))


def get_casing(word):
    """
    Determine the casing category of a word.

    Args:
        word: input word string

    Returns:
        integer index from case_index
    """
    casing = "other"

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
    digitFraction = numDigits / float(len(word)) if len(word) > 0 else 0

    if word.isdigit():
        casing = "numeric"
    elif digitFraction > 0.5:
        casing = "mainly_numeric"
    elif word.islower():
        casing = "allLower"
    elif word.isupper():
        casing = "allUpper"
    elif len(word) > 0 and word[0].isupper():
        casing = "initialUpper"
    elif numDigits > 0:
        casing = "contains_digit"

    return case_index[casing]


def lower(word):
    """Convert word to lowercase."""
    return word.lower()


def normalize_num(word):
    """Normalize numeric characters to zeros."""
    return re.sub(r"[0-9０１２３４５６７８９]", r"0", word)


# =============================================================================
# Feature Preprocessing Functions
# =============================================================================


def calculate_cardinality(feature_vector, indices=None):
    """
    Calculate cardinality of each feature.

    Args:
        feature_vector: three dimensional vector with features
        indices: list of indices of the features to be extracted

    Returns:
        a map where each key is the index of the feature and the value is a map
        feature_value -> value_index.

    NOTE: the features are indexed from 1 to n + 1. The 0 value is reserved as padding
    """
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    for index_column in range(index, len(feature_vector[0][0])):
        if indices and index_column not in indices:
            index += 1
            continue

        values = set()
        for index_document in range(0, len(feature_vector)):
            for index_row in range(0, len(feature_vector[index_document])):
                value = feature_vector[index_document][index_row][index_column]
                if value != " ":
                    values.add(value)

        values = sorted(values)
        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            # We reserve the 0 for the unseen features so the indexes will go from 1 to cardinality + 1
            values_to_int[values_list[val_num]] = val_num + 1

        columns_length.append((index, values_to_int))
        index += 1

    return columns_length


def cardinality_to_index_map(columns_length, features_max_vector_size):
    """Convert cardinality tuples to index map."""
    # Filter out the columns that are not fitting
    columns_index = []
    for index, column_content_cardinality in columns_length:
        if len(column_content_cardinality) <= features_max_vector_size:
            columns_index.append((index, column_content_cardinality))

    index_list = [ind[0] for ind in columns_index if ind[0] >= 0]
    val_to_int_map = {
        value[0]: {
            val_features: idx_features + (index * features_max_vector_size)
            for val_features, idx_features in value[1].items()
        }
        for index, value in enumerate(columns_index)
    }

    return index_list, val_to_int_map


def reduce_features_to_indexes(feature_vector, features_max_vector_size, indices=None):
    """Reduce features to index mapping."""
    cardinality = calculate_cardinality(feature_vector, indices=indices)
    index_list, map_to_integers = cardinality_to_index_map(
        cardinality, features_max_vector_size
    )
    return index_list, map_to_integers


# =============================================================================
# Base Preprocessor Class
# =============================================================================


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for DeLFT preprocessors.

    Provides common interface for both sequence labeling and text classification
    preprocessing.
    """

    def __init__(self):
        self.vocab_char = None
        self.vocab_tag = None
        self.indice_tag = None

    @abstractmethod
    def fit(self, X, y):
        """
        Build vocabulary from training data.

        Args:
            X: input data (texts or token sequences)
            y: labels

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """
        Transform data to indices.

        Args:
            X: input data
            y: optional labels

        Returns:
            transformed data
        """
        pass

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)

    @abstractmethod
    def save(self, file_path: str):
        """Save preprocessor state to file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str):
        """Load preprocessor state from file."""
        pass

    def inverse_transform(self, y):
        """
        Convert label indices back to original labels.

        Args:
            y: label indices

        Returns:
            original labels
        """
        if self.indice_tag is None:
            self.indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [self.indice_tag[y_] for y_ in y]


# =============================================================================
# Features Preprocessor
# =============================================================================


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for discrete features (layout, categorical, etc.).

    Builds vocabulary for feature values and converts them to indices.
    """

    DEFAULT_FEATURES_VOCABULARY_SIZE = 12

    def __init__(
        self,
        features_indices: Iterable[int] = None,
        features_vocabulary_size: int = DEFAULT_FEATURES_VOCABULARY_SIZE,
        features_map_to_index: Dict = None,
    ):
        if features_map_to_index is None:
            features_map_to_index = {}
        self.features_vocabulary_size = features_vocabulary_size
        self.features_indices = features_indices
        self.features_map_to_index = features_map_to_index

    def fit(self, X):
        """Build feature vocabulary from data."""
        if not self.features_indices:
            indexes, mapping = reduce_features_to_indexes(
                X, self.features_vocabulary_size
            )
        else:
            indexes, mapping = reduce_features_to_indexes(
                X, self.features_vocabulary_size, indices=self.features_indices
            )

        self.features_map_to_index = mapping
        self.features_indices = indexes
        return self

    def transform(self, X, extend=False):
        """
        Transform features into index vector.

        Args:
            X: feature data
            extend: when True, adds an additional empty feature list in the sequence

        Returns:
            numpy array of feature indices
        """
        features_vector = [
            [
                [
                    self.features_map_to_index[index][value]
                    if index in self.features_map_to_index
                    and value in self.features_map_to_index[index]
                    else 0
                    for index, value in enumerate(value_list)
                    if index in self.features_indices
                ]
                for value_list in document
            ]
            for document in X
        ]

        features_count = len(self.features_indices)

        if extend:
            for out in features_vector:
                out.append([0] * features_count)

        features_vector_padded, _ = pad_sequences(features_vector, [0] * features_count)
        output = np.asarray(features_vector_padded)

        return output

    def empty_features_vector(self) -> Iterable[int]:
        """Return an empty features vector for padding."""
        features_count = len(self.features_indices) if self.features_indices else 0
        return [0] * features_count


# =============================================================================
# BERT Preprocessor
# =============================================================================


class BERTPreprocessor(object):
    """
    Generic transformer preprocessor for sequence data.

    Handles sub-tokenization and alignment of features/labels with transformer
    tokenizers from HuggingFace.
    """

    def __init__(self, tokenizer, empty_features_vector=None, empty_char_vector=None):
        self.tokenizer = tokenizer
        self.empty_features_vector = empty_features_vector
        self.empty_char_vector = empty_char_vector

        tokenizer_name = type(self.tokenizer).__name__

        # Flag to indicate if the tokenizer is a BPE or sentence piece tokenizer
        self.is_BPE_SP = self.infer_BPE_SP_from_tokenizer_name(tokenizer_name)

    def infer_BPE_SP_from_tokenizer_name(self, tokenizer_name):
        """
        Return True if the tokenizer is a BPE using encoded leading character space.
        """
        result = False
        bpe_sp_tokenizers = [
            "roberta",
            "gpt2",
            "albert",
            "xlnet",
            "marian",
            "t5",
            "camembert",
            "bart",
            "bigbird",
            "blenderbot",
            "clip",
            "flaubert",
            "fsmt",
            "xlm",
            "longformer",
            "phobert",
            "reformer",
            "rembert",
        ]
        tokenizer_name = tokenizer_name.lower()
        for bpe_tok in bpe_sp_tokenizers:
            if tokenizer_name.find(bpe_tok) != -1:
                result = True
                break
        return result

    def tokenize_and_align_features_and_labels(
        self, texts, chars, text_features, text_labels, maxlen=512
    ):
        """
        Sub-tokenize and align features/labels with new tokens.

        Args:
            texts: list of pre-tokenized texts
            chars: character representations
            text_features: features for each token
            text_labels: labels for each token
            maxlen: maximum sequence length

        Returns:
            Tuple of aligned inputs
        """
        target_ids = []
        target_type_ids = []
        target_attention_mask = []
        input_tokens = []
        target_chars = []

        target_features = None
        if text_features is not None:
            target_features = []

        target_labels = None
        if text_labels is not None:
            target_labels = []

        for i, text in enumerate(texts):
            local_chars = chars[i]

            features = None
            if text_features is not None:
                features = text_features[i]

            label_list = None
            if text_labels is not None:
                label_list = text_labels[i]

            (
                input_ids,
                token_type_ids,
                attention_mask,
                chars_block,
                feature_blocks,
                target_tags,
                tokens,
            ) = self.convert_single_text(
                text, local_chars, features, label_list, maxlen
            )
            target_ids.append(input_ids)
            target_type_ids.append(token_type_ids)
            target_attention_mask.append(attention_mask)
            input_tokens.append(tokens)
            target_chars.append(chars_block)

            if target_features is not None:
                target_features.append(feature_blocks)

            if target_labels is not None:
                target_labels.append(target_tags)

        return (
            target_ids,
            target_type_ids,
            target_attention_mask,
            target_chars,
            target_features,
            target_labels,
            input_tokens,
        )

    def convert_single_text(
        self, text_tokens, chars_tokens, features_tokens, label_tokens, max_seq_length
    ):
        """
        Convert a single sequence input into transformer format.

        Aligns other channel input to the new sub-tokenization.
        """
        if label_tokens is None:
            label_tokens = [0] * len(text_tokens)

        if features_tokens is None:
            features_tokens = [self.empty_features_vector] * len(text_tokens)

        if chars_tokens is None:
            chars_tokens = [self.empty_char_vector] * len(text_tokens)

        # Sub-tokenization
        encoded_result = self.tokenizer(
            text_tokens,
            add_special_tokens=True,
            is_split_into_words=True,
            max_length=max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded_result.input_ids
        offsets = encoded_result.offset_mapping
        if "token_type_ids" in encoded_result:
            token_type_ids = encoded_result.token_type_ids
        else:
            token_type_ids = [0] * len(input_ids)
        attention_mask = encoded_result.attention_mask
        label_ids = []
        chars_blocks = []
        feature_blocks = []

        # Handle BPE/sentence piece tokenizers
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        new_offsets = []
        empty_token = False

        for i in range(0, len(input_ids)):
            offset = offsets[i]
            if len(self.tokenizer.decode(input_ids[i])) == 0:
                empty_token = True
                continue
            elif (
                self.is_BPE_SP
                and self.tokenizer.convert_ids_to_tokens(input_ids[i])
                not in self.tokenizer.all_special_tokens
                and offset[0] == 0
                and len(self.tokenizer.convert_ids_to_tokens(input_ids[i])) == 1
                and not empty_token
            ):
                empty_token = False
                continue
            elif (
                self.is_BPE_SP
                and self.tokenizer.convert_ids_to_tokens(input_ids[i])
                not in self.tokenizer.all_special_tokens
                and offset[0] == 0
                and not empty_token
                and not (
                    self.tokenizer.convert_ids_to_tokens(input_ids[i]).startswith("Ġ")
                    or self.tokenizer.convert_ids_to_tokens(input_ids[i]).startswith(
                        "▁"
                    )
                    or self.tokenizer.convert_ids_to_tokens(input_ids[i]).startswith(
                        " "
                    )
                )
            ):
                empty_token = False
                continue
            else:
                empty_token = False
                new_input_ids.append(input_ids[i])
                new_attention_mask.append(attention_mask[i])
                new_token_type_ids.append(token_type_ids[i])
                new_offsets.append(offsets[i])

        input_ids = new_input_ids
        attention_mask = new_attention_mask
        token_type_ids = new_token_type_ids
        offsets = new_offsets

        word_idx = -1
        for i, offset in enumerate(offsets):
            if offset[0] == 0 and offset[1] == 0:
                # Special token
                label_ids.append("<PAD>")
                chars_blocks.append(self.empty_char_vector)
                feature_blocks.append(self.empty_features_vector)
            else:
                if offset[0] == 0:
                    word_idx += 1
                    label_ids.append(label_tokens[word_idx])
                    feature_blocks.append(features_tokens[word_idx])
                    chars_blocks.append(chars_tokens[word_idx])
                else:
                    # Sub-token
                    label_ids.append("<PAD>")
                    chars_blocks.append(self.empty_char_vector)
                    feature_blocks.append(features_tokens[word_idx])

        # Pad to max sequence length
        while len(input_ids) < max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            token_type_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            label_ids.append("<PAD>")
            chars_blocks.append(self.empty_char_vector)
            feature_blocks.append(self.empty_features_vector)

        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(chars_blocks) == max_seq_length
        assert len(feature_blocks) == max_seq_length

        return (
            input_ids,
            token_type_ids,
            attention_mask,
            chars_blocks,
            feature_blocks,
            label_ids,
            offsets,
        )
