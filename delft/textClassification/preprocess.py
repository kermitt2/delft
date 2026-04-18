"""
Text classification preprocessing utilities for DeLFT.

This module contains preprocessing classes and functions for text classification.
"""

import json
import regex as re
import numpy as np

from unidecode import unidecode
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

# Import shared utilities
from delft.utilities.preprocess import (
    UNK,
    PAD,
    BasePreprocessor,
)

special_character_removal = re.compile(r"[^A-Za-z\.\-\?\!\,\#\@\% ]", re.IGNORECASE)


def to_vector_single(text, embeddings, maxlen=300):
    """
    Given a string, tokenize it, then convert it to a sequence of word embedding
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    tokens = tokenizeAndFilterSimple(clean_text(text))
    window = tokens[-maxlen:]

    # TBD: use better initializers (uniform, etc.)
    x = np.zeros(
        (maxlen, embeddings.embed_size),
    )

    # TBD: padding should be left and which vector do we use for padding?
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        x[i, :] = embeddings.get_word_vector(word).astype("float32")

    return x


def to_indices_single(text, vocab_word, maxlen=300):
    """
    Given a string, tokenize it, then convert it to a sequence of word indices.

    Args:
        text: input text string
        vocab_word: word to index mapping
        maxlen: maximum sequence length

    Returns:
        numpy array of word indices [maxlen]
    """
    tokens = tokenizeAndFilterSimple(clean_text(text))
    window = tokens[-maxlen:]

    x = np.zeros((maxlen,), dtype=np.int64)

    unk_idx = vocab_word.get(UNK, 1)
    for i, word in enumerate(window):
        x[i] = vocab_word.get(word, unk_idx)

    return x


def clean_text(text):
    x_ascii = unidecode(text)
    x_clean = special_character_removal.sub("", x_ascii)
    return x_clean


def lower(word):
    return word.lower()


def normalize_num(word):
    return re.sub(r"[0-9０１２３４５６７８９]", r"0", word)


def create_single_input_bert(text, maxlen=512, transformer_tokenizer=None):
    """
    Note: use batch method preferably for better performance
    """

    # TBD: exception if tokenizer is not valid/None
    encoded_tokens = transformer_tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=maxlen,
        padding="max_length",
    )
    # note: [CLS] and [SEP] are added by the tokenizer

    ids = encoded_tokens["input_ids"]
    masks = encoded_tokens["token_type_ids"]
    segments = encoded_tokens["attention_mask"]

    return ids, masks, segments


def create_batch_input_bert(texts, maxlen=512, transformer_tokenizer=None):
    # TBD: exception if tokenizer is not valid/None

    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    encoded_tokens = transformer_tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=maxlen,
        padding="max_length",
    )

    # note: special tokens like [CLS] and [SEP] are added by the tokenizer

    ids = encoded_tokens["input_ids"]
    masks = encoded_tokens["token_type_ids"]
    segments = encoded_tokens["attention_mask"]

    return ids, masks, segments


class TextPreprocessor(BasePreprocessor):
    """
    Preprocessor for text classification.

    Builds word vocabulary from training texts and converts texts to sequences
    of word indices. Also handles label encoding for multi-label classification.
    """

    def __init__(self, maxlen=300, lowercase=False):
        super().__init__()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_word = None
        self.vocab_label = None
        self.list_classes = None

    def fit(self, X, y=None):
        """
        Build vocabulary from training texts.

        Args:
            X: list of text strings
            y: optional labels (one-hot encoded or list of class names)

        Returns:
            self
        """
        # Build word vocabulary
        vocab = {PAD: 0, UNK: 1}
        word_counts = {}

        for text in X:
            tokens = tokenizeAndFilterSimple(clean_text(text))
            if self.lowercase:
                tokens = [t.lower() for t in tokens]
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency (most common first) and add to vocab
        sorted_words = sorted(word_counts.keys(), key=lambda w: -word_counts[w])
        for word in sorted_words:
            if word not in vocab:
                vocab[word] = len(vocab)

        self.vocab_word = vocab
        self.vocab_char = vocab  # Alias for compatibility with base class

        # Build label vocabulary if labels provided
        if y is not None:
            if isinstance(y, np.ndarray) and len(y.shape) == 2:
                # One-hot encoded - deduce classes from shape
                self.list_classes = list(range(y.shape[1]))
            elif isinstance(y, list) and len(y) > 0:
                if isinstance(y[0], (list, np.ndarray)):
                    # Multi-hot
                    self.list_classes = list(range(len(y[0])))
                else:
                    # List of class names/indices
                    unique_classes = sorted(set(y))
                    self.list_classes = unique_classes

        return self

    def transform(self, X, y=None):
        """
        Transform texts to sequences of word indices.

        Args:
            X: list of text strings
            y: optional labels

        Returns:
            tuple of (word_indices, labels) or just word_indices if y is None
        """
        word_indices = []
        for text in X:
            indices = to_indices_single(text, self.vocab_word, self.maxlen)
            word_indices.append(indices)

        word_indices = np.array(word_indices)

        if y is not None:
            # Ensure labels are numpy array
            labels = np.array(y) if not isinstance(y, np.ndarray) else y
            return word_indices, labels
        else:
            return word_indices

    def save(self, file_path: str):
        """Save preprocessor state to JSON file."""
        data = {
            "maxlen": self.maxlen,
            "lowercase": self.lowercase,
            "vocab_word": self.vocab_word,
            "list_classes": self.list_classes,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        """Load preprocessor state from JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        preprocessor = cls(
            maxlen=data.get("maxlen", 300),
            lowercase=data.get("lowercase", False),
        )
        preprocessor.vocab_word = data.get("vocab_word", {})
        preprocessor.vocab_char = preprocessor.vocab_word  # Alias
        preprocessor.list_classes = data.get("list_classes")

        return preprocessor

    def get_embedding_weights(self, embeddings):
        """
        Get embedding weight matrix for initializing nn.Embedding layer.

        Args:
            embeddings: Embeddings instance with get_word_vector method

        Returns:
            numpy array of shape [vocab_size, embed_size]
        """
        vocab_size = len(self.vocab_word)
        embed_size = embeddings.embed_size

        weights = np.zeros((vocab_size, embed_size), dtype=np.float32)

        for word, idx in self.vocab_word.items():
            if word in [PAD, UNK]:
                # Use random initialization for special tokens
                weights[idx] = np.random.uniform(-0.25, 0.25, embed_size)
            else:
                weights[idx] = embeddings.get_word_vector(word)

        return weights
