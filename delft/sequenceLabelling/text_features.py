"""
Text of a token taken from columns of its features.

In most GROBID models a position of a sequence is a token, and its text is the first
column of the training file. In the models that label lines (segmentation,
reference-segmenter), a position is a line, of which the file gives the first two
tokens in its first two columns: taking the text from the first column alone, a
model only reads the first word of each line.

With ``text_features_indices``, the text of a position is what the listed columns
hold, column 0 being the token, joined with a space: ``[0, 1]`` reads both tokens
of a line. The characters of the whole text are encoded, a transformer sub-tokenizes
the whole text, and the word embeddings of the listed columns are concatenated.
"""

from typing import List, Optional, Sequence

TEXT_SEPARATOR = " "


def text_from_features(
    x: Sequence[Sequence[str]], features: Optional[Sequence], text_features_indices: Optional[Sequence[int]]
) -> List[List[str]]:
    """
    The text of every position of every sequence, from the columns
    ``text_features_indices`` of its features. A column a row does not have counts
    as empty. Without indices, ``x`` is returned as it is.
    """
    if not text_features_indices:
        return x
    if features is None:
        raise ValueError(
            f"This model takes the text of a token from the columns {list(text_features_indices)} of its "
            "features, which were not given"
        )
    if len(features) != len(x):
        raise ValueError(f"{len(x)} sequences but features for {len(features)}")

    texts = []
    for tokens, rows in zip(x, features):
        if len(rows) != len(tokens):
            raise ValueError(f"A sequence of {len(tokens)} tokens has features for {len(rows)}")
        texts.append(
            [
                TEXT_SEPARATOR.join(str(row[index]) if index < len(row) else "" for index in text_features_indices)
                for row in rows
            ]
        )
    return texts


def tokens_per_position(text_features_indices: Optional[Sequence[int]]) -> int:
    """How many word embeddings are concatenated at each position."""
    return max(1, len(text_features_indices or ()))


def words_and_positions(texts: Sequence[str]):
    """
    The words of a sequence whose positions may hold several, and the position each
    word belongs to, for a sub-tokenizer that takes words. A position with no text
    keeps an empty word, so that it still has a place.
    """
    words, positions = [], []
    for position, text in enumerate(texts):
        for word in [word for word in text.split(TEXT_SEPARATOR) if word] or [""]:
            words.append(word)
            positions.append(position)
    return words, positions
