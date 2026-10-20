"""
Training sequences shown with some of their words glued together, as the text of some
PDFs comes out of GROBID: "UngerR" for "Unger R", "syndromeBiochim" for the end of a
title and the start of a journal, "Acta1801201020921419948243" for a journal followed
by its volume, year and pages.

The GROBID training data has almost none of them (2% of the tokens of a document at
most), and on bibliographies where a fifth of the tokens are glued, the reference
segmenter fused the references: with no word embedding for most tokens, it had only
the characters of the words, which it had never had to rely on.
"""

import random
import re
from dataclasses import dataclass
from typing import Tuple

_YEAR = re.compile(r"[12][0-9]{3}")


@dataclass(frozen=True)
class GrobidColumns:
    """Where GROBID puts each feature of a token, as indices into its features
    (the columns of a training file after the token, the label excluded)."""

    lowercase: int
    prefixes: Tuple[int, ...]
    suffixes: Tuple[int, ...]
    line: int
    block: int
    capitalisation: int
    digits: int
    # single character and the dictionary lookups, false for a glued word but for the
    # year and http ones, which GROBID looks for inside the word
    flags: Tuple[int, ...]
    year: int
    http: int
    punctuation: int


# FeaturesVectorReferenceSegmenter of GROBID
REFERENCE_SEGMENTER_COLUMNS = GrobidColumns(
    lowercase=0,
    prefixes=(1, 2, 3, 4),
    suffixes=(5, 6, 7, 8),
    line=9,
    block=24,
    capitalisation=11,
    digits=12,
    flags=tuple(range(13, 21)),
    year=18,
    http=20,
    punctuation=21,
)


class GluedWords:
    """
    In each epoch, ``probability`` of the training sequences are shown with words
    glued: a word is glued to the next one, with probability ``pair_probability``,
    when both are made of letters and digits only, on the same line, and the next one
    does not start a field. A word can be glued to several following ones.

    A glued word takes the label of its first word, and the features GROBID gives
    such a word: those of its line and block from both words, its capitalisation,
    digits, prefixes and suffixes from the glued string, and no dictionary match.
    The other features (position in the line...) are those of its first word.
    """

    def __init__(self, probability: float, columns: GrobidColumns, pair_probability: float = 0.5):
        if not 0 <= probability <= 1 or not 0 <= pair_probability <= 1:
            raise ValueError(f"probabilities must be between 0 and 1, got {probability} and {pair_probability}")
        self.probability = probability
        self.pair_probability = pair_probability
        self.columns = columns

    def __call__(self, tokens, labels, features):
        # the python random of each worker of a DataLoader is seeded by torch
        if features is None or labels is None or random.random() >= self.probability:
            return tokens, labels, features

        glued_tokens, glued_labels, glued_features = [tokens[0]], [labels[0]], [list(features[0])]
        for token, label, token_features in zip(tokens[1:], labels[1:], features[1:]):
            if self._gluable(glued_tokens[-1], glued_labels[-1], glued_features[-1], token, label, token_features):
                glued_tokens[-1] = self._glue(glued_tokens[-1], glued_features[-1], token, token_features)
            else:
                glued_tokens.append(token)
                glued_labels.append(label)
                glued_features.append(list(token_features))
        return glued_tokens, glued_labels, glued_features

    def _gluable(self, token, label, token_features, next_token, next_label, next_features):
        line = self.columns.line
        return (
            token.isalnum()
            and next_token.isalnum()
            and token_features[line] != "LINEEND"
            and next_features[line] != "LINESTART"
            and (next_label == label == "O" or (next_label.startswith("I-") and next_label[2:] == label[2:]))
            and random.random() < self.pair_probability
        )

    def _glue(self, token, token_features, next_token, next_features):
        """The glued word, its features (those of ``token``) being changed in place."""
        columns = self.columns
        glued = token + next_token

        for column, end in ((columns.line, "LINEEND"), (columns.block, "BLOCKEND")):
            if next_features[column] == end:
                token_features[column] = end

        token_features[columns.lowercase] = glued.lower()
        for n, column in enumerate(columns.prefixes, start=1):
            token_features[column] = glued[:n]
        for n, column in enumerate(columns.suffixes, start=1):
            token_features[column] = glued[-n:]

        if glued.isdigit():
            token_features[columns.digits] = "ALLDIGIT"
        elif any(c.isdigit() for c in glued):
            token_features[columns.digits] = "CONTAINSDIGITS"
        else:
            token_features[columns.digits] = "NODIGIT"

        if glued.isdigit():
            token_features[columns.capitalisation] = "NOCAPS"
        elif glued == glued.upper():
            token_features[columns.capitalisation] = "ALLCAP"
        elif glued[0].isupper():
            token_features[columns.capitalisation] = "INITCAP"
        else:
            token_features[columns.capitalisation] = "NOCAPS"

        for column in columns.flags:
            token_features[column] = "0"
        token_features[columns.year] = "1" if _YEAR.search(glued) else "0"
        token_features[columns.http] = "1" if "http" in glued.lower() else "0"
        token_features[columns.punctuation] = "NOPUNCT"
        return glued
