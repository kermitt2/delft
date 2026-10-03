"""
Sub-tokenization of a sequence as one text rather than word by word (issue #128).

A transformer is given pre-tokenized sequences. Sub-tokenizing them with
``is_split_into_words=True``, every word starts as if a space came before it: a
SentencePiece or byte-level BPE tokenizer marks each one with its leading-space symbol,
even for a comma or a closing bracket that no space precedes in real text, which is
not what the transformer was pretrained on. With ``whole_text_tokenization``, the
words are joined back into a text with the usual spacing of punctuation, the text is
sub-tokenized as a whole, and every sub-token is aligned on the word its characters
belong to with the offsets the tokenizer returns.

A sub-token may then span several words, ``50%)`` tokenized as ``50`` and ``%)`` for
the words ``50``, ``%`` and ``)``: the words it starts are all counted on it, so that
each of them gets a label when tagging (the one predicted for that sub-token), and it is
trained with the label of the first one.
"""

from bisect import bisect_right
from typing import List, Optional, Sequence, Tuple

from delft.utilities.transformer_tokenizers import call_tokenizer

# no space before a closing or a trailing punctuation, none after an opening one
NO_SPACE_BEFORE = frozenset(",.;:!?)]}»%’”'\"")
NO_SPACE_AFTER = frozenset("([{«‘“")


def join_words(words: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    The text of the ``words`` of a sequence, with a space between two words except
    before a closing or a trailing punctuation and after an opening one, and the
    ``(start, end)`` character span of every word in it. An empty word has an empty
    span. A straight double quote opens and closes in turn.

    Two punctuation marks in a row are kept apart, "J. ," rather than "J.,": glued,
    a byte-level BPE tokenizer makes one sub-token of ".,", which the two words would
    have to share, with its label (see the module). A word is never glued to a mark on
    both sides that way, so a sub-token can only span words when the tokenizer joins
    a word and a mark, as "50%", which is rare.
    """
    pieces = []
    spans = []
    position = 0
    previous = None
    previous_opens = False
    quote_open = False
    for word in words:
        word = "" if word is None else str(word)
        if not word:
            spans.append((position, position))
            continue
        opens = word[-1] in NO_SPACE_AFTER
        closes = word[0] in NO_SPACE_BEFORE
        if word == '"':
            opens, closes = not quote_open, quote_open
            quote_open = not quote_open
        if pieces:
            spaced = not closes and not previous_opens
            if not word[0].isalnum() and not previous[-1].isalnum():
                spaced = True
            if spaced:
                pieces.append(" ")
                position += 1
        pieces.append(word)
        spans.append((position, position + len(word)))
        position += len(word)
        previous = word
        previous_opens = opens
    return "".join(pieces), spans


def align_offsets(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]], special: Sequence[bool]
) -> Tuple[List[Optional[int]], List[List[int]]]:
    """
    For every sub-token, given its ``(start, end)`` character offsets in the text: the
    word it belongs to, or ``None`` for a special token or a sub-token made of
    whitespace alone, and the words that start on it, in their order. A word that no
    sub-token starts on, one of a sub-token that spans it and the word before, or one
    the tokenizer dropped, is counted on the first sub-token ending after it begins:
    every word up to the last sub-token is started on one of them.

    ``spans`` are the character spans of the words, in their order, and ``special``
    says which sub-tokens are special tokens.
    """
    starts = [start for start, _ in spans]
    words_of_subtokens: List[Optional[int]] = []
    started: List[List[int]] = []
    next_word = 0  # the first word not started yet
    for (start, end), is_special in zip(offsets, special):
        if is_special or end <= start:
            words_of_subtokens.append(None)
            started.append([])
            continue
        # the last word starting at or before the sub-token, else the first one after
        word = bisect_right(starts, start) - 1
        if word < 0 or spans[word][1] <= start:
            word += 1
        if word >= len(spans) or spans[word][0] >= end:
            # whitespace between two words, or after the last one
            words_of_subtokens.append(None)
            started.append([])
            continue
        words_of_subtokens.append(word)
        # the words beginning before the end of the sub-token that no sub-token started
        # yet: its own, and any the tokenizer gave no sub-token of its own
        begun = []
        while next_word < len(spans) and spans[next_word][0] < end:
            begun.append(next_word)
            next_word += 1
        started.append(begun)
    return words_of_subtokens, started


def subtokenize_whole_text(tokenizer, words: Sequence[str], max_length: Optional[int] = None, add_special_tokens=True):
    """
    Sub-tokenize the ``words`` joined into a text. Returns the encoding of the
    tokenizer (with its ``offset_mapping``), the word of every sub-token and the words
    started on each, as ``align_offsets`` gives them.
    """
    text, spans = join_words(words)
    arguments = {"add_special_tokens": add_special_tokens, "return_offsets_mapping": True}
    if max_length:
        arguments.update(max_length=max_length, truncation=True)
    # one call at a time: the tokenizer is shared by the threads
    encoded = call_tokenizer(tokenizer, text, **arguments)
    special_ids = set(tokenizer.all_special_ids)
    special = [input_id in special_ids for input_id in encoded.input_ids]
    words_of_subtokens, started = align_offsets(encoded.offset_mapping, spans, special)
    return encoded, words_of_subtokens, started
