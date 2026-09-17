"""
Sliding windows over training sequences.

A model takes at most ``max_sequence_length`` tokens, and a longer training
sequence is cut there: what follows the cut is never trained on, and the model
only ever sees sequences that start where a document starts. With a window
stride, a long sequence becomes several training examples instead, one window of
``max_sequence_length`` every ``stride`` tokens, so that all of it is trained on,
including windows that start and end in the middle of a labelled field, which is
what a model receives when its caller cuts long inputs before sending them.

Lengths are counted in the unit ``max_sequence_length`` is counted in: tokens for
the RNN architectures, sub-tokens for the transformer ones. The ``costs`` of a
sequence give the length of each of its tokens in that unit.
"""

from typing import Callable, List, Optional, Sequence, Tuple


def window_bounds(costs: Sequence[int], max_length: int, stride: int) -> List[Tuple[int, int]]:
    """
    The ``(start, end)`` token positions of the windows over one sequence.

    Every window holds as many tokens as fit in ``max_length``, the next one starts
    ``stride`` further, and together they cover the sequence with no gap. A token
    longer than ``max_length`` on its own gets a window to itself.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")
    if not 1 <= stride <= max_length:
        raise ValueError(f"stride must be between 1 and max_length ({max_length}), got {stride}")

    nb_tokens = len(costs)
    bounds = []
    start = 0
    covered = 0
    while True:
        end, length = start, 0
        while end < nb_tokens and length + costs[end] <= max_length:
            length += costs[end]
            end += 1
        end = max(end, min(start + 1, nb_tokens))

        if end > covered or not bounds:
            bounds.append((start, end))
            covered = end
        if end >= nb_tokens:
            return bounds

        next_start, moved = start, 0
        while next_start < end and moved < stride:
            moved += costs[next_start]
            next_start += 1
        start = max(next_start, start + 1)


def split_into_windows(
    x: Sequence,
    y: Optional[Sequence],
    features: Optional[Sequence],
    max_length: int,
    stride: int,
    token_costs: Optional[Callable[[Sequence[str]], Sequence[int]]] = None,
) -> Tuple[List, Optional[List], Optional[List]]:
    """
    Replace every sequence longer than ``max_length`` by its windows; labels and
    features are cut along with their tokens. Sequences that fit are kept as they are.

    ``token_costs`` gives the length of each token of a sequence (see
    ``subtoken_costs``); without it every token counts for one.
    """
    windowed_x = []
    windowed_y = None if y is None else []
    windowed_features = None if features is None else []

    for i, tokens in enumerate(x):
        costs = token_costs(tokens) if token_costs is not None else [1] * len(tokens)
        for start, end in window_bounds(costs, max_length, stride):
            windowed_x.append(tokens[start:end])
            if windowed_y is not None:
                windowed_y.append(y[i][start:end])
            if windowed_features is not None:
                windowed_features.append(features[i][start:end])

    return windowed_x, windowed_y, windowed_features


def subtoken_costs(tokenizer) -> Callable[[Sequence[str]], List[int]]:
    """How many sub-tokens ``tokenizer`` makes of each token of a sequence."""

    def costs(tokens: Sequence[str]) -> List[int]:
        counts = [0] * len(tokens)
        if len(tokens) == 0:
            return counts
        encoded = tokenizer(list(tokens), is_split_into_words=True, add_special_tokens=False, verbose=False)
        for word_id in encoded.word_ids():
            if word_id is not None:
                counts[word_id] += 1
        return counts

    return costs
