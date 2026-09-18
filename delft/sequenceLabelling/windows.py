"""
Sliding windows over the sequences of a training, validation or evaluation set,
and over the sequences to label.

A model takes at most ``max_sequence_length`` tokens, and a longer training
sequence is cut there: what follows the cut is never trained on, and the model
only ever sees sequences that start where a document starts. With a window
stride, a long sequence becomes several training examples instead, one window of
``max_sequence_length`` every ``stride`` tokens, so that all of it is trained on,
including windows that start and end in the middle of a labelled field, which is
what a model receives when its caller cuts long inputs before sending them.

To score a model on whole sequences, they are cut into windows side by side and
the predictions of the windows of a sequence are put back together. To label a
sequence longer than the model takes, it is cut into windows that may overlap, and
a position in an overlap is labelled by the window it is further from the edge of.

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
) -> Tuple[List, Optional[List], Optional[List], List[int]]:
    """
    Replace every sequence longer than ``max_length`` by its windows; labels and
    features are cut along with their tokens. Sequences that fit are kept as they are.
    Also returns how many windows each sequence made, which ``join_windows`` takes.

    ``token_costs`` gives the length of each token of a sequence (see
    ``subtoken_costs``); without it every token counts for one.
    """
    x, y, features, bounds = cut_into_windows(x, y, features, max_length, stride, token_costs=token_costs)
    return x, y, features, [len(sequence_bounds) for sequence_bounds in bounds]


def cut_into_windows(
    x: Sequence,
    y: Optional[Sequence],
    features: Optional[Sequence],
    max_length: int,
    stride: int,
    token_costs: Optional[Callable[[Sequence[str]], Sequence[int]]] = None,
) -> Tuple[List, Optional[List], Optional[List], List[List[Tuple[int, int]]]]:
    """
    ``split_into_windows``, returning the ``(start, end)`` positions of the windows of
    each sequence rather than how many there are, which ``join_overlapping_windows``
    takes to put windows that overlap back together.
    """
    bounds_per_sequence = []
    windowed_x = []
    windowed_y = None if y is None else []
    windowed_features = None if features is None else []

    for i, tokens in enumerate(x):
        costs = token_costs(tokens) if token_costs is not None else [1] * len(tokens)
        bounds = window_bounds(costs, max_length, stride)
        bounds_per_sequence.append(bounds)
        for start, end in bounds:
            windowed_x.append(tokens[start:end])
            if windowed_y is not None:
                windowed_y.append(y[i][start:end])
            if windowed_features is not None:
                windowed_features.append(features[i][start:end])

    return windowed_x, windowed_y, windowed_features, bounds_per_sequence


def join_windows(windows: Sequence[Sequence], window_counts: Sequence[int]) -> List[List]:
    """
    Put the windows of each sequence back end to end: the reverse of
    ``split_into_windows`` for windows cut side by side (a stride of ``max_length``),
    given in their original order.
    """
    if sum(window_counts) != len(windows):
        raise ValueError(f"{len(windows)} windows do not match the expected {sum(window_counts)}")
    joined = []
    position = 0
    for count in window_counts:
        joined.append([item for window in windows[position : position + count] for item in window])
        position += count
    return joined


def join_overlapping_windows(windows: Sequence[Sequence], bounds: Sequence[Sequence[Tuple[int, int]]]) -> List[List]:
    """
    Put the windows of each sequence back together, whatever their overlap: the
    reverse of ``cut_into_windows`` given the ``(start, end)`` positions of the windows
    of each sequence, the windows in their original order.

    Where two windows overlap, the position that is in the middle of the overlap
    changes hands: the positions before it are taken from the window that ends in the
    overlap, the others from the window that starts there. Each position is so taken
    from the window it is further from the edge of, where the window saw more of what
    surrounds it. Windows side by side are put end to end, as ``join_windows`` does.
    """
    if sum(len(sequence_bounds) for sequence_bounds in bounds) != len(windows):
        raise ValueError(f"{len(windows)} windows do not match the expected {sum(len(b) for b in bounds)}")

    joined = []
    position = 0
    for sequence_bounds in bounds:
        items = []
        for k, (start, end) in enumerate(sequence_bounds):
            window = windows[position + k]
            # where this window takes over from the previous one, and hands over to the next
            first = start if k == 0 else (start + sequence_bounds[k - 1][1]) // 2
            last = end if k == len(sequence_bounds) - 1 else (sequence_bounds[k + 1][0] + end) // 2
            if len(window) < end - start:
                raise ValueError(f"A window of positions {start} to {end} has {len(window)} items")
            items.extend(window[first - start : last - start])
        joined.append(items)
        position += len(sequence_bounds)
    return joined


def join_scored_windows(loader, *per_window: Sequence[Sequence]) -> Tuple[List, ...]:
    """
    Predictions and labels collected per example of ``loader``, put back per sequence
    when the loader cut its sequences into windows, so that a field which spans two
    windows is scored once and as a whole. Left as they are otherwise, including when
    only a part of the windows was seen (a distributed sampler).
    """
    window_counts = getattr(getattr(loader, "dataset", None), "window_counts", None)
    if not window_counts or any(sum(window_counts) != len(items) for items in per_window):
        return tuple(list(items) for items in per_window)
    return tuple(join_windows(items, window_counts) for items in per_window)


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
