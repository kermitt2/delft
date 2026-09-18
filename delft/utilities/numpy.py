from typing import List

import numpy as np

__author__ = "@de-code"

"""
Utility class, from:
https://github.com/elifesciences/sciencebeam-trainer-delft/blob/develop/sciencebeam_trainer_delft/utils/numpy.py
"""


def concatenate_or_none(arrays: List[np.array], **kwargs) -> np.array:
    """
    The sequences of several sets one after the other, None when there is none. A set
    may be missing (no validation set), and may be a list as well as an array: sequences
    have different lengths, which np.concatenate does not take from lists.
    """
    arrays = [array for array in arrays if array is not None]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    if all(isinstance(array, np.ndarray) for array in arrays):
        try:
            return np.concatenate(arrays, **kwargs)
        except ValueError:
            # a set whose sequences all have the same length is a matrix, another one is not
            pass
    concatenated = np.empty(sum(len(array) for array in arrays), dtype=object)
    position = 0
    for array in arrays:
        for item in array:
            concatenated[position] = item
            position += 1
    return concatenated


# https://stackoverflow.com/a/51526109/8676953
def shuffle_arrays(arrays: List[np.array], random_seed: int = None) -> List[np.array]:
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    random_seed : Seed value if not None, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** (32 - 1) - 1)

    for arr in arrays:
        rstate = np.random.RandomState(random_seed)  # pylint: disable=no-member
        rstate.shuffle(arr)


def shuffle_triple_with_view(a, b=None, c=None):
    """
    Shuffle a pair/triple with view, without modifying the initial elements.
    The assumption is that if b is None, c is also None.
    """
    if b is not None and c is not None:
        assert (
            "Cannot shuffle with view if the arrays have different dimensions: " + str(len(a)) + " vs " + str(len(b))
        ), len(a) == len(b) == len(c)
    elif b is not None and c is None:
        assert (
            "Cannot shuffle with view if the arrays have different dimensions: " + str(len(a)) + " vs " + str(len(b))
        ), len(a) == len(b)
    elif c is not None and b is None:
        raise Exception("b is None but c is not.")

    # generate permutation index array
    permutation = np.random.permutation(a.shape[0])

    # shuffle the arrays
    if c is not None and b is not None:
        return a[permutation], b[permutation], c[permutation]
    if b is not None and c is None:
        return a[permutation], b[permutation], None
    else:
        return a[permutation], None, None
