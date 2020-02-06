from typing import List

import numpy as np

__author__ = "@de-code"

"""
Utility class, from: 
https://github.com/elifesciences/sciencebeam-trainer-delft/blob/develop/sciencebeam_trainer_delft/utils/numpy.py
"""

def concatenate_or_none(arrays: List[np.array], **kwargs) -> np.array:
    if arrays[0] is None:
        return None
    return np.concatenate(arrays, **kwargs)


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
        random_seed = np.random.randint(0, 2**(32 - 1) - 1)

    for arr in arrays:
        rstate = np.random.RandomState(random_seed)  # pylint: disable=no-member
        rstate.shuffle(arr)


def shuffle_pair_with_view(a, b=None):
    assert "Cannot shuffle with view if the two arrays have different dimensions: " + str(len(a)) + " vs " \
           + str(len(b)), len(a) == len(b)
    # generate permutation index array
    permutation = np.random.permutation(a.shape[0])
    # shuffle the two arrays
    if b is not None:
        return a[permutation], b[permutation]
    else:
        return a[permutation]
