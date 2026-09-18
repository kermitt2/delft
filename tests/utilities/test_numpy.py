import numpy as np

from delft.utilities.numpy import concatenate_or_none

RAGGED = [["a", "b", "c"], ["d"]]


def _as_lists(array):
    return [list(item) for item in array]


class TestConcatenateOrNone:
    def test_nothing_gives_none(self):
        assert concatenate_or_none((None, None)) is None

    def test_a_missing_set_is_left_out(self):
        """Features without a validation set: this raised."""
        train = np.array(RAGGED, dtype=object)
        assert concatenate_or_none((train, None)) is train
        assert concatenate_or_none((None, train)) is train

    def test_arrays_are_concatenated_as_numpy_does(self):
        labels = concatenate_or_none([np.array([[1, 0], [0, 1]]), np.array([[1, 1]])])
        assert labels.shape == (3, 2) and labels.dtype != object

    def test_lists_of_sequences_of_different_lengths(self):
        """np.concatenate does not take them: 'inhomogeneous shape'."""
        both = concatenate_or_none((RAGGED, [["e", "f"]]))
        assert _as_lists(both) == RAGGED + [["e", "f"]]

    def test_a_set_of_sequences_of_the_same_length_with_another_one(self):
        """The first is a matrix, as the readers return it, the second is not."""
        same_length = np.array([["a", "b"], ["c", "d"]], dtype=object)
        assert same_length.ndim == 2
        both = concatenate_or_none((same_length, np.array(RAGGED, dtype=object)))
        assert _as_lists(both) == [["a", "b"], ["c", "d"]] + RAGGED
