"""
Tests for the text classification wrapper helpers.
"""

import numpy as np

from delft.textClassification.wrapper import split_train_validation


class TestSplitTrainValidation:
    def _class_ordered_corpus(self, n_per_class=50):
        """All of class 0 first, then all of class 1 - the worst case for a
        positional split."""
        texts = [f"class0-{i}" for i in range(n_per_class)] + [f"class1-{i}" for i in range(n_per_class)]
        labels = [[1.0, 0.0]] * n_per_class + [[0.0, 1.0]] * n_per_class
        return np.asarray(texts), np.asarray(labels)

    def test_validation_holds_both_classes_when_input_is_class_ordered(self):
        """
        Taking the last 10% without shuffling handed back a validation set made
        of a single class, which has no defined ROC-AUC.
        """
        np.random.seed(42)  # shuffle_triple_with_view draws from the global RNG
        x, y = self._class_ordered_corpus()

        _, _, _, y_valid = split_train_validation(x, y)

        present = set(np.argmax(y_valid, axis=1))
        assert present == {0, 1}, f"validation set covers only class(es) {present}"

    def test_split_sizes(self):
        x, y = self._class_ordered_corpus(n_per_class=50)

        x_train, y_train, x_valid, y_valid = split_train_validation(x, y)

        assert len(x_train) == len(y_train) == 90
        assert len(x_valid) == len(y_valid) == 10

    def test_ratio_is_configurable(self):
        x, y = self._class_ordered_corpus(n_per_class=50)

        x_train, _, x_valid, _ = split_train_validation(x, y, split_ratio=0.5)

        assert len(x_train) == 50
        assert len(x_valid) == 50

    def test_texts_stay_paired_with_their_labels(self):
        """Shuffling must apply one permutation to both arrays, not two."""
        x, y = self._class_ordered_corpus()

        x_train, y_train, x_valid, y_valid = split_train_validation(x, y)

        for texts, labels in ((x_train, y_train), (x_valid, y_valid)):
            for text, label in zip(texts, labels):
                expected = 0 if text.startswith("class0") else 1
                assert np.argmax(label) == expected, f"{text} lost its label"

    def test_accepts_plain_lists(self):
        texts = [f"doc-{i}" for i in range(10)]
        labels = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5

        x_train, y_train, x_valid, y_valid = split_train_validation(texts, labels)

        assert len(x_train) == 9
        assert len(x_valid) == 1
