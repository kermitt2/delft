"""
Tests for the text classification training metrics.
"""

import numpy as np

from delft.textClassification.trainer import compute_roc_auc


class TestComputeRocAuc:
    def test_perfect_separation_scores_one(self):
        y_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])

        assert compute_roc_auc(y_true, y_pred) == 1.0

    def test_single_column_is_scored_directly(self):
        y_true = np.array([[0.0], [1.0], [0.0], [1.0]])
        y_pred = np.array([[0.1], [0.9], [0.2], [0.8]])

        assert compute_roc_auc(y_true, y_pred) == 1.0

    def test_degenerate_class_does_not_poison_the_whole_score(self):
        """
        A class holding a single label value is undefined for ROC-AUC. Scoring
        every class in one averaged roc_auc_score call turned that into NaN for
        the entire evaluation - sklearn warns and returns NaN rather than
        raising, so the ValueError guard around it never fired. Only the
        degenerate class should be affected.
        """
        # class 0 separates perfectly, class 1 is never positive
        y_true = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        y_pred = np.array([[0.9, 0.01], [0.1, 0.02], [0.8, 0.01], [0.2, 0.03]])

        score = compute_roc_auc(y_true, y_pred)

        assert np.isfinite(score), "a degenerate class must not turn the score into NaN"
        # class 0 contributes 1.0, class 1 contributes its clamped r2_score
        assert 0.5 <= score <= 1.0

    def test_all_classes_degenerate_stays_finite(self):
        y_true = np.zeros((4, 2))
        y_pred = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.2], [0.1, 0.3]])

        score = compute_roc_auc(y_true, y_pred)

        assert np.isfinite(score)
        assert score >= 0.0

    def test_no_classes_returns_zero(self):
        assert compute_roc_auc(np.zeros((4, 0)), np.zeros((4, 0))) == 0.0
