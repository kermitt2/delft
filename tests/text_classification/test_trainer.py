"""
Tests for the text classification training metrics and checkpoint handling.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.trainer import Trainer, compute_roc_auc, restore_best_weights


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


class _TinyModel(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.fill_(value)
            self.linear.bias.fill_(value)


class _StubClassifier(nn.Module):
    """Minimal stand-in returning the dict shape the Trainer expects."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs, labels=None):
        logits = self.linear(inputs)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return {"loss": loss, "logits": logits}


def _make_trainer(tmp_path, use_roc_auc, max_epoch=1, patience=5):
    model_config = ModelConfig(model_name="test-monitor", architecture="gru")
    training_config = TrainingConfig(
        learning_rate=1e-3,
        max_epoch=max_epoch,
        patience=patience,
        use_roc_auc=use_roc_auc,
        early_stop=True,
    )
    return Trainer(
        _StubClassifier(),
        model_config,
        training_config,
        device="cpu",
        checkpoint_path=str(tmp_path),
    )


def _loader(n=8):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (torch.rand(n, 2) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestMonitoredMetric:
    """
    use_roc_auc used to be dead: the branch reading it was `pass`, and both
    checkpointing and early stopping tracked validation loss regardless.
    """

    def test_roc_auc_is_maximised_when_enabled(self, tmp_path):
        trainer = _make_trainer(tmp_path, use_roc_auc=True)

        assert trainer.monitor == "roc_auc"
        assert trainer.model_checkpoint.mode == "max"
        assert trainer.early_stopping.mode == "max"

    def test_loss_is_minimised_when_disabled(self, tmp_path):
        trainer = _make_trainer(tmp_path, use_roc_auc=False)

        assert trainer.monitor == "loss"
        assert trainer.model_checkpoint.mode == "min"
        assert trainer.early_stopping.mode == "min"

    def test_checkpoint_follows_roc_auc_not_loss(self, tmp_path, monkeypatch):
        """A rising ROC-AUC must keep checkpointing even as the loss worsens."""
        trainer = _make_trainer(tmp_path, use_roc_auc=True, max_epoch=2)
        scripted = iter([{"loss": 0.1, "roc_auc": 0.60}, {"loss": 0.5, "roc_auc": 0.90}])
        monkeypatch.setattr(trainer, "evaluate", lambda loader: next(scripted))

        loader = _loader()
        trainer.train(loader, valid_loader=loader)

        # tracking loss would have kept 0.1 and skipped the second epoch's save
        assert trainer.model_checkpoint.best_score == 0.90
        assert trainer.early_stopping.best_score == 0.90

    def test_checkpoint_follows_loss_when_roc_auc_disabled(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, use_roc_auc=False, max_epoch=2)
        scripted = iter([{"loss": 0.5, "roc_auc": 0.60}, {"loss": 0.1, "roc_auc": 0.20}])
        monkeypatch.setattr(trainer, "evaluate", lambda loader: next(scripted))

        loader = _loader()
        trainer.train(loader, valid_loader=loader)

        assert trainer.model_checkpoint.best_score == 0.1
        assert trainer.early_stopping.best_score == 0.1


class TestRestoreBestWeights:
    def test_best_weights_replace_the_last_epoch_weights(self, tmp_path):
        """
        The checkpoint holds the best epoch; the model in memory holds the last
        one. Without restoring, the wrapper saved the last epoch's weights -
        `patience` epochs past the best.
        """
        checkpoint = str(tmp_path / "best_model.pth")
        best = _TinyModel(value=1.0)
        torch.save(best.state_dict(), checkpoint)

        last = _TinyModel(value=9.0)  # what training ended on
        restored = restore_best_weights(last, checkpoint)

        assert restored is True
        assert torch.allclose(last.linear.weight, best.linear.weight)
        assert torch.allclose(last.linear.bias, best.linear.bias)

    def test_checkpoint_file_is_removed(self, tmp_path):
        checkpoint = str(tmp_path / "best_model.pth")
        torch.save(_TinyModel(value=1.0).state_dict(), checkpoint)

        restore_best_weights(_TinyModel(), checkpoint)

        assert not os.path.exists(checkpoint), "temporary checkpoint left behind"

    def test_missing_checkpoint_is_a_no_op(self, tmp_path):
        """No validation set, or no epoch ever improved."""
        model = _TinyModel(value=9.0)

        restored = restore_best_weights(model, str(tmp_path / "does-not-exist.pth"))

        assert restored is False
        assert torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 9.0))

    def test_ddp_wrapped_model_is_restored_through_module(self, tmp_path):
        """ModelCheckpoint saves module.state_dict() for wrapped models."""

        class _Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        checkpoint = str(tmp_path / "best_model.pth")
        torch.save(_TinyModel(value=1.0).state_dict(), checkpoint)

        wrapped = _Wrapper(_TinyModel(value=9.0))
        assert restore_best_weights(wrapped, checkpoint) is True
        assert torch.allclose(wrapped.module.linear.weight, torch.ones_like(wrapped.module.linear.weight))
