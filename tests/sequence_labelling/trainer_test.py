"""The training loop does what its configuration says."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from delft.sequenceLabelling.trainer import unique_checkpoint_path
from delft.sequenceLabelling.wrapper import Sequence, summarize_fold_scores
from delft.utilities.Utilities import set_random_seed

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
X = [WORDS, WORDS[:3], WORDS[2:], WORDS[:5]]
Y = [LABELS, LABELS[:3], LABELS[2:], LABELS[:5]]
FEATURES = [[[word, "UP" if word[0].isupper() else "LOW"] for word in sequence] for sequence in X]
# as the readers return them
X_ARRAY, Y_ARRAY, FEATURES_ARRAY = (np.array(item, dtype=object) for item in (X, Y, FEATURES))


def _sequence(tmp_path, monkeypatch, architecture="BidLSTM_CRF", **kwargs):
    monkeypatch.chdir(tmp_path)
    options = {"max_epoch": 2, "batch_size": 2, "early_stop": False, "nb_workers": 0, "device": "cpu"}
    options.update(kwargs)
    return Sequence("test-model", architecture=architecture, embeddings_name=None, **options)


def _train(sequence, **kwargs):
    sequence.train(X_ARRAY, Y_ARRAY, x_valid=X_ARRAY, y_valid=Y_ARRAY, **kwargs)


class TestConfiguration:
    def test_gradients_are_clipped_at_the_value_of_the_configuration(self, tmp_path, monkeypatch):
        sequence = _sequence(tmp_path, monkeypatch, clip_gradients=3.5)
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            _train(sequence)
        assert {call.args[1] for call in clip.call_args_list} == {3.5}

    def test_no_clipping_when_the_configuration_says_so(self, tmp_path, monkeypatch):
        sequence = _sequence(tmp_path, monkeypatch, clip_gradients=0)
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            _train(sequence)
        clip.assert_not_called()

    def test_the_learning_rate_decays_by_the_factor_of_the_configuration(self, tmp_path, monkeypatch):
        sequence = _sequence(tmp_path, monkeypatch, lr_decay=0.25)
        with patch("delft.sequenceLabelling.trainer.ReduceLROnPlateau") as scheduler:
            _train(sequence)
        assert scheduler.call_args.kwargs["factor"] == 0.25

    def test_defaults_are_what_the_training_has_been_doing(self, tmp_path, monkeypatch):
        """Clipping was fixed at 1.0 and the decay at 0.5, whatever the configuration said."""
        sequence = _sequence(tmp_path, monkeypatch)
        assert sequence.training_config.clip_gradients == 1.0
        assert sequence.training_config.lr_decay == 0.5


class TestCallbacks:
    def test_called_at_the_end_of_every_epoch_with_its_metrics(self, tmp_path, monkeypatch):
        calls = []
        _train(_sequence(tmp_path, monkeypatch), callbacks=[lambda epoch, logs: calls.append((epoch, sorted(logs)))])
        expected = ["f1", "learning_rate", "loss", "precision", "recall", "val_loss"]
        assert calls == [(1, expected), (2, expected)]

    def test_without_a_validation_set_the_loss_is_all_there_is(self, tmp_path, monkeypatch):
        calls = []
        sequence = _sequence(tmp_path, monkeypatch)
        sequence.train(X_ARRAY, Y_ARRAY, callbacks=[lambda epoch, logs: calls.append((epoch, sorted(logs)))])
        assert calls == [(1, ["loss"]), (2, ["loss"])]


def _checkpoint_files(tmp_path):
    directory = tmp_path / "data" / "models" / "sequenceLabelling" / "test-model"
    return sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []


class TestCheckpoints:
    def test_nothing_is_left_behind_by_default(self, tmp_path, monkeypatch):
        _train(_sequence(tmp_path, monkeypatch))
        assert _checkpoint_files(tmp_path) == []

    def test_nothing_is_left_behind_by_a_training_that_fails(self, tmp_path, monkeypatch):
        def fail(epoch, logs):
            raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            _train(_sequence(tmp_path, monkeypatch), callbacks=[fail])
        assert _checkpoint_files(tmp_path) == []

    def test_the_weights_of_the_last_epochs_are_kept_when_asked(self, tmp_path, monkeypatch):
        _train(_sequence(tmp_path, monkeypatch, max_epoch=4, max_checkpoints_to_keep=2))
        assert _checkpoint_files(tmp_path) == ["test-model-epoch3.pt", "test-model-epoch4.pt"]

    def test_every_training_has_a_file_of_its_own(self, tmp_path):
        """Two trainings of a same model in a same directory shared one, named after the model."""
        first = unique_checkpoint_path(str(tmp_path), "grobid-header-BidLSTM_CRF", "model_weights.pt")
        second = unique_checkpoint_path(str(tmp_path), "grobid-header-BidLSTM_CRF", "model_weights.pt")
        assert first != second
        assert os.path.dirname(first) == str(tmp_path) and os.path.basename(first).startswith("grobid-header-")

    def test_the_best_weights_are_the_ones_of_this_training(self, tmp_path, monkeypatch):
        """A stale file named after the model, left by another training, is not loaded."""
        sequence = _sequence(tmp_path, monkeypatch)
        directory = tmp_path / "data" / "models" / "sequenceLabelling" / "test-model"
        directory.mkdir(parents=True)
        stale = directory / "test-model_model_weights.pt"
        torch.save({"not": "the weights of this model"}, stale)
        _train(sequence)
        assert _checkpoint_files(tmp_path) == [stale.name]

    def test_the_best_epoch_is_restored(self, tmp_path, monkeypatch):
        sequence = _sequence(tmp_path, monkeypatch, max_epoch=3)
        states = []

        def f1_by_epoch(loader):
            states.append({name: tensor.clone() for name, tensor in sequence.model.state_dict().items()})
            return {"f1": [0.5, 0.9, 0.1][len(states) - 1], "precision": 0.0, "recall": 0.0, "loss": 0.0}

        with patch("delft.sequenceLabelling.trainer.Trainer.evaluate", side_effect=f1_by_epoch):
            _train(sequence)
        restored = sequence.model.state_dict()
        assert all(torch.equal(restored[name], states[1][name]) for name in restored)


class TestDataSets:
    def test_features_without_a_validation_set(self, tmp_path, monkeypatch):
        """This raised when putting the training and the missing validation features together."""
        sequence = _sequence(tmp_path, monkeypatch, "BidLSTM_CRF_FEATURES")
        sequence.train(X_ARRAY, Y_ARRAY, f_train=FEATURES_ARRAY)
        assert sequence.p.feature_preprocessor.features_indices == [0, 1]

    def test_sets_given_as_lists(self, tmp_path, monkeypatch):
        """Sequences have different lengths, which np.concatenate does not take from lists."""
        sequence = _sequence(tmp_path, monkeypatch, "BidLSTM_CRF_FEATURES")
        sequence.train(X, Y, f_train=FEATURES, x_valid=X, y_valid=Y, f_valid=FEATURES)

    def test_every_fold_is_trained_with_its_features(self, tmp_path, monkeypatch):
        """The loaders of train_nfold were not given the features: a FEATURES model had no input for them."""
        sequence = _sequence(tmp_path, monkeypatch, "BidLSTM_CRF_FEATURES", fold_number=2, max_epoch=1)
        sequence.train_nfold(X_ARRAY, Y_ARRAY, f_train=FEATURES_ARRAY)
        assert len(sequence.models) == 2


class TestNFoldEvaluation:
    def test_the_scores_of_every_fold_are_summarized_with_their_mean_and_std(self):
        scores = [
            {"precision": 0.5, "recall": 0.5, "f1": 0.5},
            {"precision": 0.7, "recall": 0.9, "f1": 0.8},
            {"precision": 0.6, "recall": 0.4, "f1": 0.5},
        ]
        summary = summarize_fold_scores(scores)
        assert summary["folds"] == scores
        assert summary["mean"] == pytest.approx({"precision": 0.6, "recall": 0.6, "f1": 0.6})
        assert summary["std"] == pytest.approx({"precision": 0.0816497, "recall": 0.2160247, "f1": 0.1414214})
        assert summary["best_fold"] == 1

    def test_eval_reports_the_mean_and_the_std_over_the_folds(self, tmp_path, monkeypatch, capsys):
        """eval_nfold gave the best and the average f1 alone, see issue #18."""
        sequence = _sequence(tmp_path, monkeypatch, fold_number=2, max_epoch=1)
        sequence.train_nfold(X_ARRAY, Y_ARRAY)
        summary = sequence.eval(X_ARRAY, Y_ARRAY)
        assert len(summary["folds"]) == 2
        assert set(summary["mean"]) == set(summary["std"]) == {"precision", "recall", "f1"}
        assert summary["std"]["f1"] >= 0
        assert sequence.model is sequence.models[summary["best_fold"]]
        assert "std" in capsys.readouterr().out


class TestSeed:
    @staticmethod
    def _weights(tmp_path, monkeypatch, seed):
        set_random_seed(seed)
        sequence = _sequence(tmp_path, monkeypatch, max_epoch=1)
        _train(sequence)
        return sequence.model.state_dict()

    def test_two_trainings_with_the_same_seed_are_the_same(self, tmp_path, monkeypatch):
        first, second = (self._weights(tmp_path, monkeypatch, 7) for _ in range(2))
        assert all(torch.equal(first[name], second[name]) for name in first)

    def test_another_seed_gives_another_training(self, tmp_path, monkeypatch):
        first, second = (self._weights(tmp_path, monkeypatch, seed) for seed in (7, 8))
        assert not all(torch.equal(first[name], second[name]) for name in first)

    def test_no_seed_leaves_the_generators_alone(self):
        torch.manual_seed(1)
        expected = torch.rand(1)
        torch.manual_seed(1)
        set_random_seed(None)
        assert torch.equal(torch.rand(1), expected)
