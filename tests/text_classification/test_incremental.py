"""
Incremental training of a text classifier: the training goes on from a loaded model.
The wrapper took the argument and ignored it, and the license application had no option
for it, which the cluster submitter passed all the same.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from delft.applications import licenseClassifier
from delft.textClassification.wrapper import Classifier

TEXTS = ["good work", "bad work", "good good work", "bad bad work", "work good", "work bad", "good", "bad"]
CLASSES = np.array([[1, 0], [0, 1]] * 4, dtype=np.float32)


class _Embeddings:
    """Word vectors that need no download: a word always gets the same random vector."""

    embed_size = 300

    def get_word_vector(self, word):
        return np.random.RandomState(sum(word.encode())).randn(self.embed_size).astype("float32")


def _classifier(**kwargs):
    options = dict(
        architecture="gru",
        list_classes=["a", "b"],
        maxlen=8,
        max_epoch=1,
        batch_size=4,
        early_stop=False,
        nb_workers=0,
        device="cpu",
        embeddings_name=None,
    )
    options.update(kwargs)
    classifier = Classifier("test-classifier", **options)
    classifier.embeddings = _Embeddings()
    return classifier


def _saved(tmp_path, monkeypatch):
    """A classifier trained and saved under ``tmp_path``, and its weights."""
    monkeypatch.chdir(tmp_path)
    classifier = _classifier()
    classifier.train(TEXTS, CLASSES)
    classifier.save(str(tmp_path))
    return {name: tensor.clone() for name, tensor in classifier.model.state_dict().items()}


def _loaded(tmp_path, **kwargs):
    classifier = _classifier(**kwargs)
    classifier.load(str(tmp_path))
    classifier.embeddings = _Embeddings()
    return classifier


def _same(state, other):
    return all(torch.equal(state[name], other[name]) for name in state)


class TestClassifier:
    def test_the_training_goes_on_from_the_loaded_model(self, tmp_path, monkeypatch):
        saved = _saved(tmp_path, monkeypatch)
        classifier = _loaded(tmp_path, max_epoch=0)  # no epoch: the weights are those the training starts from
        model = classifier.model

        classifier.train(TEXTS, CLASSES, incremental=True)

        assert classifier.model is model
        assert _same(saved, classifier.model.state_dict())

    def test_and_it_trains(self, tmp_path, monkeypatch):
        saved = _saved(tmp_path, monkeypatch)
        classifier = _loaded(tmp_path, max_epoch=2)

        classifier.train(TEXTS, CLASSES, incremental=True)

        assert not _same(saved, classifier.model.state_dict())
        assert classifier.predict(TEXTS, output_format="array").shape == (8, 2)

    def test_without_incremental_a_new_model_is_trained(self, tmp_path, monkeypatch):
        saved = _saved(tmp_path, monkeypatch)
        classifier = _loaded(tmp_path, max_epoch=0)
        model = classifier.model

        classifier.train(TEXTS, CLASSES)

        assert classifier.model is not model
        assert not _same(saved, classifier.model.state_dict())

    def test_no_model_loaded_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="load one first"):
            _classifier().train(TEXTS, CLASSES, incremental=True)

    def test_data_with_other_classes_than_the_model_is_an_error(self, tmp_path, monkeypatch):
        _saved(tmp_path, monkeypatch)
        three_classes = np.array([[1, 0, 0], [0, 1, 0]] * 4, dtype=np.float32)
        with pytest.raises(ValueError, match="2 classes, and the training data 3"):
            _loaded(tmp_path).train(TEXTS, three_classes, incremental=True)


def test_a_classifier_trained_with_class_weights_loads_back_and_goes_on(tmp_path, monkeypatch):
    """
    As the copyright classifier: its class weights were saved with the model, which could
    then not be loaded, to classify or to train on.
    """
    monkeypatch.chdir(tmp_path)
    classifier = _classifier(class_weights={0: 1.0, 1: 3.0})
    classifier.train(TEXTS, CLASSES)
    scores = classifier.predict(TEXTS, output_format="array")
    classifier.save(str(tmp_path))

    loaded = _loaded(tmp_path, class_weights={0: 1.0, 1: 3.0}, max_epoch=2)
    assert np.allclose(loaded.predict(TEXTS, output_format="array"), scores)
    loaded.train(TEXTS, CLASSES, incremental=True)
    assert loaded.model.loss_fn.weight.tolist() == [1.0, 3.0]
    assert not np.allclose(loaded.predict(TEXTS, output_format="array"), scores)


def test_training_over_folds_says_it_is_not_implemented(tmp_path, monkeypatch):
    """It did nothing and said nothing: started from a loaded model, that model was saved as if trained."""
    monkeypatch.chdir(tmp_path)
    classifier = _classifier(fold_number=3)
    with pytest.raises(NotImplementedError, match="3 folds"):
        classifier.train(TEXTS, CLASSES)
    with pytest.raises(NotImplementedError, match="3 folds"):
        classifier.train_nfold(TEXTS, CLASSES, incremental=True)


class _Model:
    def __init__(self):
        self.calls = []

    def load(self):
        self.calls.append("load")

    def train(self, x, y, incremental=False):
        self.calls.append(("train", incremental))

    def train_nfold(self, x, y, incremental=False):
        self.calls.append(("train_nfold", incremental))


class TestLicenseApplication:
    def test_incremental_loads_the_saved_model_and_goes_on(self):
        model = _Model()
        licenseClassifier._train(model, TEXTS, CLASSES, 1, incremental=True)
        assert model.calls == ["load", ("train", True)]

    def test_otherwise_nothing_is_loaded(self):
        model = _Model()
        licenseClassifier._train(model, TEXTS, CLASSES, 1)
        assert model.calls == [("train", False)]

    def test_with_folds(self):
        model = _Model()
        licenseClassifier._train(model, TEXTS, CLASSES, 3, incremental=True)
        assert model.calls == ["load", ("train_nfold", True)]

    def test_the_command_line_has_the_option_the_submitter_passes(self):
        """scripts/train_distributed_array.sh adds --incremental to the license profile with INCREMENTAL=true."""
        result = subprocess.run(
            [sys.executable, "-m", "delft.applications.licenseClassifier", "--help"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr[-500:]
        assert "--incremental" in result.stdout
