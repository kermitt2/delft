"""
The applications refuse what they cannot do with a non-zero exit, before loading anything: an
unknown action or architecture used to print a message and end with 0, which a script running
them could not tell from a success. Also, the actions that went unused for long enough to break.
"""

import importlib
import runpy
import sys

import numpy as np
import pytest

GROBID = ["date"]  # the model, which grobidTagger takes before the action
APPLICATIONS = {
    "grobidTagger": GROBID,
    "nerTagger": [],
    "datasetTagger": [],
    "insultTagger": [],
    "citationClassifier": [],
    "licenseClassifier": [],
    "softwareClassifier": [],
    "softwareContextClassifier": [],
    "toxicCommentClassifier": [],
}


def run(app_name, arguments, monkeypatch):
    path = importlib.import_module(f"delft.applications.{app_name}").__file__
    monkeypatch.setattr(sys, "argv", [path, *arguments])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(path, run_name="__main__")
    return exited.value.code


@pytest.mark.parametrize("app_name, before", APPLICATIONS.items())
def test_an_unknown_action_is_an_error(app_name, before, monkeypatch):
    assert run(app_name, [*before, "frobnicate"], monkeypatch) == 2


@pytest.mark.parametrize("app_name, before", APPLICATIONS.items())
def test_an_unknown_architecture_is_an_error(app_name, before, monkeypatch):
    assert run(app_name, [*before, "train", "--architecture", "Nope"], monkeypatch) == 2


def test_the_toxic_comment_classifier_refuses_bert_with_an_error(monkeypatch):
    assert run("toxicCommentClassifier", ["train", "--architecture", "bert"], monkeypatch) == 2


@pytest.mark.parametrize(
    "architecture, batch_size, max_epoch, early_stop",
    [("BidLSTM_CRF", 20, 50, True), ("BERT_CRF", 10, 3, False)],
)
def test_the_insult_tagger_has_defaults_of_its_own_for_bert(architecture, batch_size, max_epoch, early_stop):
    configure = importlib.import_module("delft.applications.insultTagger").configure
    got_batch_size, _, _, got_early_stop, got_max_epoch, _ = configure(architecture, None)
    assert (got_batch_size, got_max_epoch, got_early_stop) == (batch_size, max_epoch, early_stop)


class FakeModel:
    """Records what an application gives to the Classifier or Sequence it builds, and trains nothing."""

    built = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.calls = []
        FakeModel.built.append(self)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)

        return record


@pytest.fixture
def fake_model():
    FakeModel.built = []
    return FakeModel


@pytest.mark.parametrize("function", ["train_binary", "train_and_eval_binary"])
def test_the_binary_license_classifiers_take_the_options_of_the_command_line(function, fake_model, monkeypatch):
    app = importlib.import_module("delft.applications.licenseClassifier")
    texts = np.array([f"text {i}" for i in range(20)])

    def read_data(path, data_type):
        classes = app.list_classes_copyright if data_type == "copyrights" else app.list_classes_licenses
        return texts, np.eye(len(classes))[np.arange(len(texts)) % len(classes)]

    monkeypatch.setattr(app, "Classifier", fake_model)
    monkeypatch.setattr(app, "_read_data", read_data)

    getattr(app, function)(None, 1, architecture="gru", max_epoch=2, batch_size=8, learning_rate=0.01, num_workers=0)

    assert len(fake_model.built) == len(app.list_classes_copyright) + len(app.list_classes_licenses)
    for model in fake_model.built:
        assert model.kwargs["max_epoch"] == 2
        assert model.kwargs["batch_size"] == 8
        assert model.kwargs["learning_rate"] == 0.01
        assert model.kwargs["nb_workers"] == 0
        assert "train" in model.calls


def test_the_dataset_tagger_trains_and_evaluates(fake_model, monkeypatch):
    app = importlib.import_module("delft.applications.datasetTagger")
    x = np.array([["a", "dataset"]] * 40, dtype=object)
    y = np.array([["O", "B-dataset"]] * 40, dtype=object)
    monkeypatch.setattr(app, "load_data_and_labels_json_offsets", lambda path: (x, y))
    monkeypatch.setattr(app, "Sequence", fake_model)

    app.train_eval(architecture="BidLSTM_CRF", input_path="sentences.json", max_epoch=1, patience=2)

    (model,) = fake_model.built
    assert model.model_name == "datasets-BidLSTM_CRF"
    assert model.kwargs["max_epoch"] == 1
    assert model.kwargs["patience"] == 2
    assert model.calls == ["train", "eval", "save"]


def test_the_dataset_tagger_evaluates_a_saved_model(fake_model, monkeypatch):
    app = importlib.import_module("delft.applications.datasetTagger")
    x = np.array([["a", "dataset"]], dtype=object)
    y = np.array([["O", "B-dataset"]], dtype=object)
    monkeypatch.setattr(app, "load_data_and_labels_json_offsets", lambda path: (x, y))
    monkeypatch.setattr(app, "Sequence", fake_model)

    app.eval_(input_path="sentences.json", architecture="BidLSTM_CRF")

    (model,) = fake_model.built
    assert model.model_name == "datasets-BidLSTM_CRF"
    assert model.calls == ["load", "eval"]
