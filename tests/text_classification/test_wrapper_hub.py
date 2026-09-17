"""Where the Classifier takes a model of the Hugging Face Hub from, the loading itself being left out."""

import pytest

from delft.textClassification import wrapper
from delft.textClassification.wrapper import Classifier


@pytest.fixture
def resolved(monkeypatch):
    """The references asked to the Hub, and the directories loaded."""
    calls = []

    def resolve_model(reference, cache_dir=None, token=None, model_name=None):
        calls.append((str(reference), model_name, cache_dir, token))
        return f"{cache_dir}/{model_name}"

    monkeypatch.setattr(wrapper, "resolve_model", resolve_model)
    monkeypatch.setattr(Classifier, "_load_from_directory", lambda self, model_path: calls.append(model_path))
    return calls


def test_load_takes_a_repository_and_looks_for_its_own_name_in_it(resolved):
    Classifier("toxic-gru", device="cpu").load("hf://owner/repository@v1", cache_dir="/cache", token="secret")
    assert resolved == [("hf://owner/repository@v1", "toxic-gru", "/cache", "secret"), "/cache/toxic-gru"]


def test_load_takes_a_url(resolved):
    Classifier("toxic-gru", device="cpu").load("https://example.org/models/", cache_dir="/cache")
    assert resolved == [("https://example.org/models/", "toxic-gru", "/cache", None), "/cache/toxic-gru"]


def test_load_takes_the_directory_of_a_model_whatever_its_name(resolved, tmp_path):
    (tmp_path / Classifier.config_file).write_text("{}")
    Classifier("toxic-gru", device="cpu").load(str(tmp_path))
    assert resolved == [str(tmp_path)]
