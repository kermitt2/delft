"""
A model with a transformer inside saves the configuration and the tokenizer of the
transformer next to its weights, as DeLFT 0.4 did: it then loads without the Hugging
Face Hub, or a cache of the transformer. Without them, loading a model asked the Hub
for both, and failed offline.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from delft.sequenceLabelling.wrapper import Sequence
from delft.textClassification.wrapper import Classifier
from delft.utilities.hub_publish import prepare_model
from delft.utilities.transformer_models import TRANSFORMER_CONFIG_FILE_NAME, TRANSFORMER_TOKENIZER_DIR
from delft.utilities.transformer_tokenizers import clear_tokenizers
from tests.utilities.test_transformer_tokenizers import CLASSES, TEXTS, WORDS, X, Y, _tiny_transformer, _tokenizer


@pytest.fixture
def offline(monkeypatch):
    """Nothing comes from the Hub: the transformer, "in-memory", is on it no more than in a cache."""
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    clear_tokenizers()
    yield
    clear_tokenizers()


def _trained(make):
    with (
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _tokenizer()),
        patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
    ):
        return make()


def _sequence(**options):
    return Sequence(
        "test-model",
        architecture="BERT_CRF",
        embeddings_name=None,
        transformer_name="in-memory",
        max_sequence_length=16,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
        **options,
    )


def _classifier():
    return Classifier(
        "test-classifier",
        architecture="bert",
        transformer_name="in-memory",
        embeddings_name=None,
        list_classes=["a", "b"],
        maxlen=8,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
    )


def test_a_sequence_labelling_model_loads_and_tags_offline(tmp_path, monkeypatch, offline):
    monkeypatch.chdir(tmp_path)

    def train():
        sequence = _sequence()
        sequence.train(X, Y)
        sequence.save(str(tmp_path))
        return sequence.tag([WORDS], "list")

    tagged = _trained(train)
    directory = tmp_path / "test-model"
    assert (directory / TRANSFORMER_CONFIG_FILE_NAME).is_file()
    assert (directory / TRANSFORMER_TOKENIZER_DIR).is_dir()

    clear_tokenizers()
    loaded = _sequence()
    loaded.load(str(tmp_path))
    assert loaded.tag([WORDS], "list") == tagged


def test_a_classifier_loads_and_classifies_offline(tmp_path, monkeypatch, offline):
    monkeypatch.chdir(tmp_path)

    def train():
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)
        classifier.save(str(tmp_path))
        return classifier.predict(TEXTS, output_format="array")

    predicted = _trained(train)
    directory = tmp_path / "test-classifier"
    assert (directory / TRANSFORMER_CONFIG_FILE_NAME).is_file()
    assert (directory / TRANSFORMER_TOKENIZER_DIR).is_dir()

    clear_tokenizers()
    loaded = _classifier()
    loaded.load(str(tmp_path))
    assert np.allclose(loaded.predict(TEXTS, output_format="array"), predicted, atol=1e-6)


def test_a_model_saved_without_them_still_asks_for_the_transformer_by_name(tmp_path, monkeypatch, offline):
    """The models saved by DeLFT 1.0 and 1.1."""
    monkeypatch.chdir(tmp_path)

    def train():
        sequence = _sequence()
        sequence.train(X, Y)
        sequence.save(str(tmp_path))

    _trained(train)
    directory = tmp_path / "test-model"
    os.remove(directory / TRANSFORMER_CONFIG_FILE_NAME)
    for name in os.listdir(directory / TRANSFORMER_TOKENIZER_DIR):
        os.remove(directory / TRANSFORMER_TOKENIZER_DIR / name)
    os.rmdir(directory / TRANSFORMER_TOKENIZER_DIR)

    clear_tokenizers()
    with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")) as config:
        with pytest.raises(OSError):
            _sequence().load(str(tmp_path))
    config.assert_called_once_with("in-memory")


def test_the_tokenizer_is_published_with_the_model(tmp_path, monkeypatch, offline):
    monkeypatch.chdir(tmp_path)

    def train():
        sequence = _sequence()
        sequence.train(X, Y)
        sequence.save(str(tmp_path))

    _trained(train)
    published = prepare_model(str(tmp_path / "test-model"), str(tmp_path / "staging"))
    assert TRANSFORMER_CONFIG_FILE_NAME in published
    tokenizer_files = [name for name in published if name.startswith(TRANSFORMER_TOKENIZER_DIR + "/")]
    assert tokenizer_files
    for name in tokenizer_files:
        assert (tmp_path / "staging" / name).is_file()
