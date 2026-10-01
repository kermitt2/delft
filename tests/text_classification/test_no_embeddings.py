"""
A text classifier given no pre-trained word embeddings, as a sequence labelling model can
be: it learns the embeddings of the words of its training texts. And one given embeddings
reads vectors of their size, which was taken for 300 whatever the embeddings.
"""

import numpy as np
import pytest
import torch

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import DEFAULT_WORD_EMBEDDING_SIZE, MODEL_REGISTRY, getModel
from delft.textClassification.wrapper import Classifier

TEXTS = ["good work", "bad work", "good good work", "bad bad work", "work good", "work bad", "good", "bad"]
CLASSES = np.array([[1, 0], [0, 1]] * 4, dtype=np.float32)
ARCHITECTURES = sorted(name for name in MODEL_REGISTRY if name != "bert")


class _Embeddings:
    """Word vectors that need no download, of any size."""

    def __init__(self, embed_size):
        self.embed_size = embed_size

    def get_word_vector(self, word):
        return np.random.RandomState(sum(word.encode())).randn(self.embed_size).astype("float32")


def _classifier(embeddings=None, **kwargs):
    options = dict(
        architecture="gru",
        list_classes=["a", "b"],
        maxlen=8,
        max_epoch=2,
        batch_size=4,
        early_stop=False,
        nb_workers=0,
        device="cpu",
        embeddings_name=None,
    )
    options.update(kwargs)
    classifier = Classifier("test-classifier", **options)
    if embeddings is not None:
        # what the wrapper does when it loads embeddings by their name
        classifier.embeddings = embeddings
        classifier.model_config.word_embedding_size = embeddings.embed_size
    return classifier


class TestModels:
    @pytest.mark.parametrize("architecture", ARCHITECTURES)
    @pytest.mark.parametrize("size", [64, 256, 1024])
    def test_reads_vectors_of_the_size_of_its_embeddings(self, architecture, size):
        """potion-base-8M has 256 dimensions: every architecture expected 300 and failed on the first batch."""
        config = ModelConfig(architecture=architecture, list_classes=["a", "b"], maxlen=100, word_emb_size=size)
        model = getModel(config, TrainingConfig(learning_rate=1e-3)).eval()
        assert model(torch.randn(2, 100, size))["logits"].shape == (2, 2)

    @pytest.mark.parametrize("architecture", ARCHITECTURES)
    def test_learns_its_word_embeddings_when_the_configuration_gives_a_vocabulary(self, architecture):
        config = ModelConfig(architecture=architecture, list_classes=["a", "b"], maxlen=100, word_emb_size=0)
        config.vocab_size = 50
        model = getModel(config, TrainingConfig(learning_rate=1e-3)).eval()
        assert model.embedding.weight.shape == (50, DEFAULT_WORD_EMBEDDING_SIZE)
        assert model(torch.randint(0, 50, (2, 100)))["logits"].shape == (2, 2)

    def test_no_embedding_layer_without_a_vocabulary(self):
        config = ModelConfig(architecture="gru", list_classes=["a", "b"], word_emb_size=300)
        assert getModel(config, TrainingConfig(learning_rate=1e-3)).embedding is None


class TestClassifier:
    @pytest.mark.parametrize("architecture", ["gru", "cnn", "lstm"])
    def test_trains_classifies_and_loads_back_without_embeddings(self, tmp_path, monkeypatch, architecture):
        """It failed on the first text: there were no embeddings to take its vectors from."""
        monkeypatch.chdir(tmp_path)
        classifier = _classifier(architecture=architecture, maxlen=100 if architecture == "dpcnn" else 8)
        classifier.train(TEXTS, CLASSES)

        assert classifier.embeddings is None
        assert classifier.model_config.vocab_size == len(classifier.preprocessor.vocab_word) == 5  # 3 words, PAD, UNK
        assert classifier.model.embedding.weight.requires_grad
        classifier.eval(TEXTS, CLASSES)
        scores = classifier.predict(TEXTS, output_format="array")
        assert scores.shape == (8, 2)
        classifier.save(str(tmp_path))

        loaded = Classifier("test-classifier", device="cpu", nb_workers=0)
        loaded.load(str(tmp_path))
        assert loaded.embeddings is None and loaded.model_config.embeddings_name is None
        assert loaded.preprocessor.vocab_word == classifier.preprocessor.vocab_word
        assert np.allclose(loaded.predict(TEXTS, output_format="array"), scores)
        # a word the training texts did not have is the unknown word, not an error
        assert loaded.predict(["never seen work"], output_format="array").shape == (1, 2)

    def test_the_embeddings_are_learned(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(0)
        classifier = _classifier(max_epoch=1)
        classifier.train(TEXTS, CLASSES)
        before = classifier.model.embedding.weight.detach().clone()
        classifier.save(str(tmp_path))

        loaded = _classifier(max_epoch=3)
        loaded.load(str(tmp_path))
        loaded.train(TEXTS, CLASSES, incremental=True)
        after = loaded.model.embedding.weight.detach()
        assert not torch.equal(before[2:], after[2:])
        # incremental training keeps the vocabulary of the model
        assert loaded.preprocessor.vocab_word == classifier.preprocessor.vocab_word

    def test_with_early_stopping_the_validation_texts_are_in_the_vocabulary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier(early_stop=True)
        classifier.train(TEXTS, CLASSES)
        assert set(classifier.preprocessor.vocab_word) >= {"good", "bad", "work"}

    @pytest.mark.parametrize("size", [256, 1024])
    def test_trains_with_embeddings_that_are_not_300_wide(self, tmp_path, monkeypatch, size):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier(embeddings=_Embeddings(size))
        classifier.train(TEXTS, CLASSES)
        assert classifier.preprocessor is None and classifier.model_config.vocab_size is None
        assert classifier.model.embedding is None
        assert classifier.predict(TEXTS, output_format="array").shape == (8, 2)

    def test_a_vocabulary_left_by_an_earlier_training_is_not_taken_for_the_one_of_the_model(
        self, tmp_path, monkeypatch
    ):
        """Trained without embeddings, then with: the directory held the vocabulary of the first."""
        monkeypatch.chdir(tmp_path)
        first = _classifier()
        first.train(TEXTS, CLASSES)
        first.save(str(tmp_path))
        assert (tmp_path / "test-classifier" / "preprocessor.json").is_file()

        second = _classifier(embeddings=_Embeddings(300))
        second.train(TEXTS, CLASSES)
        scores = second.predict(TEXTS, output_format="array")
        second.save(str(tmp_path))
        assert not (tmp_path / "test-classifier" / "preprocessor.json").exists()

        loaded = Classifier("test-classifier", device="cpu", nb_workers=0)
        loaded.load(str(tmp_path))
        loaded.embeddings = _Embeddings(300)
        assert loaded.preprocessor is None
        assert np.allclose(loaded.predict(TEXTS, output_format="array"), scores)
