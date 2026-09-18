"""The features channel of a text classifier (issue #152)."""

import numpy as np
import pytest
import torch

from delft.textClassification.config import ModelConfig
from delft.textClassification.features import (
    check_features,
    encode_features,
    features_preprocessor_from_config,
    features_size,
    fit_features_preprocessor,
)
from delft.textClassification.models import MODEL_REGISTRY, getModel
from delft.textClassification.wrapper import Classifier, split_train_validation

TEXTS = [
    "Jim Henson was a puppeteer",
    "The data are available on request",
    "Mississippi is a river and a state",
    "A puppeteer lives in Mississippi",
    "Datasets were deposited in a repository",
    "The river flows to the sea",
]
CLASSES = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
# a row per text: the section the text comes from, and a number
FEATURES = [["title", "0"], ["methods", "3"], ["intro", "1"], ["title", "0"], ["methods", "4"], ["intro", "2"]]


def _config(**kwargs):
    options = dict(list_classes=["a", "b"], maxlen=8, dense_size=4)
    options.update(kwargs)
    return ModelConfig(**options)


class TestPreprocessing:
    def test_the_values_of_a_column_are_mapped_to_indices_kept_in_the_config(self):
        config = _config()
        preprocessor = fit_features_preprocessor(FEATURES, config)
        indices, continuous = encode_features(preprocessor, FEATURES)
        assert config.use_features
        assert config.features_indices == [0, 1]
        assert indices.shape == (6, 2) and indices.dtype == np.int64
        assert continuous is None
        assert (indices[0] == indices[3]).all()  # the same values
        assert (indices[0] != indices[1]).any()
        assert features_size(config) == 2 * config.features_embedding_size

    def test_a_continuous_column_is_scaled_with_the_range_seen_when_fitting(self):
        config = _config(continuous_features_indices=[1])
        preprocessor = fit_features_preprocessor(FEATURES, config)
        indices, continuous = encode_features(preprocessor, FEATURES)
        assert indices.shape == (6, 1)
        assert continuous.shape == (6, 1) and continuous.dtype == np.float32
        assert continuous[:, 0].tolist() == pytest.approx([0.0, 0.75, 0.25, 0.0, 1.0, 0.5])
        assert config.continuous_features_ranges == [[0.0, 4.0]]
        assert features_size(config) == config.features_embedding_size + 1

    def test_the_preprocessor_is_rebuilt_from_a_config_saved_as_json(self, tmp_path):
        config = _config(continuous_features_indices=[1])
        fitted = fit_features_preprocessor(FEATURES, config)
        config.save(str(tmp_path / "config.json"))
        loaded = ModelConfig.load(str(tmp_path / "config.json"))
        rebuilt = features_preprocessor_from_config(loaded)
        assert loaded.features_map_to_index == config.features_map_to_index  # keys are integers again
        for encoded, expected in zip(encode_features(rebuilt, FEATURES), encode_features(fitted, FEATURES)):
            assert np.array_equal(encoded, expected)

    def test_no_preprocessor_for_a_model_without_features(self):
        assert features_preprocessor_from_config(_config()) is None
        assert features_size(_config()) == 0

    def test_features_must_be_given_to_a_model_that_takes_some_and_only_to_it(self):
        with_features = _config()
        fit_features_preprocessor(FEATURES, with_features)
        with pytest.raises(ValueError, match="takes features"):
            check_features(with_features, None)
        with pytest.raises(ValueError, match="takes none"):
            check_features(_config(), FEATURES)
        check_features(with_features, FEATURES)
        check_features(_config(), None)

    def test_the_validation_split_keeps_the_features_with_their_text(self):
        x_train, y_train, f_train, x_valid, y_valid, f_valid = split_train_validation(
            TEXTS, CLASSES, split_ratio=0.5, features=FEATURES
        )
        assert len(x_train) == len(y_train) == len(f_train) == 3
        assert len(x_valid) == len(y_valid) == len(f_valid) == 3
        expected = dict(zip(TEXTS, FEATURES))
        for texts, features in ((x_train, f_train), (x_valid, f_valid)):
            for text, row in zip(texts, features):
                assert list(row) == expected[text]


class TestModels:
    ARCHITECTURES = [name for name in MODEL_REGISTRY if name != "bert"]

    @staticmethod
    def _batch(config, features=True):
        torch.manual_seed(0)
        vectors = torch.randn(4, config.maxlen, 300)  # a sequence of 100: dpcnn pools a shorter one to nothing
        if not features:
            return vectors
        return {"text": vectors, "features": torch.tensor([[1, 5], [2, 6], [1, 7], [3, 5]])}

    @pytest.mark.parametrize("architecture", ARCHITECTURES)
    def test_the_features_of_a_text_take_part_in_its_classification(self, architecture):
        config = _config(architecture=architecture, maxlen=100)
        fit_features_preprocessor(FEATURES, config)
        torch.manual_seed(1)
        model = getModel(config, None).eval()
        batch = self._batch(config)
        logits = model(batch)["logits"]
        assert logits.shape == (4, 2)
        other = dict(batch, features=torch.tensor([[3, 7], [3, 7], [3, 7], [3, 7]]))
        assert not torch.allclose(logits, model(other)["logits"])

    @pytest.mark.parametrize("architecture", ARCHITECTURES)
    def test_a_model_without_features_reads_a_plain_tensor_as_before(self, architecture):
        config = _config(architecture=architecture, maxlen=100)
        torch.manual_seed(1)
        model = getModel(config, None).eval()
        assert model.features_size == 0
        assert model(self._batch(config, features=False))["logits"].shape == (4, 2)

    def test_continuous_features_are_concatenated_as_numbers(self):
        config = _config(architecture="gru", continuous_features_indices=[1])
        fit_features_preprocessor(FEATURES, config)
        model = getModel(config, None).eval()
        batch = self._batch(config)
        batch["features"] = batch["features"][:, :1]
        batch["continuous_features"] = torch.tensor([[0.0], [0.5], [1.0], [0.25]])
        assert model(batch)["logits"].shape == (4, 2)

    def test_a_batch_without_the_features_the_model_takes_is_an_error(self):
        config = _config(architecture="gru")
        fit_features_preprocessor(FEATURES, config)
        model = getModel(config, None).eval()
        with pytest.raises(ValueError, match="takes features"):
            model(self._batch(config, features=False))


class _Embeddings:
    """Word vectors that need no download: a word always gets the same random vector."""

    embed_size = 300

    def get_word_vector(self, word):
        return np.random.RandomState(hash(word) % (2**32)).randn(self.embed_size).astype("float32")


def _classifier(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    options = dict(
        architecture="gru", list_classes=["a", "b"], maxlen=8, max_epoch=1, batch_size=2, nb_workers=0, early_stop=False
    )
    options.update(kwargs)
    classifier = Classifier("test-classifier", embeddings_name=None, device="cpu", **options)
    classifier.embeddings = _Embeddings()
    return classifier


class TestClassifier:
    def test_trained_with_features_it_classifies_with_them_after_a_save_and_a_load(self, tmp_path, monkeypatch):
        classifier = _classifier(tmp_path, monkeypatch)
        classifier.train(TEXTS, CLASSES, features=FEATURES)
        assert classifier.model_config.use_features
        assert classifier.model.features_size > 0
        classifier.eval(TEXTS, CLASSES, features=FEATURES)
        scores = classifier.predict(TEXTS, output_format="array", features=FEATURES)
        assert scores.shape == (6, 2)
        classifier.save(str(tmp_path))

        loaded = Classifier("test-classifier", device="cpu")
        loaded.load(str(tmp_path))
        loaded.embeddings = _Embeddings()
        assert loaded.features_preprocessor is not None
        assert np.allclose(loaded.predict(TEXTS, output_format="array", features=FEATURES), scores)
        assert loaded.predict(TEXTS, features=FEATURES)["classifications"][0]["class"] in {"a", "b"}

    def test_with_early_stopping_the_validation_set_has_its_features(self, tmp_path, monkeypatch):
        classifier = _classifier(tmp_path, monkeypatch, early_stop=True)
        classifier.train(TEXTS, CLASSES, features=FEATURES)
        assert classifier.model.features_size > 0

    def test_a_model_trained_with_features_wants_them_to_classify(self, tmp_path, monkeypatch):
        classifier = _classifier(tmp_path, monkeypatch)
        classifier.train(TEXTS, CLASSES, features=FEATURES)
        with pytest.raises(ValueError, match="takes features"):
            classifier.predict(TEXTS)

    def test_features_are_refused_by_a_model_trained_without(self, tmp_path, monkeypatch):
        classifier = _classifier(tmp_path, monkeypatch)
        classifier.train(TEXTS, CLASSES)
        assert not classifier.model_config.use_features
        with pytest.raises(ValueError, match="takes none"):
            classifier.predict(TEXTS, features=FEATURES)
        assert classifier.predict(TEXTS, output_format="array").shape == (6, 2)

    def test_features_for_another_number_of_texts_are_an_error(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="features for"):
            _classifier(tmp_path, monkeypatch).train(TEXTS, CLASSES, features=FEATURES[:2])
