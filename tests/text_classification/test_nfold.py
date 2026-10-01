"""
A text classifier trained over several folds, as DeLFT did with Keras: one model per
fold, trained on the texts of the other folds, and the models classify together. The
training over folds did nothing in the PyTorch version.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from delft.textClassification.trainer import Trainer
from delft.textClassification.wrapper import Classifier, fold_weight_file

TEXTS = [f"{word} work number {i}" for i in range(5) for word in ("good", "bad")] + ["good"]  # 11 texts
CLASSES = np.array([[1, 0], [0, 1]] * 5 + [[1, 0]], dtype=np.float32)
FOLDS = 3


class _Embeddings:
    """Word vectors that need no download: a word always gets the same random vector."""

    embed_size = 300

    def get_word_vector(self, word):
        return np.random.RandomState(sum(word.encode())).randn(self.embed_size).astype("float32")


def _classifier(embeddings=True, **kwargs):
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
        fold_number=FOLDS,
    )
    options.update(kwargs)
    classifier = Classifier("test-classifier", **options)
    if embeddings:
        classifier.embeddings = _Embeddings()
    return classifier


def _loaded(tmp_path, embeddings=True, **kwargs):
    classifier = _classifier(embeddings=embeddings, **kwargs)
    classifier.load(str(tmp_path))
    if embeddings:
        classifier.embeddings = _Embeddings()
    return classifier


def _alone(classifier, model, texts):
    """What one model of the classifier predicts by itself."""
    single = _classifier(fold_number=1)
    single.model = model
    return single.predict(texts, output_format="array")


class TestTraining:
    def test_one_model_per_fold(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)

        assert len(classifier.models) == FOLDS and classifier.model is None
        states = [model.state_dict() for model in classifier.models]
        assert len({id(model) for model in classifier.models}) == FOLDS
        assert not all(torch.equal(states[0][name], states[1][name]) for name in states[0])

    @pytest.mark.parametrize("early_stop", [False, True])
    def test_each_model_is_trained_on_the_other_folds(self, tmp_path, monkeypatch, early_stop):
        """11 texts in 3 folds: 3, 3 and the 5 the division leaves to the last one."""
        monkeypatch.chdir(tmp_path)
        seen = []

        def train(self, train_loader, valid_loader=None):
            seen.append(
                (
                    sorted(train_loader.dataset.x.tolist()),
                    None if valid_loader is None else sorted(valid_loader.dataset.x.tolist()),
                )
            )

        with patch.object(Trainer, "train", train):
            _classifier(early_stop=early_stop).train(TEXTS, CLASSES)

        assert [len(train_texts) for train_texts, _ in seen] == [8, 8, 6]
        held_out = [sorted(set(TEXTS) - set(train_texts)) for train_texts, _ in seen]
        # every text is held out of one fold, and of one only
        assert sorted(text for fold in held_out for text in fold) == sorted(TEXTS)
        if early_stop:
            # the fold is the validation set of its model
            assert [valid_texts for _, valid_texts in seen] == held_out
        else:
            assert all(valid_texts is None for _, valid_texts in seen)

    def test_texts_keep_their_labels_through_the_shuffle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pairs = []

        def train(self, train_loader, valid_loader=None):
            dataset = train_loader.dataset
            pairs.extend(zip(dataset.x.tolist(), np.argmax(dataset.y, axis=1).tolist()))

        with patch.object(Trainer, "train", train):
            _classifier().train(TEXTS, CLASSES)
        assert pairs and all(label == (0 if text.startswith("good") else 1) for text, label in pairs)

    def test_fewer_texts_than_folds_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="2 texts cannot be cut into 3 folds"):
            _classifier().train(TEXTS[:2], CLASSES[:2])

    def test_without_embeddings_the_folds_share_the_vocabulary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier(embeddings=False)
        classifier.train(TEXTS, CLASSES)
        assert classifier.model_config.vocab_size == len(classifier.preprocessor.vocab_word)
        assert all(model.embedding.num_embeddings == classifier.model_config.vocab_size for model in classifier.models)
        assert classifier.predict(TEXTS, output_format="array").shape == (11, 2)


class TestClassifying:
    def test_the_models_classify_together_by_the_geometric_mean_of_their_probabilities(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)

        each = [_alone(classifier, model, TEXTS) for model in classifier.models]
        assert not np.allclose(each[0], each[1])
        expected = (each[0] * each[1] * each[2]) ** (1 / 3)
        np.testing.assert_allclose(classifier.predict(TEXTS, output_format="array"), expected, rtol=1e-5)

        result = classifier.predict(TEXTS[:2], output_format="json")
        assert [c["class"] in {"a", "b"} for c in result["classifications"]] == [True, True]

    def test_evaluation_is_that_of_the_models_together(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)
        classifier.eval(TEXTS, CLASSES)
        assert "Evaluation on 11 instances" in capsys.readouterr().out

    def test_no_fold_models_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(OSError, match="nfolds models"):
            _classifier().predict(TEXTS)


class TestSavingAndLoading:
    def test_the_weights_of_every_fold_are_saved_and_loaded_back(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)
        scores = classifier.predict(TEXTS, output_format="array")
        classifier.save(str(tmp_path))

        directory = tmp_path / "test-classifier"
        weights = sorted(name for name in os.listdir(directory) if name.endswith(".safetensors"))
        assert weights == ["model_fold0.safetensors", "model_fold1.safetensors", "model_fold2.safetensors"]

        loaded = _loaded(tmp_path, fold_number=1)  # the number of folds is that of the saved model
        assert loaded.model_config.fold_number == FOLDS and len(loaded.models) == FOLDS and loaded.model is None
        np.testing.assert_allclose(loaded.predict(TEXTS, output_format="array"), scores, rtol=1e-6)

    def test_pickled_weights(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)
        scores = classifier.predict(TEXTS, output_format="array")
        classifier.save(str(tmp_path), weight_file="model_weights.pth")
        assert (tmp_path / "test-classifier" / "model_weights_fold2.pth").is_file()
        np.testing.assert_allclose(_loaded(tmp_path).predict(TEXTS, output_format="array"), scores, rtol=1e-6)

    def test_the_weights_of_an_earlier_training_do_not_stay(self, tmp_path, monkeypatch):
        """A single model, then three folds, then two, then a single model again, in the same directory."""
        monkeypatch.chdir(tmp_path)
        directory = tmp_path / "test-classifier"

        def weights():
            return sorted(name for name in os.listdir(directory) if name.endswith((".safetensors", ".pth")))

        for fold_number, expected in (
            (1, ["model.safetensors"]),
            (3, [fold_weight_file("model.safetensors", i) for i in range(3)]),
            (2, [fold_weight_file("model.safetensors", i) for i in range(2)]),
            (1, ["model.safetensors"]),
        ):
            classifier = _classifier(fold_number=fold_number)
            classifier.train(TEXTS, CLASSES)
            classifier.save(str(tmp_path))
            assert weights() == expected
            assert _loaded(tmp_path).predict(TEXTS, output_format="array").shape == (11, 2)


class TestIncremental:
    def test_the_training_of_every_fold_model_goes_on(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        classifier = _classifier()
        classifier.train(TEXTS, CLASSES)
        classifier.save(str(tmp_path))

        loaded = _loaded(tmp_path, max_epoch=2)
        models = list(loaded.models)
        before = [{name: tensor.clone() for name, tensor in model.state_dict().items()} for model in models]
        loaded.train(TEXTS, CLASSES, incremental=True)

        assert loaded.models == models
        for model, state in zip(loaded.models, before):
            assert not all(torch.equal(model.state_dict()[name], state[name]) for name in state)

    def test_no_fold_models_loaded_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="load them first"):
            _classifier().train(TEXTS, CLASSES, incremental=True)


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "work", "number"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


class TestTransformer:
    def test_a_transformer_classifier_over_folds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _tokenizer()),
            patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
        ):
            classifier = _classifier(embeddings=False, architecture="bert", transformer_name="in-memory", fold_number=2)
            classifier.train(TEXTS, CLASSES)
            scores = classifier.predict(TEXTS, output_format="array")
            classifier.save(str(tmp_path))
            loaded = _loaded(tmp_path, embeddings=False)
            np.testing.assert_allclose(loaded.predict(TEXTS, output_format="array"), scores, rtol=1e-5)
        assert len(classifier.models) == 2

    def test_fold_transformers_are_kept_out_of_a_gpu_between_their_runs(self):
        """Several transformers may not fit in it together: on a CPU there is nothing to keep them out of."""
        classifier = _classifier(embeddings=False, architecture="bert", transformer_name="in-memory", fold_number=2)
        assert not classifier._parks_fold_models()
        classifier.device = torch.device("cuda")
        assert classifier._parks_fold_models()
        classifier.model_config.fold_number = 1
        assert not classifier._parks_fold_models()
        rnn = _classifier()
        rnn.device = torch.device("cuda")
        assert not rnn._parks_fold_models()
