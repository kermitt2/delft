"""Columns of numbers given to a FEATURES model as numbers rather than as categories."""

import numpy as np
import pytest
import torch

from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.wrapper import Sequence
from delft.utilities.preprocess import FeaturesPreprocessor, scale_number, to_number

# column 0: token, 1: a category, 2: a number from 0 to 40
X = [["a", "b", "c"], ["d"]]
Y = [["B-x", "I-x", "O"], ["O"]]
FEATURES = [[["a", "UP", "0"], ["b", "LOW", "10"], ["c", "UP", "40"]], [["d", "LOW", "20"]]]


class TestNumbers:
    @pytest.mark.parametrize(
        "value, number", [("3", 3.0), ("-1.5", -1.5), (7, 7.0), ("x", None), ("", None), (None, None)]
    )
    def test_to_number(self, value, number):
        assert to_number(value) == number

    @pytest.mark.parametrize("value", ["nan", "inf"])
    def test_a_number_that_is_not_finite_is_not_one(self, value):
        assert to_number(value) is None

    @pytest.mark.parametrize(
        "value, scaled", [("0", 0.0), ("10", 0.25), ("40", 1.0), ("80", 1.0), ("-5", 0.0), ("x", 0.0), (None, 0.0)]
    )
    def test_scale_number_clips_and_takes_anything(self, value, scaled):
        assert scale_number(value, 0.0, 40.0) == scaled

    def test_a_column_with_a_single_value_scales_to_zero(self):
        assert scale_number("5", 5.0, 5.0) == 0.0


class TestFeaturesPreprocessor:
    def test_learns_the_range_of_each_column(self):
        fp = FeaturesPreprocessor(continuous_features_indices=[2]).fit(FEATURES)
        assert fp.continuous_features_ranges == [[0.0, 40.0]]
        assert fp.transform_continuous(FEATURES) == [[[0.0], [0.25], [1.0]], [[0.5]]]

    def test_a_column_of_numbers_is_not_a_category_as_well(self):
        fp = FeaturesPreprocessor(continuous_features_indices=[2]).fit(FEATURES)
        assert 2 not in fp.features_indices and 1 in fp.features_indices

    def test_without_continuous_columns_nothing_changes(self):
        fp = FeaturesPreprocessor().fit(FEATURES)
        assert fp.continuous_features_indices == [] and fp.continuous_features_ranges == []
        assert 2 in fp.features_indices

    def test_a_column_asked_for_both_ways_is_an_error(self):
        with pytest.raises(ValueError, match=r"Columns \[2\]"):
            FeaturesPreprocessor(features_indices=[1, 2], continuous_features_indices=[2]).fit(FEATURES)

    def test_a_column_without_a_number_is_an_error(self):
        with pytest.raises(ValueError, match="column 1 holds no number"):
            FeaturesPreprocessor(continuous_features_indices=[1]).fit(FEATURES)

    def test_a_short_row_and_the_extension_row_are_zeros(self):
        fp = FeaturesPreprocessor(continuous_features_indices=[2]).fit(FEATURES)
        assert fp.transform_continuous([[["a", "UP"]]], extend=True) == [[[0.0], [0.0]]]


def _sequence(tmp_path, monkeypatch, architecture="BidLSTM_CRF_FEATURES", **kwargs):
    monkeypatch.chdir(tmp_path)
    return Sequence(
        "test-model",
        architecture=architecture,
        embeddings_name=None,
        max_sequence_length=10,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
        **kwargs,
    )


def _arrays():
    return (np.array(X, dtype=object), np.array(Y, dtype=object), np.array(FEATURES, dtype=object))


@pytest.mark.parametrize("architecture", ["BidLSTM_CRF_FEATURES", "BidLSTM_ChainCRF_FEATURES"])
def test_trains_saves_and_tags_with_numbers(tmp_path, monkeypatch, architecture):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, architecture, continuous_features_indices=[2])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    sequence.eval(x, y, features=f)
    sequence.save(str(tmp_path))

    loaded = Sequence("test-model", nb_workers=0, device="cpu")
    loaded.load(str(tmp_path))
    assert loaded.model_config.continuous_features_indices == [2]
    assert loaded.p.feature_preprocessor.continuous_features_ranges == [[0.0, 40.0]]
    assert loaded.tag(X, "raw", features=FEATURES) == sequence.tag(X, "raw", features=FEATURES)


def test_the_loader_gives_the_numbers_of_each_token(tmp_path, monkeypatch):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, continuous_features_indices=[2])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    loader = create_dataloader(
        X, None, preprocessor=sequence.p, features=FEATURES, shuffle=False, model_config=sequence.model_config
    )
    inputs, _ = next(iter(loader))
    # the second sequence has one token: it is extended to two for the CRF, then padded to three
    assert inputs["continuous_features_input"].tolist() == [[[0.0], [0.25], [1.0]], [[0.5], [0.0], [0.0]]]


def test_the_numbers_reach_the_model(tmp_path, monkeypatch):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, continuous_features_indices=[2])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    sequence.model.eval()

    def logits(features):
        loader = create_dataloader(
            X, None, preprocessor=sequence.p, features=features, shuffle=False, model_config=sequence.model_config
        )
        inputs, _ = next(iter(loader))
        with torch.no_grad():
            return sequence.model(inputs)["logits"]

    other = [[[token, category, "40"] for token, category, _ in document] for document in FEATURES]
    assert not torch.allclose(logits(FEATURES), logits(other))


def test_a_model_without_numbers_has_no_such_input(tmp_path, monkeypatch):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch)
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    loader = create_dataloader(
        X, None, preprocessor=sequence.p, features=FEATURES, shuffle=False, model_config=sequence.model_config
    )
    inputs, _ = next(iter(loader))
    assert "continuous_features_input" not in inputs


@pytest.fixture
def number_tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "##c", "##d"]
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


def test_with_a_transformer_the_numbers_of_a_word_are_on_all_its_sub_tokens(number_tokenizer):
    from unittest.mock import patch

    from delft.sequenceLabelling.config import ModelConfig
    from delft.sequenceLabelling.preprocess import prepare_preprocessor

    words = [["a", "bcd", "a"]]
    labels = [["B-x", "I-x", "O"]]
    features = [[["a", "UP", "0"], ["bcd", "LOW", "10"], ["a", "UP", "40"]]]
    config = ModelConfig(
        architecture="BERT_CRF_FEATURES",
        embeddings_name=None,
        transformer_name="in-memory",
        max_sequence_length=16,
        continuous_features_indices=[2],
    )
    preprocessor = prepare_preprocessor(words, labels, config, features=features)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=number_tokenizer):
        loader = create_dataloader(
            words, labels, preprocessor=preprocessor, features=features, shuffle=False, model_config=config
        )
        inputs, _ = next(iter(loader))

    # [CLS] a b ##c ##d a [SEP]
    assert number_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "[CLS]", "a", "b", "##c", "##d", "a", "[SEP]",
    ]  # fmt: skip
    assert inputs["continuous_features_input"][0].squeeze(-1).tolist() == [0.0, 0.0, 0.25, 0.25, 0.25, 1.0, 0.0]
    assert inputs["continuous_features_input"].shape[:2] == inputs["features_input"].shape[:2]
