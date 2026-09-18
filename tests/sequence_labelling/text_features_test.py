"""The text of a token can be taken from columns of its features."""

import numpy as np
import pytest
import torch

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import Preprocessor, to_vector_single
from delft.sequenceLabelling.text_features import text_from_features, tokens_per_position
from delft.sequenceLabelling.wrapper import Sequence

# two lines: a position is a line, columns 0 and 1 are its first two tokens
X = [["Deep", "John"]]
FEATURES = [[["Deep", "learning", "BLOCKSTART"], ["John", "Smith", "BLOCKIN"]]]
Y = [["B-title", "B-author"]]


class TestTextFromFeatures:
    def test_without_indices_the_tokens_are_the_text(self):
        assert text_from_features(X, FEATURES, None) is X
        assert text_from_features(X, None, []) is X

    def test_joins_the_columns_of_each_position(self):
        assert text_from_features(X, FEATURES, [0, 1]) == [["Deep learning", "John Smith"]]
        assert text_from_features(X, FEATURES, [1]) == [["learning", "Smith"]]

    def test_a_column_that_a_row_does_not_have_is_empty(self):
        assert text_from_features([["Deep"]], [[["Deep"]]], [0, 1]) == [["Deep "]]

    def test_needs_the_features(self):
        with pytest.raises(ValueError, match=r"columns \[0, 1\]"):
            text_from_features(X, None, [0, 1])

    def test_needs_features_for_every_token(self):
        with pytest.raises(ValueError, match="2 tokens has features for 1"):
            text_from_features(X, [FEATURES[0][:1]], [0, 1])

    def test_tokens_per_position(self):
        assert [tokens_per_position(i) for i in (None, [], [0], [0, 1])] == [1, 1, 1, 2]


class FakeEmbeddings:
    embed_size = 2
    VECTORS = {"deep": [1, 1], "learning": [2, 2], "john": [3, 3], "smith": [4, 4]}

    def get_word_vector(self, word):
        return np.array(self.VECTORS.get(word.lower(), [9, 9]), dtype=np.float32)


class TestConcatenatedEmbeddings:
    def test_one_vector_per_token_of_a_position(self):
        vectors = to_vector_single(["Deep learning", "John Smith"], FakeEmbeddings(), 3, tokens_per_position=2)
        assert vectors.tolist() == [[1, 1, 2, 2], [3, 3, 4, 4], [0, 0, 0, 0]]

    def test_a_missing_token_leaves_zeros_at_its_place(self):
        vectors = to_vector_single(["Deep ", " Smith"], FakeEmbeddings(), 2, tokens_per_position=2)
        assert vectors.tolist() == [[1, 1, 0, 0], [0, 0, 4, 4]]

    def test_a_single_token_per_position_is_unchanged(self):
        assert to_vector_single(["Deep"], FakeEmbeddings(), 1).tolist() == [[1, 1]]


def _config(text_features_indices):
    return ModelConfig(
        architecture="BidLSTM_CRF",
        embeddings_name=None,
        max_sequence_length=10,
        text_features_indices=text_features_indices,
    )


def _chars(text_features_indices):
    """The characters the model is given for the first position, as text."""
    config = _config(text_features_indices)
    preprocessor = Preprocessor()
    preprocessor.fit(text_from_features(X, FEATURES, text_features_indices), Y)
    loader = create_dataloader(X, Y, preprocessor=preprocessor, features=FEATURES, shuffle=False, model_config=config)
    inputs, _ = next(iter(loader))
    index_to_char = {index: char for char, index in preprocessor.vocab_char.items()}
    return "".join(index_to_char[i] for i in inputs["char_input"][0][0].tolist() if i != 0)


class TestLoader:
    def test_the_model_reads_the_token_by_default(self):
        assert _chars(None) == "Deep"

    def test_the_model_reads_the_columns_asked_for(self):
        assert _chars([0, 1]) == "Deep learning"


def _sequence(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    return Sequence(
        "test-model",
        architecture="BidLSTM_CRF",
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
    x = np.array([X[0], X[0][:1]], dtype=object)
    y = np.array([Y[0], Y[0][:1]], dtype=object)
    f = np.array([FEATURES[0], FEATURES[0][:1]], dtype=object)
    return x, y, f


def test_trains_saves_and_tags_with_the_text_of_the_features(tmp_path, monkeypatch):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, text_features_indices=[0, 1])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    # characters that only the second column has are known to the model
    assert "g" in sequence.p.vocab_char and "S" in sequence.p.vocab_char
    sequence.eval(x, y, features=f)
    sequence.save(str(tmp_path))

    loaded = Sequence("test-model", nb_workers=0, device="cpu")
    loaded.load(str(tmp_path))
    assert loaded.model_config.text_features_indices == [0, 1]

    tagged = loaded.tag([X[0]], "raw", features=[FEATURES[0]])
    # the tokens given come back, not the text read from the features
    assert [token for token, _ in tagged[0]] == X[0]
    assert tagged == sequence.tag([X[0]], "raw", features=[FEATURES[0]])


def test_tagging_without_the_features_is_an_error(tmp_path, monkeypatch):
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, text_features_indices=[0, 1])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    with pytest.raises(ValueError, match="features, which were not given"):
        sequence.tag([X[0]], "raw")


def test_the_text_changes_what_the_model_predicts_from(tmp_path, monkeypatch):
    """Same tokens, another second column: the input of the model differs."""
    x, y, f = _arrays()
    sequence = _sequence(tmp_path, monkeypatch, text_features_indices=[0, 1])
    sequence.train(x, y, f_train=f, x_valid=x, y_valid=y, f_valid=f)
    other = [[["Deep", "Smith", "BLOCKSTART"], ["John", "learning", "BLOCKIN"]]]

    def logits(features):
        loader = create_dataloader(
            X, None, preprocessor=sequence.p, features=features, shuffle=False, model_config=sequence.model_config
        )
        inputs, _ = next(iter(loader))
        sequence.model.eval()
        with torch.no_grad():
            return sequence.model(inputs)["logits"]

    assert not torch.allclose(logits(FEATURES), logits(other))


class TestWordsAndPositions:
    def test_every_word_knows_its_position(self):
        from delft.sequenceLabelling.text_features import words_and_positions

        words, positions = words_and_positions(["Deep learning", "John", "John Smith"])
        assert words == ["Deep", "learning", "John", "John", "Smith"]
        assert positions == [0, 0, 1, 2, 2]

    def test_a_position_without_text_keeps_its_place(self):
        from delft.sequenceLabelling.text_features import words_and_positions

        assert words_and_positions(["Deep ", " ", " Smith"]) == (["Deep", "", "Smith"], [0, 1, 2])


@pytest.fixture
def line_tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Deep", "learn", "##ing", "John", "Smith"]
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


def test_with_a_transformer_only_the_first_sub_token_of_a_position_has_its_label(line_tokenizer):
    from unittest.mock import patch

    config = ModelConfig(
        architecture="BERT_CRF",
        embeddings_name=None,
        transformer_name="in-memory",
        max_sequence_length=16,
        text_features_indices=[0, 1],
    )
    preprocessor = Preprocessor(return_chars=False)
    preprocessor.fit(text_from_features(X, FEATURES, [0, 1]), Y)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=line_tokenizer):
        loader = create_dataloader(
            X, Y, preprocessor=preprocessor, features=FEATURES, shuffle=False, model_config=config
        )
        inputs, labels = next(iter(loader))

    index_to_tag = {index: tag for tag, index in preprocessor.vocab_tag.items()}
    # [CLS] Deep learn ##ing John Smith [SEP]
    assert line_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "[CLS]", "Deep", "learn", "##ing", "John", "Smith", "[SEP]",
    ]  # fmt: skip
    assert [index_to_tag[i] for i in labels[0].tolist()] == [
        "<PAD>", "B-title", "<PAD>", "<PAD>", "B-author", "<PAD>", "<PAD>",
    ]  # fmt: skip
    assert inputs["word_start_mask"][0].tolist() == [0, 1, 0, 0, 1, 0, 0]
