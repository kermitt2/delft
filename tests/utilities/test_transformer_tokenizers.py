"""
The tokenizer of a transformer is loaded once for the process, and not for every call to
tag or classify, which also asked the Hugging Face Hub about it every time.
"""

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from delft.sequenceLabelling.wrapper import Sequence
from delft.textClassification.wrapper import Classifier
from delft.utilities.transformer_tokenizers import (
    call_tokenizer,
    clear_tokenizers,
    get_tokenizer,
    loses_word_starts,
    mark_word_starts,
)

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
X = np.array([WORDS, WORDS[:3], WORDS[2:], WORDS[:5]], dtype=object)
Y = np.array([LABELS, LABELS[:3], LABELS[2:], LABELS[:5]], dtype=object)
TEXTS = ["Jim was a puppeteer", "Henson in Mississippi", "a puppeteer today", "Jim Henson"]
CLASSES = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)


def _tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + WORDS
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


def byte_level_tokenizer(**options):
    """
    A byte-level BPE tokenizer (RoBERTa, GPT-2) built in memory, given as the generic
    PreTrainedTokenizerFast that does not know add_prefix_space, as scilons/SciLaD-M-custom
    has it: by default its pre-tokenizer adds no space before a word.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator([" ".join(WORDS)] * 20, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        **options,
    )


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


class TestGetTokenizer:
    def test_loaded_once_for_the_same_arguments(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: object()) as load:
            first = get_tokenizer("some/model", add_prefix_space=True)
            assert get_tokenizer("some/model", add_prefix_space=True) is first
        load.assert_called_once_with("some/model", add_prefix_space=True)

    def test_other_arguments_give_another_tokenizer(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: object()) as load:
            with_space = get_tokenizer("some/model", add_prefix_space=True)
            assert get_tokenizer("some/model") is not with_space
            assert get_tokenizer("another/model", add_prefix_space=True) is not with_space
        assert load.call_count == 3

    def test_loaded_again_once_cleared(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: object()) as load:
            first = get_tokenizer("some/model")
            clear_tokenizers()
            assert get_tokenizer("some/model") is not first
        assert load.call_count == 2

    def test_a_load_that_fails_is_tried_again(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=[OSError("no network"), "tokenizer"]):
            with pytest.raises(OSError):
                get_tokenizer("some/model")
            assert get_tokenizer("some/model") == "tokenizer"


def test_a_shared_tokenizer_is_called_by_one_thread_at_a_time():
    """
    A call sets its truncation and padding on the tokenizer, where a call of another
    thread running at the same time finds them instead of its own: with the tokenizer of
    SciBERT, one sequence in ten came back cut at the wrong length, or not cut.
    """
    inside = []
    overlaps = []

    def tokenizer(thread):
        inside.append(thread)
        time.sleep(0.02)  # the tokenizers release the interpreter lock while they encode
        if len(inside) > 1:
            overlaps.append(list(inside))
        inside.remove(thread)
        return thread

    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(call_tokenizer(tokenizer, i))) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlaps == []
    assert sorted(results) == list(range(6))


class TestTagging:
    def test_the_tokenizer_is_loaded_once_whatever_the_number_of_calls(self, tmp_path, monkeypatch):
        """Every call to tag loaded it again: 0.5 to 5 seconds, and a request to the Hub, per sequence."""
        monkeypatch.chdir(tmp_path)
        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _tokenizer()) as load,
            patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
        ):
            sequence = Sequence(
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
            )
            sequence.train(X, Y, x_valid=X, y_valid=Y)
            sequence.eval(X, Y)
            first = sequence.tag([WORDS], "list")
            for _ in range(3):
                assert sequence.tag([WORDS], "list") == first
            assert [word for word, _ in first[0]] == WORDS
        load.assert_called_once_with("in-memory", add_prefix_space=True)

    def test_and_once_with_windows_which_tokenize_twice(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _tokenizer()) as load,
            patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
        ):
            sequence = Sequence(
                "test-model",
                architecture="BERT_CRF",
                embeddings_name=None,
                transformer_name="in-memory",
                max_sequence_length=6,
                window_stride=2,
                max_epoch=1,
                batch_size=2,
                early_stop=False,
                nb_workers=0,
                device="cpu",
            )
            sequence.train(X, Y)
            for _ in range(3):
                tagged = sequence.tag([WORDS], "list")
                assert [word for word, _ in tagged[0]] == WORDS
        assert load.call_count == 1


def test_a_classifier_loads_its_tokenizer_once_whatever_the_number_of_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _tokenizer()) as load,
        patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
    ):
        classifier = Classifier(
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
        classifier.train(TEXTS, CLASSES)
        classifier.eval(TEXTS, CLASSES)
        first = classifier.predict(TEXTS, output_format="array")
        for _ in range(3):
            assert np.allclose(classifier.predict(TEXTS, output_format="array"), first)
    load.assert_called_once_with("in-memory")


class TestWordStarts:
    """
    A byte-level BPE tokenizer given words apart must mark the start of each of them, or
    the transformer reads "Jim Henson was" as "JimHensonwas".
    """

    def test_a_byte_level_tokenizer_without_prefix_space_loses_word_starts(self):
        tokenizer = byte_level_tokenizer()
        assert loses_word_starts(tokenizer)
        pieces = tokenizer.convert_ids_to_tokens(
            tokenizer(WORDS, is_split_into_words=True, add_special_tokens=False)["input_ids"]
        )
        assert not any(piece.startswith("Ġ") for piece in pieces)

    def test_marked_word_starts_match_the_text_written_whole(self):
        tokenizer = byte_level_tokenizer()
        assert mark_word_starts(tokenizer)
        assert not loses_word_starts(tokenizer)
        encoding = tokenizer(WORDS, is_split_into_words=True, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()
        firsts = [piece for i, (piece, word) in enumerate(zip(pieces, word_ids)) if i == 0 or word != word_ids[i - 1]]
        assert len(firsts) == len(WORDS) and all(piece.startswith("Ġ") for piece in firsts)
        whole = tokenizer(" " + " ".join(WORDS), add_special_tokens=False)["input_ids"]
        assert encoding["input_ids"] == whole

    def test_marking_twice_changes_nothing_more(self):
        tokenizer = byte_level_tokenizer()
        assert mark_word_starts(tokenizer)
        assert not mark_word_starts(tokenizer)

    def test_a_wordpiece_tokenizer_is_left_as_it_is(self):
        tokenizer = _tokenizer()
        before = tokenizer.backend_tokenizer.to_str()
        assert not loses_word_starts(tokenizer)
        assert not mark_word_starts(tokenizer)
        assert tokenizer.backend_tokenizer.to_str() == before

    def test_get_tokenizer_marks_word_starts_when_asked_for_a_prefix_space(self):
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: byte_level_tokenizer()):
            assert not loses_word_starts(get_tokenizer("some/roberta", add_prefix_space=True))
            assert loses_word_starts(get_tokenizer("some/roberta"))
