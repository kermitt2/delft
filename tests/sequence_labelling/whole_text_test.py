"""A sequence sub-tokenized as one text, its sub-tokens aligned on its words by their offsets (issue #128)."""

from unittest.mock import patch

import pytest

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import BERTPreprocessor, Preprocessor
from delft.sequenceLabelling.tagger import word_positions
from delft.sequenceLabelling.whole_text import align_offsets, join_words, subtokenize_whole_text
from delft.sequenceLabelling.windows import subtoken_costs
from delft.sequenceLabelling.wrapper import Sequence

# "%)" is one sub-token of the tokenizer below, spanning the two last words
WORDS = ["Deep", "(", "John", ",", "50", "%", ")"]
LABELS = ["B-title", "O", "B-author", "O", "B-num", "I-num", "O"]
TEXT = "Deep (John, 50%)"


class TestJoinWords:
    def test_no_space_before_a_closing_punctuation_and_none_after_an_opening_one(self):
        text, spans = join_words(WORDS)
        assert text == TEXT
        assert [text[start:end] for start, end in spans] == WORDS

    def test_words_are_separated_by_a_space_otherwise(self):
        assert join_words(["a", "b", "-", "c"])[0] == "a b - c"

    def test_an_empty_word_has_an_empty_span_and_no_space_of_its_own(self):
        text, spans = join_words(["a", "", "b"])
        assert text == "a b"
        assert spans == [(0, 1), (1, 1), (2, 3)]

    def test_no_words(self):
        assert join_words([]) == ("", [])


class TestAlignOffsets:
    SPANS = [(0, 4), (5, 6), (6, 10), (10, 11), (12, 14), (14, 15), (15, 16)]  # the words of TEXT

    def test_every_sub_token_belongs_to_the_word_of_its_characters(self):
        # <s> Deep ( Jo hn , 50 %) </s>
        offsets = [(0, 0), (0, 4), (5, 6), (6, 8), (8, 10), (10, 11), (12, 14), (14, 16), (0, 0)]
        special = [True, False, False, False, False, False, False, False, True]
        words, started = align_offsets(offsets, self.SPANS, special)
        assert words == [None, 0, 1, 2, 2, 3, 4, 5, None]
        assert started == [[], [0], [1], [2], [], [3], [4], [5, 6], []]

    def test_offsets_that_include_the_leading_space_belong_to_the_word_after_it(self):
        """A SentencePiece tokenizer may count the space before a word in its offsets."""
        offsets = [(0, 4), (4, 6), (6, 10)]
        words, started = align_offsets(offsets, self.SPANS, [False] * 3)
        assert words == [0, 1, 2]
        assert started == [[0], [1], [2]]

    def test_a_word_the_tokenizer_dropped_starts_on_the_next_sub_token(self):
        offsets = [(0, 4), (6, 10)]  # no sub-token for "("
        words, started = align_offsets(offsets, self.SPANS, [False] * 2)
        assert words == [0, 2]
        assert started == [[0], [1, 2]]

    def test_a_sub_token_of_whitespace_alone_belongs_to_no_word(self):
        words, started = align_offsets([(0, 4), (4, 5), (5, 6)], self.SPANS, [False] * 3)
        assert words == [0, None, 1]
        assert started == [[0], [], [1]]


class TestWordPositions:
    def test_the_words_a_sub_token_starts_all_take_its_position(self):
        assert word_positions([0, 1, 1, 0, 2, 0]) == [1, 2, 4, 4]
        assert word_positions([]) == []


def _merges(piece):
    """The BPE merges that build ``piece`` from its characters, left to right."""
    return [(piece[: i + 1], piece[i + 1]) for i in range(len(piece) - 1)]


def _byte_level_tokenizer():
    """
    A RoBERTa-like byte-level BPE built in memory, 'Ġ' standing for a leading space. It
    makes "ĠJohn" of " John" but "Jo" and "hn" of "John", and "%)" is one sub-token.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    pieces = ["ĠDeep", "Ġ(", "ĠJohn", "Jo", "hn", "Ġ,", "Ġ50", "%)", "Ġ%", "Ġ)"]
    merges = [merge for piece in pieces for merge in _merges(piece)]
    characters = sorted({character for piece in pieces for character in piece})
    tokens = ["<pad>", "<s>", "</s>", "<unk>"] + characters + [left + right for left, right in merges]
    vocabulary = {}
    for token in tokens:
        vocabulary.setdefault(token, len(vocabulary))
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=merges, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


@pytest.fixture
def tokenizer():
    return _byte_level_tokenizer()


class TestSubtokenizeWholeText:
    def test_the_text_is_sub_tokenized_as_a_whole_and_aligned_on_the_words(self, tokenizer):
        encoded, words, started = subtokenize_whole_text(tokenizer, WORDS)
        assert tokenizer.convert_ids_to_tokens(encoded.input_ids) == [
            "<s>",
            "ĠDeep",
            "Ġ(",
            "Jo",
            "hn",
            ",",
            "Ġ50",
            "%)",
            "</s>",
        ]
        assert words == [None, 0, 1, 2, 2, 3, 4, 5, None]
        assert started == [[], [0], [1], [2], [], [3], [4], [5, 6], []]

    def test_word_by_word_every_word_gets_a_space_before_it(self, tokenizer):
        """What whole text tokenization is for: "(John," and "50%)" are never seen as such."""
        word_by_word = tokenizer(WORDS, is_split_into_words=True)
        assert tokenizer.convert_ids_to_tokens(word_by_word.input_ids)[1:-1] == [
            "ĠDeep",
            "Ġ(",
            "ĠJohn",
            "Ġ,",
            "Ġ50",
            "Ġ%",
            "Ġ)",
        ]

    def test_the_sub_token_costs_of_the_words(self, tokenizer):
        assert subtoken_costs(tokenizer, whole_text=True)(WORDS) == [1, 1, 2, 1, 1, 1, 0]
        assert subtoken_costs(tokenizer)(WORDS) == [1, 1, 1, 1, 1, 1, 1]
        assert subtoken_costs(tokenizer, whole_text=True)([]) == []


class TestPreprocessor:
    def test_labels_and_word_starts_are_aligned_on_the_sub_tokens(self, tokenizer):
        preprocessor = BERTPreprocessor(tokenizer, whole_text=True)
        *_, word_starts, labels, _ = preprocessor.tokenize_and_align_features_and_labels(
            [WORDS], [None], None, [LABELS], maxlen=12
        )
        assert labels[0] == ["<PAD>", "B-title", "O", "B-author", "<PAD>", "O", "B-num", "I-num"] + ["<PAD>"] * 4
        assert word_starts[0] == [0, 1, 1, 1, 0, 1, 1, 2, 0, 0, 0, 0]

    def test_word_by_word_as_before_without_the_option(self, tokenizer):
        preprocessor = BERTPreprocessor(tokenizer)
        preprocessor.is_BPE_SP = True
        *_, word_starts, labels, _ = preprocessor.tokenize_and_align_features_and_labels(
            [WORDS], [None], None, [LABELS], maxlen=12
        )
        assert [label for label in labels[0] if label != "<PAD>"] == LABELS
        assert word_starts[0] == [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def _config(**kwargs):
    options = dict(architecture="BERT_CRF", embeddings_name=None, transformer_name="in-memory", max_sequence_length=12)
    options.update(kwargs)
    return ModelConfig(**options)


class TestLoader:
    def test_the_batch_counts_the_words_starting_on_each_sub_token(self, tokenizer):
        preprocessor = Preprocessor(return_chars=False)
        preprocessor.fit([WORDS], [LABELS])
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            loader = create_dataloader(
                [WORDS],
                [LABELS],
                preprocessor=preprocessor,
                shuffle=False,
                model_config=_config(whole_text_tokenization=True),
            )
            inputs, labels = next(iter(loader))
        assert inputs["word_start_mask"][0].tolist() == [0, 1, 1, 1, 0, 1, 1, 2, 0]
        assert len(word_positions(inputs["word_start_mask"][0].tolist())) == len(WORDS)

    def test_the_option_is_saved_in_the_config_of_the_model(self, tmp_path):
        config = _config(whole_text_tokenization=True)
        config.save(str(tmp_path / "config.json"))
        assert ModelConfig.load(str(tmp_path / "config.json")).whole_text_tokenization is True
        assert ModelConfig.load(str(tmp_path / "config.json")) is not None
        assert _config().whole_text_tokenization is False


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


class TestSequence:
    def test_trained_and_tagging_as_one_text_every_word_gets_a_label(self, tmp_path, monkeypatch, tokenizer):
        monkeypatch.chdir(tmp_path)
        sequence = Sequence(
            "test-model",
            architecture="BERT_CRF",
            embeddings_name=None,
            transformer_name="in-memory",
            max_sequence_length=12,
            max_epoch=1,
            batch_size=2,
            early_stop=False,
            nb_workers=0,
            device="cpu",
            whole_text_tokenization=True,
        )
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
        ):
            sequence.train([WORDS, WORDS[:3]], [LABELS, LABELS[:3]])
            assert sequence.model_config.whole_text_tokenization
            sequence.eval([WORDS], [LABELS])
            tagged = sequence.tag([WORDS, WORDS[2:]], "list")
            assert [word for word, _ in tagged[0]] == WORDS
            assert [word for word, _ in tagged[1]] == WORDS[2:]
            # the two words of the sub-token "%)" share its label
            assert tagged[0][5][1] == tagged[0][6][1]
            sequence.save(str(tmp_path))
            loaded = Sequence("test-model", device="cpu")
            loaded.load(str(tmp_path))
            assert loaded.model_config.whole_text_tokenization
            assert loaded.tag([WORDS], "list") == sequence.tag([WORDS], "list")
