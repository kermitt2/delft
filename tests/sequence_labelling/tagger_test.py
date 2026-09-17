import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from delft.sequenceLabelling.tagger import get_entities_with_offsets
from delft.utilities.Tokenizer import tokenizeAndFilter

LOGGER = logging.getLogger(__name__)


def test_get_entities_with_offsets():
    original_string = "(Mo -x 1 T x ) 3 Sb 7 with \uf084 x 0.1"
    tokens = [
        "(",
        "Mo",
        "-",
        "x",
        "1",
        "T",
        "x",
        ")",
        "3",
        "Sb",
        "7",
        "with",
        "\uf084",
        "x",
        "0",
        ".",
        "1",
    ]
    tags = [
        "B-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "I-<formula>",
        "O",
        "O",
        "B-<variable>",
        "B-<value>",
        "I-<value>",
        "I-<value>",
    ]
    # types = [tag.split("-")[-1] for tag in tags]

    offsets = [
        (0, 1),
        (1, 3),
        (4, 5),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 19),
        (20, 21),
        (22, 26),
        (27, 28),
        (29, 30),
        (31, 32),
        (32, 33),
        (33, 34),
    ]

    # spaces = [offsets[offsetIndex][1] != offsets[offsetIndex + 1][0] for offsetIndex in range(0, len(offsets) - 1)]

    for index in range(0, len(offsets)):
        chunk = original_string[offsets[index][0] : offsets[index][1]]

        assert chunk == tokens[index]

    entities_with_offsets = get_entities_with_offsets(tags, offsets)
    # (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    assert len(entities_with_offsets) == 3
    entity0 = entities_with_offsets[0]
    assert entity0[0] == "<formula>"
    entity_text = original_string[entity0[3] : entity0[4] + 1]
    assert entity_text == "(Mo -x 1 T x ) 3 Sb 7"
    assert tokens[entity0[1] : entity0[2]] == tokenizeAndFilter(entity_text)[0]

    entity1 = entities_with_offsets[1]
    assert entity1[0] == "<variable>"
    entity_text = original_string[entity1[3] : entity1[4] + 1]
    assert entity_text == "x"
    assert tokens[entity1[1] : entity1[2]] == tokenizeAndFilter(entity_text)[0]

    entity2 = entities_with_offsets[2]
    assert entity2[0] == "<value>"
    entity_text = original_string[entity2[3] : entity2[4] + 1]
    assert entity_text == "0.1"
    assert tokens[entity2[1] : entity2[2]] == tokenizeAndFilter(entity_text)[0]

    # for item in entities_with_offsets:
    #     type = item[0]
    #     token_start = item[1]
    #     token_end = item[2]
    #     char_start = item[3]
    #     char_end = item[4]
    #
    #     text = ''.join([tokens[idx] + (' ' if spaces[idx] else '') for idx in range(token_start, token_end)])
    #     if text.endswith(' '):
    #         text = text[0:-1]
    #
    #     assert text == original_string[char_start: char_end + 1]


WORDS = ["Jim", "Hensonization", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]


class PositionEchoModel(torch.nn.Module):
    """Answers position p of its input with the tag "P<p>", so that the tag a token
    comes back with names the position its label was read from."""

    def __init__(self, tag_index, key):
        super().__init__()
        self.tag_index = tag_index
        self.key = key

    def decode(self, inputs):
        nb_rows, length = inputs[self.key].shape[:2]
        return [[self.tag_index[f"P{p}"] for p in range(length)] for _ in range(nb_rows)]


def _position_tagger(max_sequence_length, transformer_name=None):
    from delft.sequenceLabelling.config import ModelConfig
    from delft.sequenceLabelling.preprocess import Preprocessor
    from delft.sequenceLabelling.tagger import Tagger

    preprocessor = Preprocessor(return_chars=transformer_name is None)
    preprocessor.fit([WORDS], [[f"P{p}" for p in range(32)]])
    model_config = ModelConfig(
        model_name="test",
        architecture="BERT" if transformer_name else "BidLSTM_CRF",
        embeddings_name=None,
        transformer_name=transformer_name,
        max_sequence_length=max_sequence_length,
        batch_size=3,
    )
    model = PositionEchoModel(preprocessor.vocab_tag, "input_ids" if transformer_name else "char_input")
    return Tagger(model, model_config, preprocessor=preprocessor, device=torch.device("cpu"))


def _tags(tagged):
    return [tag for _, tag in tagged]


class TestTaggerLongSequences:
    """A sequence longer than the model takes is truncated: that is said, not silent."""

    def test_warns_that_the_last_tokens_are_not_labelled(self, caplog):
        with caplog.at_level(logging.WARNING, logger="delft.sequenceLabelling.tagger"):
            tagged = _position_tagger(max_sequence_length=5).tag([WORDS[:3], WORDS, WORDS], "raw")
        assert [[token for token, _ in sequence] for sequence in tagged] == [WORDS[:3], WORDS[:5], WORDS[:5]]
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "2 of 3 sequences" in message and "max_sequence_length=5)" in message
        assert "sequence 1 has 8 tokens and 5 labels" in message

    def test_says_nothing_when_every_sequence_fits(self, caplog):
        with caplog.at_level(logging.WARNING, logger="delft.sequenceLabelling.tagger"):
            tagged = _position_tagger(max_sequence_length=8).tag([WORDS, WORDS[:3]], "raw")
        assert [len(sequence) for sequence in tagged] == [8, 3]
        assert caplog.records == []

    def test_with_a_transformer_the_limit_is_in_sub_tokens(self, caplog, wordpiece_tokenizer):
        # 8 sub-tokens, [CLS] and [SEP] included, hold "Jim Hensonization was a" only
        tagger = _position_tagger(max_sequence_length=8, transformer_name="in-memory")
        with caplog.at_level(logging.WARNING, logger="delft.sequenceLabelling.tagger"):
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=wordpiece_tokenizer):
                tagged = tagger.tag([WORDS], "raw")[0]
        assert [token for token, _ in tagged] == WORDS[:4]
        message = caplog.records[0].getMessage()
        assert "max_sequence_length=8 sub-tokens" in message and "8 tokens and 4 labels" in message


@pytest.mark.parametrize("architecture", ["BidLSTM_CRF", "BidLSTM_ChainCRF"])
def test_tags_with_either_crf_layer(architecture):
    from delft.sequenceLabelling.config import ModelConfig
    from delft.sequenceLabelling.models import get_model
    from delft.sequenceLabelling.preprocess import Preprocessor
    from delft.sequenceLabelling.tagger import Tagger

    preprocessor = Preprocessor(return_chars=True)
    preprocessor.fit([WORDS], [LABELS])
    model_config = ModelConfig(
        model_name="test",
        architecture=architecture,
        embeddings_name=None,
        word_embedding_size=0,
        max_sequence_length=30,
        batch_size=2,
    )
    model_config.char_vocab_size = len(preprocessor.vocab_char)
    model = get_model(model_config, len(preprocessor.vocab_tag), load_pretrained_weights=False)
    tagger = Tagger(model, model_config, preprocessor=preprocessor, device=torch.device("cpu"))

    tagged = tagger.tag([WORDS], "raw")[0]
    assert [token for token, _ in tagged] == WORDS
    assert all(tag in preprocessor.vocab_tag for _, tag in tagged)


@pytest.fixture
def wordpiece_tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Jim", "He", "##nson", "##ization", "was", "a"]
    vocabulary += ["puppet", "##eer", "in", "Mississippi", "today"]
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


class TestTaggerTransformerAlignment:
    """A transformer predicts one label per sub-token: a word takes the label predicted
    at its first sub-token, not the one at the position of the word."""

    # [CLS] Jim He ##nson ##ization was a puppet ##eer in Mississippi today [SEP]
    FIRST_SUB_TOKENS = ["P1", "P2", "P5", "P6", "P7", "P9", "P10", "P11"]

    @staticmethod
    def _tag(tokenizer, max_sequence_length, texts):
        tagger = _position_tagger(max_sequence_length=max_sequence_length, transformer_name="in-memory")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            return tagger.tag(texts, "raw")

    def test_word_takes_the_label_of_its_first_sub_token(self, wordpiece_tokenizer):
        tagged = self._tag(wordpiece_tokenizer, 32, [WORDS])[0]
        assert [token for token, _ in tagged] == WORDS
        assert _tags(tagged) == self.FIRST_SUB_TOKENS

    def test_alignment_holds_in_a_padded_batch(self, wordpiece_tokenizer):
        tagged = self._tag(wordpiece_tokenizer, 32, [WORDS[:3], WORDS])
        assert _tags(tagged[0]) == self.FIRST_SUB_TOKENS[:3]
        assert _tags(tagged[1]) == self.FIRST_SUB_TOKENS


class TestTaggerWorkers:
    """DataLoader worker processes at tagging time.

    GROBID embeds DeLFT through JEP and calls ``tag()`` per sequence, so it
    must be able to request 0 workers, i.e. in-process loading with nothing
    forked off the host interpreter.
    """

    @staticmethod
    def _tagger(**kwargs):
        from delft.sequenceLabelling.tagger import Tagger

        model_config = MagicMock()
        model_config.batch_size = 20
        return Tagger(
            MagicMock(),
            model_config,
            preprocessor=MagicMock(),
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_defaults_to_in_process_loading(self):
        assert self._tagger().nb_workers == 0

    def test_clamps_negative_worker_count(self):
        assert self._tagger(nb_workers=-1).nb_workers == 0
        assert self._tagger(nb_workers=None).nb_workers == 0

    @pytest.mark.parametrize("nb_workers", [0, 3])
    def test_passes_nb_workers_to_the_dataloader(self, nb_workers):
        tagger = self._tagger(nb_workers=nb_workers)
        with patch("delft.sequenceLabelling.tagger.create_dataloader", return_value=[]) as create_dataloader:
            tagger.tag([["some", "tokens"]], "raw")
        assert create_dataloader.call_args.kwargs["num_workers"] == nb_workers


class TestSequenceTagWorkers:
    @staticmethod
    def _sequence(nb_workers, explicit):
        from delft.sequenceLabelling.wrapper import Sequence

        sequence = Sequence.__new__(Sequence)
        sequence.model = MagicMock()
        sequence.model_config = MagicMock()
        sequence.embeddings = None
        sequence.p = MagicMock()
        sequence.device = torch.device("cpu")
        sequence.nb_workers = nb_workers
        sequence.nb_workers_explicit = explicit
        return sequence

    @staticmethod
    def _tag(sequence, **kwargs):
        with patch("delft.sequenceLabelling.tagger.Tagger") as tagger_class:
            sequence.tag([["some", "tokens"]], "raw", **kwargs)
        return tagger_class.call_args.kwargs["nb_workers"]

    def test_defaults_to_no_worker_when_unset(self):
        # the constructor default (min(4, cpu_count - 1)) is a training
        # setting: workers are respawned on every tag() call.
        assert self._tag(self._sequence(nb_workers=4, explicit=False)) == 0

    def test_uses_the_constructor_value_when_set(self):
        assert self._tag(self._sequence(nb_workers=2, explicit=True)) == 2

    def test_zero_survives_the_constructor(self):
        assert self._tag(self._sequence(nb_workers=0, explicit=True)) == 0

    def test_per_call_value_wins(self):
        assert self._tag(self._sequence(nb_workers=4, explicit=True), nb_workers=0) == 0


class TestTaggerReuse:
    """GROBID calls tag() per sequence, so the Tagger is built once and kept.

    It must still be rebuilt whenever something it closes over is replaced,
    otherwise a load() or a fold selection would keep tagging with the old
    model.
    """

    @staticmethod
    def _sequence():
        from delft.sequenceLabelling.wrapper import Sequence

        sequence = Sequence.__new__(Sequence)
        sequence.model = MagicMock()
        sequence.model_config = MagicMock()
        sequence.model_config.batch_size = 20
        sequence.embeddings = None
        sequence.p = MagicMock()
        sequence.device = torch.device("cpu")
        sequence.nb_workers = 0
        sequence.nb_workers_explicit = True
        sequence._tagger = None
        return sequence

    def test_reuses_the_same_tagger_across_calls(self):
        sequence = self._sequence()
        assert sequence._get_tagger(0) is sequence._get_tagger(0)

    def test_rebuilds_when_the_model_is_replaced(self):
        sequence = self._sequence()
        first = sequence._get_tagger(0)
        sequence.model = MagicMock()
        assert sequence._get_tagger(0) is not first

    def test_rebuilds_when_the_preprocessor_is_replaced(self):
        sequence = self._sequence()
        first = sequence._get_tagger(0)
        sequence.p = MagicMock()
        assert sequence._get_tagger(0) is not first

    def test_rebuilds_for_a_different_worker_count(self):
        sequence = self._sequence()
        first = sequence._get_tagger(0)
        assert sequence._get_tagger(2) is not first

    def test_takes_the_device_resolved_by_the_wrapper(self):
        sequence = self._sequence()
        with patch("delft.utilities.Utilities.pick_device") as pick_device:
            tagger = sequence._get_tagger(0)
        pick_device.assert_not_called()
        assert tagger.device == torch.device("cpu")


class TestSequenceWorkerConfiguration:
    def test_keeps_an_explicit_zero(self):
        from delft.sequenceLabelling.wrapper import Sequence

        sequence = Sequence("test-model", architecture="BidLSTM_CRF", nb_workers=0)
        assert sequence.nb_workers == 0
        assert sequence.nb_workers_explicit is True

    def test_defaults_to_a_training_worker_pool(self):
        from delft.sequenceLabelling.wrapper import Sequence

        sequence = Sequence("test-model", architecture="BidLSTM_CRF")
        assert sequence.nb_workers >= 1
        assert sequence.nb_workers_explicit is False
