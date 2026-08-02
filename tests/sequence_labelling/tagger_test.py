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
