"""Training sequences with words glued together, as GROBID reads the text of some PDFs."""

import numpy as np
import pytest

from delft.sequenceLabelling.augmentation import REFERENCE_SEGMENTER_COLUMNS, GluedWords
from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.wrapper import Sequence


def _row(line):
    """Token, features (the label left out) of a line of a GROBID reference-segmenter file."""
    columns = line.split()
    return columns[0], columns[1:]


# as GROBID writes them, the last row from Prog Lipid Res 2011 50(1-4) 14-27, whose PDF has "UngerR"
UNGER = _row(
    "Unger unger U Un Ung Unge r er ger nger LINESTART ALIGNEDLEFT INITCAP NODIGIT 0 1 0 0 0 0 0 0 NOPUNCT 0 8 BLOCKSTART 8 0"
)
R = _row("R r R R R R R R R R LINEIN ALIGNEDLEFT ALLCAP NODIGIT 1 0 0 0 0 0 0 0 NOPUNCT 0 8 BLOCKIN 8 0")
DOT = _row(". . . . . . . . . . LINEIN ALIGNEDLEFT ALLCAP NODIGIT 1 0 0 0 0 0 0 0 DOT 0 8 BLOCKIN 8 0")
UNGER_R = _row(
    "UngerR ungerr U Un Ung Unge R rR erR gerR LINESTART ALIGNEDLEFT INITCAP NODIGIT 0 0 0 0 0 0 0 0 NOPUNCT 0 8 BLOCKSTART 8 0"
)

ACTA = _row(
    "Acta acta A Ac Act Acta a ta cta Acta LINEIN LINEINDENT INITCAP NODIGIT 0 0 0 0 0 0 0 0 NOPUNCT 10 5 BLOCKIN no 0"
)
VOLUME = _row(
    "1801 1801 1 18 180 1801 1 01 801 1801 LINEIN LINEINDENT NOCAPS ALLDIGIT 0 0 0 0 0 0 0 0 NOPUNCT 10 5 BLOCKIN no 0"
)
YEAR = _row(
    "2010 2010 2 20 201 2010 0 10 010 2010 LINEEND LINEINDENT NOCAPS ALLDIGIT 0 0 0 0 0 1 0 0 NOPUNCT 10 5 BLOCKIN no 0"
)
ACTA_1801_2010 = _row(
    "Acta18012010 acta18012010 A Ac Act Acta 0 10 010 2010 LINEEND LINEINDENT INITCAP CONTAINSDIGITS "
    + "0 0 0 0 0 1 0 0 NOPUNCT 10 5 BLOCKIN no 0"
)


def _glue(rows, labels, probability=1.0):
    tokens = [token for token, _ in rows]
    features = [list(features) for _, features in rows]
    return GluedWords(probability, REFERENCE_SEGMENTER_COLUMNS, pair_probability=1.0)(tokens, labels, features)


class TestGluedWords:
    def test_a_glued_word_has_the_features_grobid_gives_it(self):
        tokens, labels, features = _glue([UNGER, R, DOT], ["B-<reference>", "I-<reference>", "I-<reference>"])

        assert tokens == ["UngerR", "."]
        assert labels == ["B-<reference>", "I-<reference>"]
        assert features == [UNGER_R[1], DOT[1]]

    def test_words_and_numbers_glue_up_to_the_end_of_their_line(self):
        tokens, _, features = _glue([ACTA, VOLUME, YEAR], ["I-<reference>"] * 3)

        assert tokens == ["Acta18012010"]
        assert features == [ACTA_1801_2010[1]]

    def test_a_word_is_not_glued_to_the_one_that_starts_a_field(self):
        tokens, labels, _ = _glue([UNGER, R], ["B-<reference>", "B-<reference>"])

        assert tokens == ["Unger", "R"]
        assert labels == ["B-<reference>", "B-<reference>"]

    def test_a_word_is_not_glued_to_one_of_another_field(self):
        assert _glue([UNGER, R], ["B-<label>", "I-<reference>"])[0] == ["Unger", "R"]

    def test_a_word_is_not_glued_to_the_one_starting_the_next_line(self):
        assert _glue([YEAR, UNGER], ["I-<reference>"] * 2)[0] == ["2010", "Unger"]

    def test_punctuation_is_not_glued(self):
        assert _glue([R, DOT, R], ["I-<reference>"] * 3)[0] == ["R", ".", "R"]

    def test_with_a_probability_of_zero_the_sequence_is_kept(self):
        rows = [UNGER, R, DOT]
        labels = ["B-<reference>", "I-<reference>", "I-<reference>"]

        tokens, glued_labels, features = _glue(rows, labels, probability=0)

        assert tokens == ["Unger", "R", "."]
        assert glued_labels == labels
        assert features == [UNGER[1], R[1], DOT[1]]

    def test_the_sequence_given_is_not_changed(self):
        tokens = ["Unger", "R"]
        features = [list(UNGER[1]), list(R[1])]

        GluedWords(1, REFERENCE_SEGMENTER_COLUMNS, pair_probability=1)(tokens, ["O", "O"], features)

        assert tokens == ["Unger", "R"]
        assert features == [UNGER[1], R[1]]

    @pytest.mark.parametrize("probability", [-0.1, 1.5])
    def test_rejects_a_probability_out_of_range(self, probability):
        with pytest.raises(ValueError, match="between 0 and 1"):
            GluedWords(probability, REFERENCE_SEGMENTER_COLUMNS)


WORDS = ["Jim", "Henson", "was", "a", "puppeteer"]
LABELS = ["B-per", "I-per", "O", "O", "O"]


class Recorder:
    """An augmentation that glues the first two words of every sequence, and keeps what it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, tokens, labels, features):
        self.seen.append(list(tokens))
        return [tokens[0] + tokens[1]] + list(tokens[2:]), [labels[0]] + list(labels[2:]), features


class TestTheTrainingSequencesAreAugmented:
    def test_the_loader_gives_the_augmented_sequences(self):
        preprocessor = Preprocessor()
        preprocessor.fit([WORDS], [LABELS])
        loader = create_dataloader(
            [WORDS],
            [LABELS],
            batch_size=1,
            preprocessor=preprocessor,
            shuffle=False,
            model_config=ModelConfig(architecture="BidLSTM_CRF", embeddings_name=None, max_sequence_length=10),
            augment=Recorder(),
        )

        inputs, labels = next(iter(loader))

        assert inputs["length"].tolist() == [4]
        assert labels.shape == (1, 4)

    def test_sequences_to_label_are_not_augmented(self):
        preprocessor = Preprocessor()
        preprocessor.fit([WORDS], [LABELS])
        recorder = Recorder()
        loader = create_dataloader(
            [WORDS],
            None,
            batch_size=1,
            preprocessor=preprocessor,
            shuffle=False,
            model_config=ModelConfig(architecture="BidLSTM_CRF", embeddings_name=None, max_sequence_length=10),
            augment=recorder,
        )

        inputs, _ = next(iter(loader))

        assert inputs["length"].tolist() == [5]
        assert recorder.seen == []

    def test_only_the_training_set_is_augmented(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorder = Recorder()
        sequence = Sequence(
            "test-model",
            architecture="BidLSTM_CRF",
            embeddings_name=None,
            max_epoch=1,
            batch_size=1,
            early_stop=False,
            nb_workers=0,
            device="cpu",
            augment=recorder,
        )
        valid = ["Kermit", "was", "a", "frog"]

        sequence.train(
            np.array([WORDS], dtype=object),
            np.array([LABELS], dtype=object),
            x_valid=np.array([valid], dtype=object),
            y_valid=np.array([["B-per", "O", "O", "O"]], dtype=object),
        )

        assert recorder.seen == [WORDS]

    def test_augmenting_with_a_transformer_is_an_error(self):
        with pytest.raises(ValueError, match="cannot be augmented with a transformer"):
            create_dataloader(
                [WORDS],
                [LABELS],
                preprocessor=Preprocessor(),
                model_config=ModelConfig(architecture="BERT", transformer_name="bert-base-cased"),
                augment=Recorder(),
            )
