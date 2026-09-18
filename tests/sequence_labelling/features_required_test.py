"""A model trained with features cannot do without them."""

import numpy as np
import pytest

from delft.sequenceLabelling.wrapper import Sequence

X = np.array([["Jim", "Henson", "was"], ["a", "puppeteer"]], dtype=object)
Y = np.array([["B-per", "I-per", "O"], ["O", "O"]], dtype=object)
FEATURES = np.array(
    [[["Jim", "UP"], ["Henson", "UP"], ["was", "LOW"]], [["a", "LOW"], ["puppeteer", "LOW"]]], dtype=object
)


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sequence = Sequence(
        "test-model",
        architecture="BidLSTM_CRF_FEATURES",
        embeddings_name=None,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
    )
    sequence.train(X, Y, f_train=FEATURES, x_valid=X, y_valid=Y, f_valid=FEATURES)
    return sequence


def test_labels_with_features(sequence):
    tagged = sequence.tag([list(X[0])], "raw", features=[list(FEATURES[0])])
    assert [token for token, _ in tagged[0]] == list(X[0])


def test_labelling_without_features_is_an_error(sequence):
    """Zeros of another shape stood in for them, and the model labelled from that."""
    with pytest.raises(ValueError, match="trained with features"):
        sequence.tag([list(X[0])], "raw")


def test_evaluating_without_features_is_an_error(sequence):
    with pytest.raises(ValueError, match="trained with features"):
        sequence.eval(X, Y)
