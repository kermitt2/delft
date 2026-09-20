"""
The architectures with a casing channel, and BidLSTM_CNN_CRF, train, save, load and tag
through the wrapper: the batch held no casing input, and the CNN CRF fed its CRF the
output of its dense layer rather than one emission per tag.
"""

import numpy as np
import pytest

from delft.sequenceLabelling.preprocess import architecture_uses_casing
from delft.sequenceLabelling.wrapper import Sequence

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
X = np.array([WORDS, WORDS[:3], WORDS[2:], WORDS[:5]], dtype=object)
Y = np.array([LABELS, LABELS[:3], LABELS[2:], LABELS[:5]], dtype=object)


@pytest.mark.parametrize("architecture", ["BidLSTM_CNN", "BidLSTM_CNN_CRF", "BidLSTM_CRF_CASING"])
def test_trains_saves_loads_and_tags(tmp_path, monkeypatch, architecture):
    monkeypatch.chdir(tmp_path)
    sequence = Sequence(
        "test-model",
        architecture=architecture,
        embeddings_name=None,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
    )
    sequence.train(X, Y, x_valid=X, y_valid=Y)
    assert sequence.p.return_casing == architecture_uses_casing(architecture)
    sequence.eval(X, Y)
    sequence.save(str(tmp_path))

    loaded = Sequence("test-model", device="cpu", nb_workers=0)
    loaded.load(str(tmp_path))
    assert loaded.p.return_casing == architecture_uses_casing(architecture)
    tagged = loaded.tag([WORDS, WORDS[2:]], "list")
    assert [word for word, _ in tagged[0]] == WORDS
    assert [word for word, _ in tagged[1]] == WORDS[2:]
    # one label of the model per word; an untrained CRF may pick the padding tag on a
    # real token, as BidLSTM_CRF does after one epoch, which is not a decoding fault
    assert all(label in sequence.p.vocab_tag for _, label in tagged[0])


def test_the_casing_architectures():
    assert architecture_uses_casing("BidLSTM_CNN") and architecture_uses_casing("BidLSTM_CRF_CASING")
    assert not architecture_uses_casing("BidLSTM_CNN_CRF") and not architecture_uses_casing("BidLSTM_CRF")
