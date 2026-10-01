"""
The architectures with a casing channel, and BidLSTM_CNN_CRF, train, save, load and tag
through the wrapper: the batch held no casing input, and the CNN CRF fed its CRF the
output of its dense layer rather than one emission per tag.
"""

import numpy as np
import pytest
import torch

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import get_model
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


@pytest.mark.parametrize(
    "architecture", ["BidLSTM", "BidLSTM_CRF", "BidLSTM_CNN", "BidLSTM_CNN_CRF", "BidLSTM_CRF_CASING", "BidGRU_CRF"]
)
def test_the_scores_of_a_sentence_do_not_change_with_the_padding_of_its_batch(architecture):
    """BidLSTM_CNN ran its LSTM over the padding: the backward one started from it."""
    torch.manual_seed(0)
    config = ModelConfig(architecture=architecture, embeddings_name=None, word_embedding_size=8)
    config.char_vocab_size = 20
    config.case_vocab_size = 8
    model = get_model(config, 5, load_pretrained_weights=False).eval()

    length, padded_length, characters = 3, 7, 6
    words = torch.randn(1, length, 8)
    chars = torch.randint(1, 20, (1, length, characters))
    casing = torch.randint(1, 8, (1, length))

    def inputs(total):
        """The sentence, first of a batch whose other sentence has ``total`` tokens."""
        batch = {
            "word_input": torch.randn(2, total, 8),
            "char_input": torch.randint(1, 20, (2, total, characters)),
            "casing_input": torch.randint(1, 8, (2, total)),
        }
        for name, value in (("word_input", words), ("char_input", chars), ("casing_input", casing)):
            batch[name][0] = 0
            batch[name][0, :length] = value[0]
        return batch

    with torch.no_grad():
        alone = model(inputs(length))["logits"][0]
        padded = model(inputs(padded_length))["logits"][0, :length]
    torch.testing.assert_close(padded, alone, rtol=1e-5, atol=1e-6)
