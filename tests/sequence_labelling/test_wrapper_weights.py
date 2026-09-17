"""A model saved by the wrapper loads back the same, whichever format its weights are in."""

import pytest
import torch

from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.trainer import DEFAULT_WEIGHT_FILE_NAME
from delft.sequenceLabelling.wrapper import Sequence
from delft.utilities.weights import SAFETENSORS_WEIGHT_FILE_NAME

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]


def _sequence(architecture):
    sequence = Sequence("test-model", architecture=architecture, embeddings_name=None, device="cpu")
    sequence.p = Preprocessor(return_chars=True)
    sequence.p.fit([WORDS], [LABELS])
    sequence.model_config.char_vocab_size = len(sequence.p.vocab_char)
    sequence.model = get_model(sequence.model_config, len(sequence.p.vocab_tag), load_pretrained_weights=False)
    return sequence


@pytest.mark.parametrize("architecture", ["BidLSTM_CRF", "BidLSTM_ChainCRF"])
@pytest.mark.parametrize("weight_file", [DEFAULT_WEIGHT_FILE_NAME, SAFETENSORS_WEIGHT_FILE_NAME])
def test_saved_model_loads_back_the_same(tmp_path, architecture, weight_file):
    saved = _sequence(architecture)
    saved.save(str(tmp_path), weight_file=weight_file)
    assert (tmp_path / "test-model" / weight_file).is_file()

    # loaded without naming the format, as GROBID and the applications do
    loaded = Sequence("test-model", device="cpu")
    loaded.load(str(tmp_path))

    state, loaded_state = saved.model.state_dict(), loaded.model.state_dict()
    assert state.keys() == loaded_state.keys()
    for name in state:
        assert torch.equal(state[name], loaded_state[name]), name


@pytest.mark.parametrize("weight_file", [DEFAULT_WEIGHT_FILE_NAME, SAFETENSORS_WEIGHT_FILE_NAME])
def test_loaded_model_tags_like_the_saved_one(tmp_path, weight_file):
    saved = _sequence("BidLSTM_CRF")
    saved.save(str(tmp_path), weight_file=weight_file)
    loaded = Sequence("test-model", device="cpu")
    loaded.load(str(tmp_path))
    assert loaded.tag([WORDS], "raw") == saved.tag([WORDS], "raw")


def test_default_format_is_unchanged(tmp_path):
    _sequence("BidLSTM_CRF").save(str(tmp_path))
    assert sorted(path.name for path in (tmp_path / "test-model").iterdir()) == [
        "config.json",
        DEFAULT_WEIGHT_FILE_NAME,
        "preprocessor.json",
    ]
