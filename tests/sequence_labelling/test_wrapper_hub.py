"""Loading a model that is on the Hugging Face Hub, played here by a local directory."""

import os
import shutil
from fnmatch import fnmatch

import pytest
import torch

from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.wrapper import Sequence
from delft.utilities.hub_models import HubModelNotFoundError
from delft.utilities.weights import SAFETENSORS_WEIGHT_FILE_NAME

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
MODEL_NAME = "grobid-date-BidLSTM_CRF-test"
REPOSITORY = "lfoppiano/grobid-model-date"


def _trained(model_name):
    sequence = Sequence(model_name, architecture="BidLSTM_CRF", embeddings_name=None, device="cpu")
    sequence.p = Preprocessor(return_chars=True)
    sequence.p.fit([WORDS], [LABELS])
    sequence.model_config.char_vocab_size = len(sequence.p.vocab_char)
    sequence.model = get_model(sequence.model_config, len(sequence.p.vocab_tag), load_pretrained_weights=False)
    return sequence


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A model published as safetensors in the repository of its task."""
    hub = tmp_path / "hub"
    model = _trained(MODEL_NAME)
    model.save(str(hub / REPOSITORY), weight_file=SAFETENSORS_WEIGHT_FILE_NAME)

    def snapshot_download(repo_id, revision=None, allow_patterns=None, local_dir=None, token=None):
        repository = str(hub / repo_id)
        for directory, _, names in os.walk(repository):
            for name in names:
                path = os.path.relpath(os.path.join(directory, name), repository).replace(os.sep, "/")
                if any(fnmatch(path, pattern) for pattern in allow_patterns):
                    os.makedirs(os.path.dirname(os.path.join(local_dir, path)), exist_ok=True)
                    shutil.copy(os.path.join(repository, path), os.path.join(local_dir, path))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return model


def _assert_same_model(loaded, published):
    state, loaded_state = published.model.state_dict(), loaded.model.state_dict()
    assert all(torch.equal(state[name], loaded_state[name]) for name in state)
    assert loaded.tag([WORDS], "raw") == published.tag([WORDS], "raw")


def test_load_downloads_a_model_that_is_not_on_disk(published, tmp_path):
    models_dir = str(tmp_path / "models")
    sequence = Sequence(MODEL_NAME, device="cpu")
    sequence.load(models_dir)

    assert os.path.isdir(os.path.join(models_dir, MODEL_NAME))
    _assert_same_model(sequence, published)


def test_load_prefers_the_model_on_disk(published, tmp_path, monkeypatch):
    models_dir = str(tmp_path / "models")
    local = _trained(MODEL_NAME)
    local.save(models_dir)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: pytest.fail("downloaded"))

    sequence = Sequence(MODEL_NAME, device="cpu")
    sequence.load(models_dir)
    _assert_same_model(sequence, local)


def test_load_of_a_model_that_is_nowhere_tells_where_it_looked(published, tmp_path):
    sequence = Sequence("grobid-date-BidLSTM_CRF-unknown", device="cpu")
    with pytest.raises(HubModelNotFoundError, match="hf://lfoppiano/grobid-model-date/grobid-date-BidLSTM_CRF-unknown"):
        sequence.load(str(tmp_path / "models"))


def test_load_takes_a_repository_and_looks_for_its_own_name_in_it(published, tmp_path):
    sequence = Sequence(MODEL_NAME, device="cpu")
    sequence.load(f"hf://{REPOSITORY}", cache_dir=str(tmp_path / "cache"))

    assert os.path.isdir(tmp_path / "cache" / MODEL_NAME)
    _assert_same_model(sequence, published)


def test_load_takes_the_reference_of_a_model(published, tmp_path):
    sequence = Sequence("whatever", device="cpu")
    sequence.load(f"hf://{REPOSITORY}/{MODEL_NAME}", cache_dir=str(tmp_path / "cache"))

    assert sequence.model_config.model_name == MODEL_NAME
    _assert_same_model(sequence, published)


def test_load_of_a_reference_ignores_the_registry(published, tmp_path):
    sequence = Sequence("grobid-date-BidLSTM_CRF-unknown", device="cpu")
    with pytest.raises(HubModelNotFoundError, match="hf://someone/else/grobid-date-BidLSTM_CRF-unknown"):
        sequence.load("hf://someone/else", cache_dir=str(tmp_path / "cache"))


def test_load_takes_the_directory_of_a_model_whatever_its_name(tmp_path):
    saved = _trained("a-model")
    saved.save(str(tmp_path))
    os.rename(tmp_path / "a-model", tmp_path / "renamed")

    sequence = Sequence("whatever", device="cpu")
    sequence.load(str(tmp_path / "renamed"))
    _assert_same_model(sequence, saved)


def test_load_takes_the_url_of_an_archive(published, tmp_path, http_server):
    shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", tmp_path / "hub" / REPOSITORY, MODEL_NAME)

    sequence = Sequence("whatever", device="cpu")
    sequence.load(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=str(tmp_path / "cache"))

    assert os.path.isdir(tmp_path / "cache" / MODEL_NAME)
    _assert_same_model(sequence, published)


def test_load_takes_the_url_of_a_folder_and_looks_for_its_own_name_in_it(published, tmp_path, http_server):
    shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", tmp_path / "hub" / REPOSITORY, MODEL_NAME)

    sequence = Sequence(MODEL_NAME, device="cpu")
    sequence.load(http_server.url + "/", cache_dir=str(tmp_path / "cache"))
    _assert_same_model(sequence, published)
