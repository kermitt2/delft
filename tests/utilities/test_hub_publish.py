"""Publishing to the Hugging Face Hub, played by local directories (tests/fake_hub.py)."""

import os

import pytest
import torch

from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.wrapper import Sequence
from delft.utilities.hub_models import SOURCE_FILE_NAME, list_models, resolve_model
from delft.utilities.hub_publish import TABLE_END, TABLE_START, prepare_model, push_location, push_model

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
REPOSITORY = "hf://lfoppiano/grobid-model-date"
BUCKET = "hf://buckets/lfoppiano/experiments"


def _save(models_dir, model_name, **kwargs):
    sequence = Sequence(model_name, architecture="BidLSTM_CRF", embeddings_name=None, device="cpu")
    sequence.p = Preprocessor(return_chars=True)
    sequence.p.fit([WORDS], [LABELS])
    sequence.model_config.char_vocab_size = len(sequence.p.vocab_char)
    sequence.model = get_model(sequence.model_config, len(sequence.p.vocab_tag), load_pretrained_weights=False)
    sequence.save(str(models_dir), **kwargs)
    return sequence, os.path.join(str(models_dir), model_name)


class TestPrepareModel:
    def test_pickled_weights_are_published_as_safetensors(self, tmp_path):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        published = prepare_model(model_dir, str(tmp_path / "staging"))
        assert published == ["config.json", "model.safetensors", "preprocessor.json"]
        assert sorted(os.listdir(tmp_path / "staging")) == published

    def test_pickled_weights_can_be_kept(self, tmp_path):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        published = prepare_model(model_dir, str(tmp_path / "staging"), safetensors=False)
        assert published == ["config.json", "model_weights.pt", "preprocessor.json"]

    def test_safetensors_weights_are_published_alone(self, tmp_path):
        sequence, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        sequence.save(str(tmp_path / "models"), weight_file="model.safetensors")
        assert "model_weights.pt" in os.listdir(model_dir)
        published = prepare_model(model_dir, str(tmp_path / "staging"))
        assert published == ["config.json", "model.safetensors", "preprocessor.json"]

    def test_record_of_a_download_is_not_published(self, tmp_path):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        open(os.path.join(model_dir, SOURCE_FILE_NAME), "w").close()
        assert SOURCE_FILE_NAME not in prepare_model(model_dir, str(tmp_path / "staging"))

    def test_several_pickled_weights_are_ambiguous(self, tmp_path):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        open(os.path.join(model_dir, "model_weights0.pt"), "w").close()
        with pytest.raises(ValueError, match="ambiguous"):
            prepare_model(model_dir, str(tmp_path / "staging"))


@pytest.mark.parametrize("location", [REPOSITORY, BUCKET], ids=["repo", "bucket"])
class TestPushModel:
    def test_published_model_loads_back_the_same(self, hub, tmp_path, location):
        saved, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF-potion-base-8M")
        reference = push_model(model_dir, location)
        assert str(reference) == location + "/grobid-date-BidLSTM_CRF-potion-base-8M"

        loaded = Sequence("grobid-date-BidLSTM_CRF-potion-base-8M", device="cpu")
        loaded.load(str(reference), cache_dir=str(tmp_path / "cache"))
        state, loaded_state = saved.model.state_dict(), loaded.model.state_dict()
        assert state.keys() == loaded_state.keys()
        assert all(torch.equal(state[name], loaded_state[name]) for name in state)
        assert loaded.tag([WORDS], "raw") == saved.tag([WORDS], "raw")

    def test_creates_the_place_it_publishes_to(self, hub, tmp_path, location):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        push_model(model_dir, location, private=True)
        assert hub.created == [(location.split("/", 2)[2].replace("buckets/", ""), True)]

    def test_readme_lists_the_models_and_keeps_what_was_written_in_it(self, hub, tmp_path, location):
        repo_id = location.split("/", 2)[2].replace("buckets/", "")
        _, first = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        _, second = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF-potion-base-8M")

        push_model(first, location)
        readme = hub.read(repo_id, "README.md")
        assert "| `grobid-date-BidLSTM_CRF` | BidLSTM_CRF |  |  |  |" in readme
        hub.batch_bucket_files(repo_id, add=[(readme.replace("# grobid-model-date", "# Dates").encode(), "README.md")])
        hub.batch_bucket_files(repo_id, add=[((readme + "\nScores: see the paper.\n").encode(), "README.md")])

        push_model(second, location)
        readme = hub.read(repo_id, "README.md")
        assert "| `grobid-date-BidLSTM_CRF` | BidLSTM_CRF |  |  |  |" in readme
        assert "| `grobid-date-BidLSTM_CRF-potion-base-8M` | BidLSTM_CRF |  |  | potion-base-8M |" in readme
        assert readme.count(TABLE_START) == 1 and readme.count(TABLE_END) == 1
        assert readme.rstrip().endswith("Scores: see the paper.")
        assert list_models(location) == ["grobid-date-BidLSTM_CRF", "grobid-date-BidLSTM_CRF-potion-base-8M"]

    def test_place_naming_a_model_is_refused(self, hub, tmp_path, location):
        _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
        with pytest.raises(ValueError, match="names a model"):
            push_model(model_dir, location + "/grobid-date-BidLSTM_CRF")

    def test_directory_without_a_model(self, hub, tmp_path, location):
        with pytest.raises(FileNotFoundError):
            push_model(str(tmp_path / "nothing"), location)
        assert hub.created == []


def test_publishing_again_replaces_the_files_of_the_model(hub, tmp_path):
    _, model_dir = _save(tmp_path / "models", "grobid-date-BidLSTM_CRF")
    push_model(model_dir, REPOSITORY, safetensors=False)
    push_model(model_dir, REPOSITORY)
    directory = resolve_model(REPOSITORY + "/grobid-date-BidLSTM_CRF", cache_dir=str(tmp_path / "cache"))
    assert "model_weights.pt" not in os.listdir(directory)
    assert "model.safetensors" in os.listdir(directory)


class TestPushLocation:
    REGISTRY = {
        "models-hub": {
            "repo": "lfoppiano/grobid-model-{short_name}",
            "revision": "v1.1.0",
            "bucket": "lfoppiano/delft-models",
        }
    }

    def test_model_is_published_to_the_default_branch_of_the_repository_of_its_task(self):
        assert str(push_location("grobid-date-BidLSTM_CRF", self.REGISTRY)) == "hf://lfoppiano/grobid-model-date"

    def test_bucket(self):
        location = push_location("grobid-date-BidLSTM_CRF", self.REGISTRY, bucket=True)
        assert str(location) == "hf://buckets/lfoppiano/delft-models"

    def test_no_place(self):
        with pytest.raises(ValueError, match="no repository"):
            push_location("license_gru", self.REGISTRY)
        with pytest.raises(ValueError, match="no bucket"):
            push_location("grobid-date-BidLSTM_CRF", {"models-hub": {"repo": "a/b-{short_name}"}}, bucket=True)
