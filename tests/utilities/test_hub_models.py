"""The Hub is played by local directories: nothing here goes to the network."""

import os

import pytest

from delft.utilities import hub_models
from delft.utilities.hub_models import (
    HubModelNotFoundError,
    HubReference,
    downloaded_from,
    fetch_model,
    hub_references,
    list_models,
    parse_reference,
    pull_models,
    resolve_model,
    select_models,
)
from tests.fake_hub import HEADER_MODELS


class TestParseReference:
    @pytest.mark.parametrize(
        "reference, expected",
        [
            (
                "hf://lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF",
                HubReference("lfoppiano/grobid-model-header", "grobid-header-BidLSTM_CRF"),
            ),
            (
                "hf://lfoppiano/grobid-model-header@v1.1.0/grobid-header-BidLSTM_CRF",
                HubReference("lfoppiano/grobid-model-header", "grobid-header-BidLSTM_CRF", revision="v1.1.0"),
            ),
            (
                "hf://buckets/lfoppiano/delft-models/grobid-header-BidLSTM_CRF",
                HubReference("lfoppiano/delft-models", "grobid-header-BidLSTM_CRF", bucket=True),
            ),
            ("hf://lfoppiano/grobid-model-header", HubReference("lfoppiano/grobid-model-header")),
            (
                "hf://lfoppiano/grobid-model-header@main/",
                HubReference("lfoppiano/grobid-model-header", revision="main"),
            ),
            ("hf://buckets/lfoppiano/delft-models", HubReference("lfoppiano/delft-models", bucket=True)),
        ],
    )
    def test_parses_and_writes_back(self, reference, expected):
        assert parse_reference(reference) == expected
        assert str(expected) == reference.rstrip("/")

    @pytest.mark.parametrize(
        "reference",
        [
            "lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF",
            "https://huggingface.co/lfoppiano/grobid-model-header",
            "hf://grobid-model-header",
            "hf://lfoppiano/grobid-model-header/a-model/and-more",
            "hf://lfoppiano//a-model",
            "hf://buckets/lfoppiano/delft-models@main/a-model",
            "hf://lfoppiano/@main/a-model",
        ],
    )
    def test_rejects(self, reference):
        with pytest.raises(ValueError):
            parse_reference(reference)


@pytest.mark.parametrize(
    "location", ["hf://lfoppiano/grobid-model-header", "hf://buckets/lfoppiano/delft-models"], ids=["repo", "bucket"]
)
class TestResolveModel:
    def test_downloads_the_model_under_its_name(self, hub, models_dir, location):
        directory = resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert directory == os.path.join(models_dir, "grobid-header-BidLSTM_CRF")
        assert sorted(os.listdir(directory)) == [".hub-source", "config.json", "model.safetensors", "preprocessor.json"]
        with open(os.path.join(directory, "config.json")) as f:
            assert f.read() == "grobid-header-BidLSTM_CRF/config.json"

    def test_leaves_alone_the_models_whose_name_starts_the_same(self, hub, models_dir, location):
        resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert os.listdir(models_dir) == ["grobid-header-BidLSTM_CRF"]

    def test_model_on_disk_is_not_downloaded_again(self, hub, models_dir, location):
        resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert hub.downloads == 1

    def test_force_downloads_again(self, hub, models_dir, location):
        directory = resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        open(os.path.join(directory, "stale"), "w").close()
        resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir, force=True)
        assert hub.downloads == 2
        assert "stale" not in os.listdir(directory)

    def test_unknown_model(self, hub, models_dir, location):
        with pytest.raises(HubModelNotFoundError, match="grobid-header-BidGRU_CRF"):
            resolve_model(location + "/grobid-header-BidGRU_CRF", cache_dir=models_dir)
        assert os.listdir(models_dir) == []

    def test_unknown_repository(self, hub, models_dir, location):
        with pytest.raises(HubModelNotFoundError):
            resolve_model(location + "-that-is-not/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert os.listdir(models_dir) == []

    def test_interrupted_download_leaves_nothing_passing_for_a_model(self, hub, models_dir, location):
        hub.fail_after_first_file = True
        with pytest.raises(ConnectionError):
            resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert os.listdir(models_dir) == []

        hub.fail_after_first_file = False
        directory = resolve_model(location + "/grobid-header-BidLSTM_CRF", cache_dir=models_dir)
        assert len(os.listdir(directory)) == 4

    def test_reference_without_a_model(self, hub, models_dir, location):
        with pytest.raises(ValueError, match="names no model"):
            resolve_model(location, cache_dir=models_dir)


class TestRevisions:
    """A model on disk is known by its name alone: what it was downloaded from is
    written in it, so that asking for another revision does not keep the one there."""

    REFERENCE = "hf://lfoppiano/grobid-model-header@v1.1.0/grobid-header-BidLSTM_CRF"

    def test_same_revision_is_not_downloaded_again(self, hub, models_dir):
        resolve_model(self.REFERENCE, cache_dir=models_dir)
        resolve_model(self.REFERENCE, cache_dir=models_dir)
        assert hub.downloads == 1

    def test_another_revision_replaces_the_model(self, hub, models_dir):
        directory = resolve_model(self.REFERENCE, cache_dir=models_dir)
        open(os.path.join(directory, "stale"), "w").close()
        resolve_model(self.REFERENCE.replace("v1.1.0", "v1.2.0"), cache_dir=models_dir)
        assert hub.downloads == 2
        assert "stale" not in os.listdir(directory)
        assert downloaded_from(directory) == self.REFERENCE.replace("v1.1.0", "v1.2.0")

    def test_model_that_was_not_downloaded_is_never_replaced(self, hub, models_dir):
        trained = os.path.join(models_dir, "grobid-header-BidLSTM_CRF")
        os.makedirs(trained)
        open(os.path.join(trained, "config.json"), "w").close()
        assert resolve_model(self.REFERENCE, cache_dir=models_dir) == trained
        assert fetch_model("grobid-header-BidLSTM_CRF", models_dir, TestRegistry.REGISTRY) == trained
        assert hub.downloads == 0

    def test_new_revision_in_the_registry_reaches_the_models_on_disk(self, hub, models_dir):
        registry = {"models-hub": {"repo": "lfoppiano/grobid-model-{short_name}", "revision": "v1.1.0"}}
        directory = fetch_model("grobid-header-BidLSTM_CRF", models_dir, registry)
        fetch_model("grobid-header-BidLSTM_CRF", models_dir, registry)
        assert hub.downloads == 1

        registry["models-hub"]["revision"] = "v1.2.0"
        fetch_model("grobid-header-BidLSTM_CRF", models_dir, registry)
        assert hub.downloads == 2
        assert downloaded_from(directory) == "hf://lfoppiano/grobid-model-header@v1.2.0/grobid-header-BidLSTM_CRF"


def test_models_go_to_the_directory_of_the_environment_variable(hub, tmp_path, monkeypatch):
    monkeypatch.setenv(hub_models.MODELS_DIR_VARIABLE, str(tmp_path / "from-env"))
    directory = resolve_model("hf://lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF")
    assert directory == str(tmp_path / "from-env" / "grobid-header-BidLSTM_CRF")


class TestListAndSelect:
    @pytest.mark.parametrize("location", ["hf://lfoppiano/grobid-model-header", "hf://buckets/lfoppiano/delft-models"])
    def test_lists_the_models(self, hub, location):
        assert [name for name in list_models(location) if "header" in name] == sorted(HEADER_MODELS)

    def test_architecture_takes_its_models_with_and_without_suffix_only(self):
        assert select_models(HEADER_MODELS, architecture="BidLSTM_CRF") == [
            "grobid-header-BidLSTM_CRF",
            "grobid-header-BidLSTM_CRF-potion-base-8M",
        ]

    def test_architecture_and_suffix_take_one_model(self):
        assert select_models(HEADER_MODELS, architecture="BidLSTM_CRF", suffix="potion-base-8M") == [
            "grobid-header-BidLSTM_CRF-potion-base-8M"
        ]

    def test_empty_suffix_takes_the_model_without_one(self):
        assert select_models(HEADER_MODELS, architecture="BidLSTM_CRF", suffix="") == ["grobid-header-BidLSTM_CRF"]

    def test_wapiti_is_selected_like_an_architecture(self):
        assert select_models(HEADER_MODELS, architecture="wapiti") == ["grobid-header-wapiti"]

    def test_pattern(self):
        assert select_models(HEADER_MODELS, match="*scibert*") == ["grobid-header-BERT_CRF-scibert_scivocab_cased"]

    def test_no_selection_takes_all(self):
        assert select_models(HEADER_MODELS) == HEADER_MODELS

    def test_pull_downloads_the_selection(self, hub, models_dir):
        directories = pull_models(
            "hf://lfoppiano/grobid-model-header", cache_dir=models_dir, architecture="BidLSTM_CRF"
        )
        assert sorted(os.listdir(models_dir)) == [
            "grobid-header-BidLSTM_CRF",
            "grobid-header-BidLSTM_CRF-potion-base-8M",
        ]
        assert sorted(directories) == sorted(os.path.join(models_dir, name) for name in os.listdir(models_dir))

    def test_pull_downloads_everything(self, hub, models_dir):
        pull_models("hf://lfoppiano/grobid-model-header", cache_dir=models_dir)
        assert sorted(os.listdir(models_dir)) == sorted(HEADER_MODELS)


class TestRegistry:
    REGISTRY = {
        "models-hub": {
            "repo": "lfoppiano/grobid-model-{short_name}",
            "revision": "v1.1.0",
            "bucket": "lfoppiano/delft-models",
            "overrides": {"datasets": "lfoppiano/delft-model-datasets", "grobid-header-BERT_CRF": "other/bert-models"},
        }
    }

    def test_grobid_model_is_in_the_repository_of_its_task_then_in_the_bucket(self):
        name = "grobid-affiliation-address-BidLSTM_CRF-potion-base-8M"
        assert hub_references(name, self.REGISTRY) == [
            HubReference("lfoppiano/grobid-model-affiliation-address", name, revision="v1.1.0"),
            HubReference("lfoppiano/delft-models", name, bucket=True),
        ]

    def test_other_model_has_a_repository_by_override_only(self):
        assert [str(r) for r in hub_references("datasets-BidLSTM_CRF", self.REGISTRY)] == [
            "hf://lfoppiano/delft-model-datasets@v1.1.0/datasets-BidLSTM_CRF",
            "hf://buckets/lfoppiano/delft-models/datasets-BidLSTM_CRF",
        ]
        assert [str(r) for r in hub_references("license_gru", self.REGISTRY)] == [
            "hf://buckets/lfoppiano/delft-models/license_gru"
        ]

    def test_override_wins_over_the_rule_and_matches_whole_parts_of_a_name(self):
        assert hub_references("grobid-header-BERT_CRF-scibert", self.REGISTRY)[0].repo_id == "other/bert-models"
        assert (
            hub_references("grobid-header-BERT_CRF_FEATURES", self.REGISTRY)[0].repo_id
            == "lfoppiano/grobid-model-header"
        )
        assert hub_references("datasetsplus-BidLSTM_CRF", self.REGISTRY)[0].bucket

    @pytest.mark.parametrize("registry", [None, {}, {"models-hub": None}, {"models-hub": {}}])
    def test_no_section_no_place_on_the_hub(self, registry):
        assert hub_references("grobid-header-BidLSTM_CRF", registry) == []
        assert fetch_model("grobid-header-BidLSTM_CRF", "unused", registry) is None

    def test_fetch_takes_the_model_from_the_bucket_when_the_repository_has_none(self, hub, models_dir):
        directory = fetch_model("grobid-date-BidLSTM_CRF", models_dir, self.REGISTRY)
        assert directory == os.path.join(models_dir, "grobid-date-BidLSTM_CRF")
        assert downloaded_from(directory) == "hf://buckets/lfoppiano/delft-models/grobid-date-BidLSTM_CRF"

    def test_fetch_tells_every_place_it_looked(self, hub, models_dir):
        with pytest.raises(HubModelNotFoundError) as error:
            fetch_model("grobid-figure-BidLSTM_CRF", models_dir, self.REGISTRY)
        assert "hf://lfoppiano/grobid-model-figure@v1.1.0/grobid-figure-BidLSTM_CRF" in str(error.value)
        assert "hf://buckets/lfoppiano/delft-models/grobid-figure-BidLSTM_CRF" in str(error.value)


def test_shipped_registry_gives_the_grobid_models_a_repository():
    from delft import DELFT_PROJECT_DIR
    from delft.utilities.Embeddings import load_resource_registry

    registry = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))
    assert [str(r) for r in hub_references("grobid-header-BidLSTM_CRF_FEATURES", registry)] == [
        "hf://lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF_FEATURES"
    ]
