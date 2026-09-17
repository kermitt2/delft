"""Models taken over HTTP: the pages of the Hub, and the archives of anywhere else."""

import io
import os
import shutil
import tarfile
import zipfile

import pytest

from delft.utilities.hub_models import (
    SOURCE_FILE_NAME,
    HubModelNotFoundError,
    HubReference,
    hub_reference_of_url,
    is_remote,
    resolve_model,
)

MODEL_NAME = "grobid-date-BidLSTM_CRF"
FILES = ("config.json", "preprocessor.json", "model.safetensors")


@pytest.fixture
def model(tmp_path):
    """A directory holding the one of a model."""
    directory = tmp_path / "source" / MODEL_NAME
    directory.mkdir(parents=True)
    for name in FILES:
        (directory / name).write_text(name)
    return directory.parent


def _files(model_path):
    return sorted(name for name in os.listdir(model_path) if name != SOURCE_FILE_NAME)


class TestHubUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://huggingface.co/owner/repository", HubReference("owner/repository")),
            ("https://huggingface.co/owner/repository/", HubReference("owner/repository")),
            ("https://huggingface.co/owner/repository/tree/main", HubReference("owner/repository", None, "main")),
            (
                "https://huggingface.co/owner/repository/tree/v1.1.0/grobid-date-BidLSTM_CRF",
                HubReference("owner/repository", "grobid-date-BidLSTM_CRF", "v1.1.0"),
            ),
        ],
    )
    def test_page_of_a_repository(self, url, expected):
        assert hub_reference_of_url(url) == expected

    def test_url_of_somewhere_else(self):
        assert hub_reference_of_url("https://example.org/owner/repository") is None

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co/owner",
            "https://huggingface.co/datasets/owner/name",
            "https://huggingface.co/buckets/owner/name",
            "https://huggingface.co/owner/repository/blob/main/model/config.json",
            "https://huggingface.co/owner/repository/tree/main/model/config.json",
        ],
    )
    def test_other_pages_of_the_hub(self, url):
        with pytest.raises(ValueError, match="Invalid URL"):
            hub_reference_of_url(url)

    def test_is_downloaded_as_the_reference_is(self, hub, models_dir):
        url = "https://huggingface.co/lfoppiano/grobid-model-header/tree/main/grobid-header-BidLSTM_CRF"
        model_path = resolve_model(url, cache_dir=models_dir)

        assert _files(model_path) == sorted(FILES)
        with open(os.path.join(model_path, SOURCE_FILE_NAME)) as f:
            assert f.read().strip() == "hf://lfoppiano/grobid-model-header@main/grobid-header-BidLSTM_CRF"

    def test_of_a_repository_takes_the_model_that_is_named(self, hub, models_dir):
        url = "https://huggingface.co/lfoppiano/grobid-model-header"
        model_path = resolve_model(url, cache_dir=models_dir, model_name="grobid-header-BidLSTM_CRF")
        assert os.path.basename(model_path) == "grobid-header-BidLSTM_CRF"


def test_is_remote():
    assert is_remote("hf://owner/repository")
    assert is_remote(HubReference("owner/repository"))
    assert is_remote("https://example.org/model.zip")
    assert is_remote("HTTP://example.org/model.zip")
    assert not is_remote("data/models/sequenceLabelling/")
    assert not is_remote(None)


class TestArchive:
    @pytest.mark.parametrize("archive_format, extension", [("zip", ".zip"), ("gztar", ".tar.gz"), ("tar", ".tar")])
    def test_holding_the_folder_of_the_model(self, model, http_server, models_dir, archive_format, extension):
        shutil.make_archive(str(http_server.directory / MODEL_NAME), archive_format, model, MODEL_NAME)
        url = f"{http_server.url}/{MODEL_NAME}{extension}"

        model_path = resolve_model(url, cache_dir=models_dir)

        assert model_path == os.path.join(models_dir, MODEL_NAME)
        assert _files(model_path) == sorted(FILES)
        with open(os.path.join(model_path, SOURCE_FILE_NAME)) as f:
            assert f.read().strip() == url
        assert os.listdir(models_dir) == [MODEL_NAME]

    def test_holding_the_files_of_the_model(self, model, http_server, models_dir):
        shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", model / MODEL_NAME)
        model_path = resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir)
        assert _files(model_path) == sorted(FILES)

    def test_is_named_after_the_model_whatever_the_folder_in_it(self, model, http_server, models_dir):
        shutil.make_archive(str(http_server.directory / "another-name"), "zip", model, MODEL_NAME)
        model_path = resolve_model(f"{http_server.url}/another-name.zip?version=2", cache_dir=models_dir)
        assert os.path.basename(model_path) == "another-name"

    def test_of_the_model_that_is_named_in_a_folder(self, model, http_server, models_dir):
        shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", model, MODEL_NAME)
        model_path = resolve_model(http_server.url + "/", cache_dir=models_dir, model_name=MODEL_NAME)
        assert _files(model_path) == sorted(FILES)

    def test_folder_without_a_model_name(self, http_server, models_dir):
        with pytest.raises(ValueError, match="names no model"):
            resolve_model(http_server.url + "/", cache_dir=models_dir)

    def test_is_downloaded_once(self, model, http_server, models_dir):
        shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", model, MODEL_NAME)
        url = f"{http_server.url}/{MODEL_NAME}.zip"
        resolve_model(url, cache_dir=models_dir)
        os.remove(http_server.directory / (MODEL_NAME + ".zip"))

        assert resolve_model(url, cache_dir=models_dir) == os.path.join(models_dir, MODEL_NAME)
        with pytest.raises(HubModelNotFoundError):
            resolve_model(url, cache_dir=models_dir, force=True)
        # the model that was there is still
        assert _files(os.path.join(models_dir, MODEL_NAME)) == sorted(FILES)

    def test_from_another_url_replaces_the_model(self, model, http_server, models_dir):
        shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", model, MODEL_NAME)
        (http_server.directory / "v2").mkdir()
        (model / MODEL_NAME / "config.json").write_text("v2")
        shutil.make_archive(str(http_server.directory / "v2" / MODEL_NAME), "zip", model, MODEL_NAME)

        resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir)
        model_path = resolve_model(f"{http_server.url}/v2/{MODEL_NAME}.zip", cache_dir=models_dir)
        with open(os.path.join(model_path, "config.json")) as f:
            assert f.read() == "v2"

    def test_that_is_not_there(self, http_server, models_dir):
        with pytest.raises(HubModelNotFoundError, match="HTTP 404"):
            resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir)
        assert os.listdir(models_dir) == []

    def test_without_a_model(self, http_server, models_dir):
        with zipfile.ZipFile(http_server.directory / (MODEL_NAME + ".zip"), "w") as zip_file:
            zip_file.writestr("README.md", "nothing here")
        with pytest.raises(HubModelNotFoundError, match="holds no config.json"):
            resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir)
        assert os.listdir(models_dir) == []

    def test_that_is_no_archive(self, http_server, models_dir):
        (http_server.directory / (MODEL_NAME + ".tar.gz")).write_text("<html>moved</html>")
        with pytest.raises(ValueError, match="not an archive"):
            resolve_model(f"{http_server.url}/{MODEL_NAME}.tar.gz", cache_dir=models_dir)

    def test_zip_writing_out_of_its_directory(self, http_server, models_dir, tmp_path):
        with zipfile.ZipFile(http_server.directory / (MODEL_NAME + ".zip"), "w") as zip_file:
            zip_file.writestr("config.json", "{}")
            zip_file.writestr("../../escaped", "out")
        with pytest.raises(ValueError, match="out of its directory"):
            resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir)
        assert not (tmp_path / "escaped").exists()

    def test_tar_holding_a_link(self, http_server, models_dir):
        with tarfile.open(http_server.directory / (MODEL_NAME + ".tar"), "w") as tar_file:
            config = tarfile.TarInfo("config.json")
            tar_file.addfile(config, io.BytesIO(b""))
            link = tarfile.TarInfo("passwd")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar_file.addfile(link)
        with pytest.raises(ValueError, match="neither a file nor a directory"):
            resolve_model(f"{http_server.url}/{MODEL_NAME}.tar", cache_dir=models_dir)

    def test_token_is_sent(self, model, http_server, models_dir, monkeypatch):
        import requests

        shutil.make_archive(str(http_server.directory / MODEL_NAME), "zip", model, MODEL_NAME)
        sent = {}
        get = requests.get

        def spy(url, **kwargs):
            sent.update(kwargs["headers"])
            return get(url, **kwargs)

        monkeypatch.setattr(requests, "get", spy)
        resolve_model(f"{http_server.url}/{MODEL_NAME}.zip", cache_dir=models_dir, token="secret")
        assert sent == {"Authorization": "Bearer secret"}
