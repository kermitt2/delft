import gzip
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from delft.utilities import Embeddings as embeddings_module
from delft.utilities.Embeddings import (
    Embeddings,
    close_lmdb_env,
    download_hub_file,
    hub_file_of,
)

GLOVE = {"repo_id": "stanfordnlp/glove", "filename": "glove.840B.300d.zip", "repo_type": None, "revision": None}
WORD2VEC_REPO = "sciencialab/word2vec-google-news-negative-300"
WORD2VEC_FILE = "GoogleNews-vectors-negative300.vec.gz"


def _hub_file(repo_id, filename, repo_type=None, revision=None):
    return {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}


class TestHubFileOf:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("hf://stanfordnlp/glove/glove.840B.300d.zip", GLOVE),
            (
                "hf://datasets/" + WORD2VEC_REPO + "/" + WORD2VEC_FILE,
                _hub_file(WORD2VEC_REPO, WORD2VEC_FILE, repo_type="dataset"),
            ),
            (
                "hf://datasets/" + WORD2VEC_REPO + "@v1/" + WORD2VEC_FILE,
                _hub_file(WORD2VEC_REPO, WORD2VEC_FILE, repo_type="dataset", revision="v1"),
            ),
            (
                "hf://owner/vectors@refs%2Fpr%2F1/en/vectors.vec",
                _hub_file("owner/vectors", "en/vectors.vec", revision="refs/pr/1"),
            ),
            (
                "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip",
                _hub_file("stanfordnlp/glove", "glove.840B.300d.zip", revision="main"),
            ),
            (
                "https://huggingface.co/datasets/"
                + WORD2VEC_REPO
                + "/resolve/main/"
                + WORD2VEC_FILE
                + "?download=true",
                _hub_file(WORD2VEC_REPO, WORD2VEC_FILE, repo_type="dataset", revision="main"),
            ),
            (
                "https://huggingface.co/stanfordnlp/glove/blob/refs%2Fpr%2F2/glove.6B.zip",
                _hub_file("stanfordnlp/glove", "glove.6B.zip", revision="refs/pr/2"),
            ),
        ],
    )
    def test_a_file_of_the_hub(self, url, expected):
        assert hub_file_of(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
            # the page of a repository, not of a file
            "https://huggingface.co/stanfordnlp/glove",
            "https://huggingface.co/stanfordnlp/glove/tree/main",
            "/data/embeddings/glove.840B.300d.txt",
            "",
            None,
        ],
    )
    def test_not_a_file_of_the_hub(self, url):
        assert hub_file_of(url) is None

    @pytest.mark.parametrize(
        "reference",
        [
            # no file
            "hf://stanfordnlp/glove",
            "hf://datasets/" + WORD2VEC_REPO,
            "hf://owner//vectors.vec",
            "hf://owner/@main/vectors.vec",
            "hf://buckets/owner/bucket/vectors.vec",
        ],
    )
    def test_an_invalid_reference(self, reference):
        with pytest.raises(ValueError):
            hub_file_of(reference)

    def test_the_registry_references(self):
        with open(os.path.join(os.path.dirname(embeddings_module.__file__), "..", "resources-registry.json")) as f:
            registry = json.load(f)
        urls = [entry["url"] for entry in registry["embeddings"] if entry.get("url", "").startswith("hf://")]
        assert len(urls) >= 2
        for url in urls:
            assert hub_file_of(url) is not None


class TestDownloadHubFile:
    def test_through_the_hub_client(self, monkeypatch):
        monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
        with patch("huggingface_hub.hf_hub_download", return_value="/cache/glove.840B.300d.zip") as download:
            assert download_hub_file("hf://stanfordnlp/glove/glove.840B.300d.zip") == "/cache/glove.840B.300d.zip"
        download.assert_called_once_with(**GLOVE, token=None)

    def test_with_the_token_of_the_transformers(self, monkeypatch):
        monkeypatch.setenv("HF_ACCESS_TOKEN", "hf_secret")
        with patch("huggingface_hub.hf_hub_download") as download:
            download_hub_file("hf://stanfordnlp/glove/glove.840B.300d.zip")
        assert download.call_args.kwargs["token"] == "hf_secret"

    def test_not_a_file_of_the_hub(self):
        with pytest.raises(ValueError):
            download_hub_file("https://example.org/vectors.vec")


def _registry(tmp_path, url, path=None):
    return {
        "embedding-lmdb-path": str(tmp_path / "db"),
        "embedding-download-path": str(tmp_path / "download"),
        "embeddings": [
            {
                "name": "tiny-vectors",
                "path": path or str(tmp_path / "missing" / "vectors.vec"),
                "type": "glove",
                "format": "vec",
                "lang": "en",
                "item": "word",
                "url": url,
            }
        ],
        "embeddings-contextualized": [],
        "transformers": [],
    }


def _write_vectors(path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("3 4\n")
        f.write("the 0.1 0.2 0.3 0.4\n")
        f.write("cat 1.0 2.0 3.0 4.0\n")
        f.write("sat -1.0 -2.0 -3.0 -4.0\n")
    return str(path)


class TestEmbeddingsFromTheHub:
    def test_compiled_from_a_file_of_the_hub(self, tmp_path, monkeypatch):
        # a small map is enough, and does not depend on the file system to be sparse
        monkeypatch.setattr(embeddings_module, "map_size", 10 * 1024 * 1024)
        cache = tmp_path / "hf-cache"
        cache.mkdir()
        vectors = _write_vectors(cache / WORD2VEC_FILE)
        (tmp_path / "download").mkdir()
        (tmp_path / "download" / "leftover.vec").write_text("")

        registry = _registry(tmp_path, "hf://datasets/" + WORD2VEC_REPO + "/" + WORD2VEC_FILE)
        with (
            patch("huggingface_hub.hf_hub_download", return_value=vectors) as download,
            patch.object(embeddings_module, "download_file") as plain_download,
        ):
            embeddings = Embeddings("tiny-vectors", resource_registry=registry)
        try:
            download.assert_called_once()
            assert download.call_args.kwargs["repo_id"] == WORD2VEC_REPO
            assert download.call_args.kwargs["repo_type"] == "dataset"
            plain_download.assert_not_called()

            assert embeddings.embed_size == 4
            np.testing.assert_array_equal(embeddings.get_word_vector("cat"), [1.0, 2.0, 3.0, 4.0])
            np.testing.assert_array_equal(embeddings.get_word_vector("dog"), np.zeros(4))
            # the file stays in the cache of the Hub, the download directory is emptied as before
            assert (cache / WORD2VEC_FILE).is_file()
            assert list((tmp_path / "download").iterdir()) == []
        finally:
            close_lmdb_env(str(tmp_path / "db" / "tiny-vectors"))

    def test_a_local_file_is_not_downloaded(self, tmp_path):
        vectors = _write_vectors(tmp_path / "vectors.vec.gz")
        registry = _registry(tmp_path, "hf://stanfordnlp/glove/glove.840B.300d.zip", path=vectors)
        embeddings = Embeddings("tiny-vectors", resource_registry=registry, load=False)
        with patch("huggingface_hub.hf_hub_download") as download:
            assert embeddings.get_embedding_path(registry["embeddings"][0]) == vectors
        download.assert_not_called()

    def test_another_url_is_downloaded_as_before(self, tmp_path):
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        registry = _registry(tmp_path, url)
        embeddings = Embeddings("tiny-vectors", resource_registry=registry, load=False)
        with (
            patch("huggingface_hub.hf_hub_download") as download,
            patch.object(embeddings_module, "download_file", return_value=None) as plain_download,
        ):
            embeddings.get_embedding_path(registry["embeddings"][0])
        download.assert_not_called()
        plain_download.assert_called_once_with(url, registry["embedding-download-path"])

    def test_a_failed_download(self, tmp_path):
        registry = _registry(tmp_path, "hf://stanfordnlp/glove/glove.840B.300d.zip")
        embeddings = Embeddings("tiny-vectors", resource_registry=registry, load=False)
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("no network")):
            assert embeddings.get_embedding_path(registry["embeddings"][0]) is None
