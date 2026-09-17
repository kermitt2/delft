"""The Hugging Face Hub played by local directories, so that tests stay off the network."""

import os
import shutil
import threading
from fnmatch import fnmatch
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import errors

HEADER_MODELS = [
    "grobid-header-BidLSTM_CRF",
    "grobid-header-BidLSTM_CRF-potion-base-8M",
    "grobid-header-BidLSTM_CRF_FEATURES",
    "grobid-header-BERT_CRF-scibert_scivocab_cased",
    "grobid-header-wapiti",
]


class FakeHub:
    """Repositories and buckets as directories, ``{root}/{owner}/{name}/{model}/{file}``."""

    def __init__(self, root):
        self.root = str(root)
        self.downloads = 0
        self.fail_after_first_file = False

    def add(self, repo_id, model_name, files=("config.json", "preprocessor.json", "model.safetensors")):
        directory = os.path.join(self.root, repo_id, model_name)
        os.makedirs(directory)
        for name in files:
            with open(os.path.join(directory, name), "w") as f:
                f.write(f"{model_name}/{name}")

    def _files(self, repo_id):
        repository = os.path.join(self.root, repo_id)
        if not os.path.isdir(repository):
            response = httpx.Response(404, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}"))
            raise errors.RepositoryNotFoundError(f"no repository {repo_id}", response=response)
        for directory, _, names in os.walk(repository):
            for name in names:
                yield os.path.relpath(os.path.join(directory, name), repository).replace(os.sep, "/")

    def _copy(self, repo_id, path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(os.path.join(self.root, repo_id, path), local_path)
        if self.fail_after_first_file:
            raise ConnectionError("connection lost")

    # --- what huggingface_hub offers
    def snapshot_download(self, repo_id, revision=None, allow_patterns=None, local_dir=None, token=None):
        self.downloads += 1
        for path in sorted(self._files(repo_id)):
            if any(fnmatch(path, pattern) for pattern in allow_patterns):
                self._copy(repo_id, path, os.path.join(local_dir, path))

    def list_bucket_tree(self, bucket_id, prefix=None, recursive=False, token=None):
        paths = sorted(self._files(bucket_id))
        if not recursive:
            return [SimpleNamespace(type="directory", path=name) for name in sorted({p.split("/")[0] for p in paths})]
        return [SimpleNamespace(type="file", path=p) for p in paths if p.startswith(prefix or "")]

    def download_bucket_files(self, bucket_id, files, token=None):
        self.downloads += 1
        for path, local_path in files:
            self._copy(bucket_id, path, local_path)

    def list_repo_tree(self, repo_id, revision=None, token=None):
        from huggingface_hub.hf_api import RepoFolder

        names = sorted({path.split("/")[0] for path in self._files(repo_id)})
        return [RepoFolder(path=name, oid="0") for name in names]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "hub")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake.snapshot_download)
    monkeypatch.setattr("huggingface_hub.list_bucket_tree", fake.list_bucket_tree)
    monkeypatch.setattr("huggingface_hub.download_bucket_files", fake.download_bucket_files)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_tree", lambda self, *args, **kwargs: fake.list_repo_tree(*args, **kwargs)
    )
    for model_name in HEADER_MODELS:
        fake.add("lfoppiano/grobid-model-header", model_name)
        fake.add("lfoppiano/delft-models", model_name)
    fake.add("lfoppiano/delft-models", "grobid-date-BidLSTM_CRF")
    return fake


@pytest.fixture
def models_dir(tmp_path):
    return str(tmp_path / "models")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def http_server(tmp_path):
    """A server of the files of its ``directory``, at ``url``."""
    directory = tmp_path / "http"
    directory.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(directory=directory, url=f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()
    server.server_close()
