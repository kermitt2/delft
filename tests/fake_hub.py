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
        self.created = []
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

    @staticmethod
    def _folders(paths):
        """Top-level folders: a file at the root, a README for instance, is not one."""
        return sorted({path.split("/")[0] for path in paths if "/" in path})

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
            return [SimpleNamespace(type="directory", path=name) for name in self._folders(paths)]
        return [SimpleNamespace(type="file", path=p) for p in paths if p.startswith(prefix or "")]

    def download_bucket_files(self, bucket_id, files, token=None):
        self.downloads += 1
        for path, local_path in files:
            if not os.path.isfile(os.path.join(self.root, bucket_id, path)):
                raise FileNotFoundError(path)
            self._copy(bucket_id, path, local_path)

    # --- publishing
    def create(self, repo_id, private=None, exist_ok=False, token=None):
        self.created.append((repo_id, private))
        os.makedirs(os.path.join(self.root, repo_id), exist_ok=exist_ok)

    def upload_folder(self, repo_id, folder_path, path_in_repo, delete_patterns=None, **kwargs):
        folder = os.path.join(self.root, repo_id, path_in_repo)
        if delete_patterns and os.path.isdir(folder):
            shutil.rmtree(folder)
        shutil.copytree(folder_path, folder, dirs_exist_ok=True)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kwargs):
        self.batch_bucket_files(repo_id, add=[(path_or_fileobj, path_in_repo)])

    def batch_bucket_files(self, bucket_id, add=None, token=None):
        for source, destination in add or []:
            path = os.path.join(self.root, bucket_id, destination)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if isinstance(source, bytes):
                with open(path, "wb") as f:
                    f.write(source)
            else:
                shutil.copy(source, path)

    def hf_hub_download(self, repo_id, filename, revision=None, local_dir=None, token=None):
        path = os.path.join(self.root, repo_id, filename)
        if not os.path.isfile(path):
            raise errors.EntryNotFoundError(f"no {filename} in {repo_id}")
        return path

    def read(self, repo_id, path):
        with open(os.path.join(self.root, repo_id, path), encoding="utf-8") as f:
            return f.read()

    def list_repo_tree(self, repo_id, revision=None, token=None):
        from huggingface_hub.hf_api import RepoFolder

        return [RepoFolder(path=name, oid="0") for name in self._folders(self._files(repo_id))]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "hub")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake.snapshot_download)
    monkeypatch.setattr("huggingface_hub.list_bucket_tree", fake.list_bucket_tree)
    monkeypatch.setattr("huggingface_hub.download_bucket_files", fake.download_bucket_files)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr("huggingface_hub.create_repo", fake.create)
    monkeypatch.setattr("huggingface_hub.create_bucket", fake.create)
    monkeypatch.setattr("huggingface_hub.batch_bucket_files", fake.batch_bucket_files)
    monkeypatch.setattr("huggingface_hub.HfApi.upload_folder", lambda self, **kwargs: fake.upload_folder(**kwargs))
    monkeypatch.setattr("huggingface_hub.HfApi.upload_file", lambda self, **kwargs: fake.upload_file(**kwargs))
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
