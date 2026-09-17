"""
Models on the Hugging Face Hub.

A model is a directory, named as ``delft.utilities.model_names`` says, holding its
configuration, its preprocessor and its weights. On the Hub the same directory is a
folder of either

- a model repository, one per task, holding the models of that task:
  ``hf://lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF_FEATURES``, the
  revision being optional: ``hf://lfoppiano/grobid-model-header@v1.1.0/grobid-header-...``;
- a bucket, the mutable and unversioned storage of the Hub, mirroring a models
  directory: ``hf://buckets/lfoppiano/delft-models/grobid-header-BidLSTM_CRF_FEATURES``.

This is the syntax of the ``hf://`` paths of ``huggingface_hub``. A folder keeps the
name of the local directory, so getting a model is a copy, with nothing to rename.

A model is also taken over HTTP, from

- the page of a repository of the Hub, as a browser shows it:
  ``https://huggingface.co/lfoppiano/grobid-model-header/tree/v1.1.0/grobid-header-...``,
  which stands for the ``hf://`` reference of the same model;
- anywhere else, an archive of the model directory, named after the model:
  ``https://example.org/models/grobid-header-BidLSTM_CRF_FEATURES.zip``, a ``.zip``,
  ``.tar.gz``, ``.tgz`` or ``.tar`` file holding the files of the model, at its root or
  in a single folder.

Nothing here depends on the working directory, nor reads the DeLFT resources registry
unless given one: an application embedding DeLFT passes references and a ``cache_dir``
of its own.
"""

import fnmatch
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote, unquote, urlparse

from delft.utilities.model_names import split_model_name

LOGGER = logging.getLogger(__name__)

HF_SCHEME = "hf://"
BUCKETS = "buckets"

HTTP_SCHEMES = ("http://", "https://")
HUB_HOSTS = ("huggingface.co", "www.huggingface.co")
# what the Hub serves under the same paths as the model repositories
HUB_NOT_MODELS = (BUCKETS, "datasets", "spaces")

ARCHIVE_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar")
# the one of the archive looked for under a URL that names none
DEFAULT_ARCHIVE_EXTENSION = ".zip"

# the file both wrappers save the configuration of a model in
CONFIG_FILE_NAME = "config.json"

# section of the resources registry telling where the models are on the Hub
REGISTRY_SECTION = "models-hub"

# Written in a downloaded model, it holds the reference the model comes from. A model on
# disk is otherwise known by its name alone, and asking for another revision of it
# would silently keep the one that is there.
SOURCE_FILE_NAME = ".hub-source"

MODELS_DIR_VARIABLE = "DELFT_MODELS_DIR"
DEFAULT_MODELS_DIR = os.path.join("~", ".cache", "delft", "models")


class HubModelNotFoundError(FileNotFoundError):
    """The model is at none of the places of the Hub it was looked for."""


@dataclass(frozen=True)
class HubReference:
    """A repository or a bucket of the Hub, and the model in it when ``model_name`` is set."""

    repo_id: str
    model_name: Optional[str] = None
    revision: Optional[str] = None
    bucket: bool = False

    def for_model(self, model_name):
        return HubReference(self.repo_id, model_name, self.revision, self.bucket)

    def __str__(self):
        location = f"{BUCKETS}/{self.repo_id}" if self.bucket else self.repo_id
        if self.revision:
            location += "@" + self.revision
        return HF_SCHEME + location + ("/" + self.model_name if self.model_name else "")


def is_hub_reference(value):
    return isinstance(value, HubReference) or (isinstance(value, str) and value.startswith(HF_SCHEME))


def is_http_url(value):
    return isinstance(value, str) and value.lower().startswith(HTTP_SCHEMES)


def is_remote(value):
    """Whether ``value`` is a place ``resolve_model`` downloads a model from."""
    return is_hub_reference(value) or is_http_url(value)


def hub_reference_of_url(url):
    """
    The reference of ``https://huggingface.co/{owner}/{repository}[/tree/{revision}[/{model}]]``,
    the page of a model repository, or None for a URL that is not one of the Hub.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() not in HUB_HOSTS:
        return None

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    expected = f"expected https://{HUB_HOSTS[0]}/owner/repository[/tree/revision[/model]]"
    if len(parts) < 2 or parts[0] in HUB_NOT_MODELS:
        raise ValueError(f"Invalid URL {url!r}, {expected}: for a bucket, use {HF_SCHEME}{BUCKETS}/owner/bucket")
    rest = parts[2:]
    if rest and (rest[0] != "tree" or len(rest) not in (2, 3)):
        raise ValueError(f"Invalid URL {url!r}, {expected}")
    revision = rest[1] if rest else None
    model_name = rest[2] if len(rest) == 3 else None
    return HubReference(f"{parts[0]}/{parts[1]}", model_name, revision)


def parse_reference(reference):
    """
    ``hf://{owner}/{repository}[@{revision}][/{model}]`` or
    ``hf://buckets/{owner}/{bucket}[/{model}]``. A bucket has no revision.
    """
    if isinstance(reference, HubReference):
        return reference
    if not is_hub_reference(reference):
        raise ValueError(f"Not a reference to the Hugging Face Hub, which starts with {HF_SCHEME}: {reference!r}")

    parts = reference[len(HF_SCHEME) :].strip("/").split("/")
    bucket = parts[0] == BUCKETS
    if bucket:
        parts = parts[1:]
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(
            f"Invalid reference {reference!r}, expected {HF_SCHEME}owner/repository[@revision][/model] "
            f"or {HF_SCHEME}{BUCKETS}/owner/bucket[/model]"
        )

    owner, name = parts[0], parts[1]
    name, _, revision = name.partition("@")
    if revision and bucket:
        raise ValueError(f"Invalid reference {reference!r}: a bucket is not versioned and takes no revision")
    if not name:
        raise ValueError(f"Invalid reference {reference!r}: no repository name")
    model_name = parts[2] if len(parts) == 3 else None
    return HubReference(f"{owner}/{name}", model_name, revision or None, bucket)


def default_models_dir():
    """Directory the models are downloaded to when a caller names none."""
    return os.path.expanduser(os.environ.get(MODELS_DIR_VARIABLE) or DEFAULT_MODELS_DIR)


def _not_found_errors():
    from huggingface_hub import errors

    return (
        errors.RepositoryNotFoundError,
        errors.RevisionNotFoundError,
        errors.EntryNotFoundError,
        errors.BucketNotFoundError,
    )


def _download_from_repository(reference, directory, token):
    from huggingface_hub import snapshot_download

    # the folder of the model alone: a repository holds every model of a task, and the
    # ones with a transformer inside weigh hundreds of megabytes each
    snapshot_download(
        reference.repo_id,
        revision=reference.revision,
        allow_patterns=[reference.model_name + "/*"],
        local_dir=directory,
        token=token,
    )


def _download_from_bucket(reference, directory, token):
    from huggingface_hub import download_bucket_files, list_bucket_tree

    # with its slash, the prefix does not match the models whose name only starts the same
    prefix = reference.model_name + "/"
    entries = list_bucket_tree(reference.repo_id, prefix=prefix, recursive=True, token=token)
    paths = [entry.path for entry in entries if entry.type == "file"]
    files = [(path, os.path.join(directory, *path.split("/"))) for path in paths if path.startswith(prefix)]
    for _, local_path in files:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if files:
        download_bucket_files(reference.repo_id, files=files, token=token)


def _is_model_directory(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def downloaded_from(model_path):
    """The reference the model in ``model_path`` was downloaded from, as a string, or
    None for a model that was not downloaded: trained here, or copied by hand."""
    try:
        with open(os.path.join(model_path, SOURCE_FILE_NAME)) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _install(source, model_name, cache_dir, force, download):
    """
    Return ``{cache_dir}/{model_name}``, after ``download(directory)`` has left the model
    in ``{directory}/{model_name}`` when it is not there, was downloaded from somewhere
    else than ``source``, or ``force`` is set.

    A model is downloaded next to its place and moved there when complete, so that an
    interrupted download never leaves a directory passing for a model.
    """
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else default_models_dir()
    target = os.path.join(cache_dir, model_name)
    if _is_model_directory(target) and not force:
        previous = downloaded_from(target)
        if previous is None or previous == source:
            return target
        LOGGER.info("%s was downloaded from %s, replacing it", target, previous)

    os.makedirs(cache_dir, exist_ok=True)
    download_dir = tempfile.mkdtemp(prefix=f".{model_name}.", dir=cache_dir)
    try:
        LOGGER.info("downloading %s to %s", source, target)
        download(download_dir)

        downloaded = os.path.join(download_dir, model_name)
        if not _is_model_directory(downloaded):
            # no source fails on a folder that does not exist: they download nothing
            raise HubModelNotFoundError(f"{source} not found: no such model there")

        with open(os.path.join(downloaded, SOURCE_FILE_NAME), "w") as f:
            f.write(source + "\n")

        if os.path.isdir(target):
            shutil.rmtree(target)
        try:
            os.replace(downloaded, target)
        except OSError:
            # another process downloaded the model in the meantime
            if not _is_model_directory(target):
                raise
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

    return target


def resolve_model(reference, cache_dir=None, token=None, force=False, model_name=None):
    """
    Return the local directory of the model ``reference`` points to,
    ``{cache_dir}/{model name}``, downloading it when it is not there. ``reference`` is
    a ``hf://`` reference, or a URL (see the top of this module). When it is the one of
    a place holding models, a repository, a bucket or the URL of a folder, the model is
    the one named ``model_name`` there.

    A model on disk is kept, unless it was downloaded from another reference, another
    revision for instance, or ``force`` is set. A reference without a revision follows a
    branch that moves, and a URL a file that can change: only ``force`` gets what was
    published since the download.
    """
    if is_http_url(reference):
        hub_reference = hub_reference_of_url(reference)
        if hub_reference is None:
            return _resolve_archive(reference, model_name, cache_dir, token, force)
        reference = hub_reference

    reference = parse_reference(reference)
    if reference.model_name is None and model_name:
        reference = reference.for_model(model_name)
    if reference.model_name is None:
        raise ValueError(f"{reference} names no model, expected {reference}/<model name>")

    def download(directory):
        download_from = _download_from_bucket if reference.bucket else _download_from_repository
        try:
            download_from(reference, directory, token)
        except _not_found_errors() as e:
            raise HubModelNotFoundError(f"{reference} not found: {e}") from e

    return _install(str(reference), reference.model_name, cache_dir, force, download)


def _archive_extension(url):
    path = urlparse(url).path.lower()
    return next((extension for extension in ARCHIVE_EXTENSIONS if path.endswith(extension)), None)


def _download_file(url, path, token):
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, stream=True, headers=headers, timeout=(10, 120)) as response:
        if response.status_code in (404, 410):
            raise HubModelNotFoundError(f"{url} not found: HTTP {response.status_code}")
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _extract(archive, directory):
    """Extract a zip or tar archive, refusing the members that would land out of ``directory``."""
    root = os.path.realpath(directory)

    def check(name):
        if os.path.commonpath([root, os.path.realpath(os.path.join(root, name))]) != root:
            raise ValueError(f"{name!r} of the archive would be extracted out of its directory")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zip_file:
            for name in zip_file.namelist():
                check(name)
            zip_file.extractall(directory)
        return

    with tarfile.open(archive) as tar_file:
        members = tar_file.getmembers()
        for member in members:
            check(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"{member.name!r} of the archive is neither a file nor a directory")
        # the filter of Python, where it has one, on top of the checks above
        filtered = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar_file.extractall(directory, members=members, **filtered)


def _model_root(directory):
    """The directory of the model in an extracted archive: the archive itself, or the
    only folder it holds."""
    if os.path.isfile(os.path.join(directory, CONFIG_FILE_NAME)):
        return directory
    folders = [entry.path for entry in os.scandir(directory) if entry.is_dir()]
    if len(folders) == 1 and os.path.isfile(os.path.join(folders[0], CONFIG_FILE_NAME)):
        return folders[0]
    return None


def _resolve_archive(url, model_name, cache_dir, token, force):
    extension = _archive_extension(url)
    if extension is not None:
        # the archive is named after the model, as its directory is
        file_name = unquote(os.path.basename(urlparse(url).path))
        model_name = file_name[: -len(extension)]
    elif model_name:
        extension = DEFAULT_ARCHIVE_EXTENSION
        url = url.rstrip("/") + "/" + quote(model_name) + extension
    if not model_name:
        raise ValueError(
            f"{url} names no model, expected the URL of an archive ({', '.join(ARCHIVE_EXTENSIONS)}) "
            "named after the model"
        )

    def download(directory):
        archive = os.path.join(directory, "archive" + extension)
        extracted = os.path.join(directory, "extracted")
        _download_file(url, archive, token)
        try:
            _extract(archive, extracted)
        except (zipfile.BadZipFile, tarfile.TarError) as e:
            raise ValueError(f"{url} is not an archive of a model: {e}") from e
        model_root = _model_root(extracted)
        if model_root is None:
            raise HubModelNotFoundError(
                f"{url} not found: the archive holds no {CONFIG_FILE_NAME}, at its root or in a single folder"
            )
        os.replace(model_root, os.path.join(directory, model_name))

    return _install(url, model_name, cache_dir, force, download)


def list_models(location, token=None) -> List[str]:
    """Names of the models of a repository or of a bucket, sorted."""
    location = parse_reference(location)
    if location.bucket:
        from huggingface_hub import list_bucket_tree

        entries = list_bucket_tree(location.repo_id, recursive=False, token=token)
        folders = [entry.path for entry in entries if entry.type == "directory"]
    else:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFolder

        entries = HfApi().list_repo_tree(location.repo_id, revision=location.revision, token=token)
        folders = [entry.path for entry in entries if isinstance(entry, RepoFolder)]
    return sorted(folder.strip("/") for folder in folders)


def select_models(model_names, architecture=None, suffix=None, match=None) -> List[str]:
    """
    The names among ``model_names`` that

    - have the given ``architecture``, with any suffix or none;
    - and the given ``suffix``, the empty string selecting the models without one;
    - and match the shell-style pattern ``match``.

    Names are split rather than matched for an architecture: the pattern
    ``*-BidLSTM_CRF-*`` misses the model without a suffix, and ``*-BidLSTM_CRF*`` takes
    ``BidLSTM_CRF_FEATURES`` along.
    """
    selected = []
    for model_name in model_names:
        if architecture is not None or suffix is not None:
            split = split_model_name(model_name)
            if split is None:
                continue
            if architecture is not None and split.architecture != architecture:
                continue
            if suffix is not None and (split.suffix or "") != suffix:
                continue
        if match is not None and not fnmatch.fnmatchcase(model_name, match):
            continue
        selected.append(model_name)
    return selected


def pull_models(location, cache_dir=None, architecture=None, suffix=None, match=None, token=None, force=False):
    """Download the models of a repository or bucket that ``select_models`` selects, all
    of them without a selection, and return their local directories."""
    location = parse_reference(location)
    model_names = select_models(list_models(location, token=token), architecture, suffix, match)
    return [
        resolve_model(location.for_model(name), cache_dir=cache_dir, token=token, force=force) for name in model_names
    ]


def hub_references(model_name, registry) -> List[HubReference]:
    """
    Where the resources registry says the model ``model_name`` is on the Hub, in the
    order to look: its repository, then the bucket. The section reads

        "models-hub": {
            "repo": "lfoppiano/grobid-model-{short_name}",
            "revision": "v1.1.0",
            "bucket": "lfoppiano/delft-models",
            "overrides": {"datasets": "lfoppiano/delft-model-datasets"}
        }

    ``repo`` is the repository of the models whose name starts with "grobid-", given
    their short name. ``overrides`` gives the repository of the others, and of any model
    to take from somewhere else, by their name or the start of it up to a dash, the
    longest start winning.
    Every key is optional, and so is the section.
    """
    section = (registry or {}).get(REGISTRY_SECTION) or {}
    references = []

    repo_id = None
    overrides = section.get("overrides") or {}
    for start in sorted(overrides, key=len, reverse=True):
        # up to a dash, so that an architecture does not match the longer ones it starts
        if model_name == start or model_name.startswith(start + "-"):
            repo_id = overrides[start]
            break
    if repo_id is None and section.get("repo"):
        split = split_model_name(model_name)
        if split is not None and split.prefix:
            repo_id = section["repo"].format(short_name=split.short_name)
    if repo_id is not None:
        references.append(HubReference(repo_id, model_name, section.get("revision") or None))

    if section.get("bucket"):
        references.append(HubReference(section["bucket"], model_name, bucket=True))
    return references


def fetch_model(model_name, models_dir, registry, token=None):
    """
    Return the directory of the model ``model_name`` in ``models_dir``, downloading it
    from the first place of the Hub it is found at when it is not there, or None when
    the registry gives the model no place on the Hub. This is what the wrappers load
    a model through.

    A model on disk that was not downloaded is never touched. One that was is downloaded
    again when the registry no longer names the place it came from, which is how a new
    ``revision`` in the registry reaches the models already on disk.
    """
    references = hub_references(model_name, registry)
    if not references:
        return None

    target = os.path.join(models_dir, model_name)
    if _is_model_directory(target):
        source = downloaded_from(target)
        if source is None or source in [str(reference) for reference in references]:
            return target

    failures = []
    for reference in references:
        try:
            return resolve_model(reference, cache_dir=models_dir, token=token)
        except HubModelNotFoundError as e:
            failures.append(str(e))
    raise HubModelNotFoundError(
        f"Model {model_name} is not in {models_dir} and could not be downloaded: " + "; ".join(failures)
    )
