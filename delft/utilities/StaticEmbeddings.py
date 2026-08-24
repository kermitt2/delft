"""
Modern static embeddings, i.e. token embedding matrices distilled from a
transformer and shipped on the Hugging Face hub as a plain embedding matrix
plus a fast tokenizer.

Two families are supported, they only differ by their file layout:

- the sentence-transformers *static embeddings*
  (``sentence-transformers/static-retrieval-mrl-en-v1``,
  ``sentence-transformers/static-similarity-mrl-multilingual-v1``, see
  https://huggingface.co/blog/static-embeddings), which store the matrix under
  ``0_StaticEmbedding/``,
- the Model2Vec *potion* models (``minishlab/potion-base-8M`` and friends),
  which store it at the root of the repository.

Compared to the static embeddings historically used by DeLFT (glove, word2vec,
fasttext .vec), these are one to two orders of magnitude smaller, have no
out-of-vocabulary words at all - a word absent from the vocabulary is split
into sub-word units by the tokenizer - and report clearly better quality on
retrieval and similarity benchmarks.

A word vector is obtained the same way the original models produce a text
embedding: tokenize the word without special tokens, then average the vectors
of the resulting sub-word units (and L2-normalize when the model asks for it).
Lookups are memoized, so the tokenization cost is paid once per distinct word.

The matryoshka (MRL) models allow the vectors to be truncated to a smaller
number of dimensions at almost no quality cost, which directly reduces the
size of the RNN input layer: pass ``dimensions`` (or set ``dimensions`` in the
embeddings registry entry) to do so.
"""

import json
import os
import threading

import numpy as np

# candidate locations of the tokenizer and of the embedding matrix, in the
# order they are looked up: root first (Model2Vec), then the
# sentence-transformers static embedding module directory
TOKENIZER_FILES = ("tokenizer.json", "0_StaticEmbedding/tokenizer.json")
WEIGHT_FILES = ("model.safetensors", "0_StaticEmbedding/model.safetensors")

# name of the embedding matrix inside the safetensors file, per family
EMBEDDING_TENSOR_KEYS = (
    "embeddings",
    "embedding.weight",
    "embedding_bag.weight",
)

CONFIG_FILE = "config.json"
MODULES_FILE = "modules.json"

# maximum number of distinct words kept in the word vector cache, a bit above
# the vocabulary size of a large training corpus
DEFAULT_CACHE_SIZE = 300000


def _import_tokenizers():
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError(
            "the tokenizers library is required to use modern static embeddings, "
            "it is normally installed together with transformers"
        ) from e
    return Tokenizer


def _import_huggingface_hub():
    """
    The hub downloader, the exception raised when a file is not in a
    repository, and the one raised when it could not be fetched at all.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "the huggingface_hub library is required to download static embeddings, "
            "it is normally installed together with transformers"
        ) from e
    try:
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError:
        try:
            from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        except ImportError:
            EntryNotFoundError = ()
            LocalEntryNotFoundError = ()
    return hf_hub_download, EntryNotFoundError, LocalEntryNotFoundError


def _import_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "the safetensors library is required to use modern static embeddings, "
            "it is normally installed together with transformers"
        ) from e
    return safe_open


class StaticTransformerEmbeddings:
    """
    Word embeddings backed by a static embedding model from the Hugging Face
    hub (or from a local copy of such a model).

    ``model_reference`` is either a hub identifier (``minishlab/potion-base-8M``)
    or a local directory containing the model files.
    """

    def __init__(
        self,
        model_reference,
        dimensions=None,
        normalize=None,
        cache_size=DEFAULT_CACHE_SIZE,
        local_files_only=False,
        lang="en",
    ):
        self.model_reference = model_reference
        self.dimensions = dimensions
        self.normalize_override = normalize
        self.cache_size = cache_size
        self.local_files_only = local_files_only
        self.lang = lang

        self.embed_size = 0
        self.vocab_size = 0
        self.normalize = False

        self._tokenizer = None
        self._matrix = None
        self._cache = {}
        self._download_errors = []
        self._load_lock = threading.Lock()

        self.load()

    def __getstate__(self):
        """
        Drop the embedding matrix, the tokenizer and the memoized vectors when
        the object is serialized: a DataLoader worker process would otherwise
        receive its own full copy of the matrix. They are rebuilt from the
        (already downloaded) model files on the other side.
        """
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_matrix"] = None
        state["_cache"] = {}
        state["_download_errors"] = []
        state["_load_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_lock = threading.Lock()
        self.load()

    def __repr__(self):
        return "StaticTransformerEmbeddings(%s, dimensions=%d, vocab_size=%d, normalize=%s)" % (
            self.model_reference,
            self.embed_size,
            self.vocab_size,
            self.normalize,
        )

    def load(self):
        """Resolve the model files (downloading them if needed) and load them."""
        with self._load_lock:
            if self._matrix is not None and self._tokenizer is not None:
                return
            tokenizer_path, weights_path, config, modules = self._resolve_model_files()

            Tokenizer = _import_tokenizers()
            tokenizer = Tokenizer.from_file(tokenizer_path)

            matrix = self._read_embedding_matrix(weights_path)
            if matrix.ndim != 2:
                raise ValueError(
                    "the embedding matrix of %s has %d dimensions, 2 expected" % (self.model_reference, matrix.ndim)
                )
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)

            if self.dimensions is not None:
                if self.dimensions < 1 or self.dimensions > matrix.shape[1]:
                    raise ValueError(
                        "cannot truncate the embeddings of %s to %s dimensions, the model provides %d"
                        % (self.model_reference, self.dimensions, matrix.shape[1])
                    )
                # matryoshka truncation: the first dimensions carry most of the
                # information, so the vectors can simply be cut
                matrix = np.ascontiguousarray(matrix[:, : self.dimensions])

            self._tokenizer = tokenizer
            self._matrix = matrix
            self.vocab_size = matrix.shape[0]
            self.embed_size = matrix.shape[1]
            self.normalize = self._resolve_normalize(config, modules)
            self._cache = {}

    def _resolve_model_files(self):
        """
        Locate ``tokenizer.json`` and the safetensors embedding matrix, either
        in a local directory or in a hub repository, and read the optional
        configuration files. Returns (tokenizer path, weights path, config,
        modules).
        """
        reference = self.model_reference
        if reference is None or len(str(reference).strip()) == 0:
            raise ValueError("no model reference provided for static embeddings")

        if os.path.isdir(reference):
            tokenizer_path = _first_existing_file(reference, TOKENIZER_FILES)
            weights_path = _first_existing_file(reference, WEIGHT_FILES)
            if tokenizer_path is None or weights_path is None:
                raise ValueError(
                    "%s does not look like a static embedding model, "
                    "a tokenizer.json and a model.safetensors file are expected "
                    "(at the root of the directory or under 0_StaticEmbedding/)" % reference
                )
            config = _read_json_file(os.path.join(reference, CONFIG_FILE))
            modules = _read_json_file(os.path.join(reference, MODULES_FILE))
            return tokenizer_path, weights_path, config, modules

        self._download_errors = []
        tokenizer_path = self._download_first_available(TOKENIZER_FILES)
        weights_path = self._download_first_available(WEIGHT_FILES)
        if tokenizer_path is None or weights_path is None:
            message = (
                "could not find a static embedding model at %s, a tokenizer.json and a "
                "model.safetensors file are expected (at the root of the repository or "
                "under 0_StaticEmbedding/)" % reference
            )
            if len(self._download_errors) > 0:
                message += (
                    ". The model files could not be retrieved, check the network connection or, "
                    "when working offline, that the model is in the HuggingFace cache (HF_HOME): %s"
                    % self._download_errors[0]
                )
            raise ValueError(message)
        config = _read_json_file(self._download_optional(CONFIG_FILE))
        modules = _read_json_file(self._download_optional(MODULES_FILE))
        return tokenizer_path, weights_path, config, modules

    def _download_first_available(self, filenames):
        for filename in filenames:
            path = self._download_optional(filename)
            if path is not None:
                return path
        return None

    def _download_optional(self, filename):
        """
        Download a file from the hub, returning None when it is not there. A
        file simply absent from the repository is expected - the two supported
        layouts are tried in turn - anything else (no network, unknown
        repository, nothing in the local cache in offline mode) is kept to be
        reported if no model can be loaded at all.
        """
        hf_hub_download, entry_not_found, local_entry_not_found = _import_huggingface_hub()
        try:
            return hf_hub_download(
                repo_id=self.model_reference,
                filename=filename,
                local_files_only=self.local_files_only,
            )
        except local_entry_not_found as e:
            # nothing in the local cache, and the hub could not be reached
            self._download_errors.append(e)
            return None
        except entry_not_found:
            return None
        except Exception as e:
            self._download_errors.append(e)
            return None

    def _read_embedding_matrix(self, weights_path):
        safe_open = _import_safetensors()
        with safe_open(weights_path, framework="numpy") as weights:
            keys = list(weights.keys())
            name = next((key for key in EMBEDDING_TENSOR_KEYS if key in keys), None)
            if name is None:
                # unknown family: fall back on the only 2 dimensional tensor
                candidates = [key for key in keys if len(weights.get_slice(key).get_shape()) == 2]
                if len(candidates) != 1:
                    raise ValueError(
                        "could not identify the embedding matrix of %s among the tensors %s"
                        % (self.model_reference, keys)
                    )
                name = candidates[0]
            return weights.get_tensor(name)

    def _resolve_normalize(self, config, modules):
        """
        Whether pooled vectors must be L2-normalized. Model2Vec states it in
        config.json, sentence-transformers as a Normalize module in
        modules.json. An explicit value in the registry wins over both.
        """
        if self.normalize_override is not None:
            return bool(self.normalize_override)
        if isinstance(config, dict) and "normalize" in config:
            return bool(config["normalize"])
        if isinstance(modules, list):
            return any("Normalize" in str(module.get("type", "")) for module in modules)
        return False

    def get_word_vector(self, word):
        """
        Vector of a word: the average of the vectors of its sub-word units, as
        the model itself embeds a text. Words are never out-of-vocabulary, but
        a word that the tokenizer maps to nothing at all (an empty or blank
        string) gets a zero vector, as with the other DeLFT embeddings.
        """
        vector = self._cache.get(word)
        if vector is not None:
            return vector

        if self._matrix is None or self._tokenizer is None:
            self.load()

        ids = self._tokenizer.encode(word, add_special_tokens=False).ids
        if len(ids) == 0:
            vector = np.zeros((self.embed_size,), dtype=np.float32)
        else:
            vector = self._matrix[ids].mean(axis=0)
            if self.normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vector = np.asarray(vector, dtype=np.float32)

        if len(self._cache) < self.cache_size:
            self._cache[word] = vector
        return vector


def is_static_embedding_directory(path):
    """Whether a local directory holds a static embedding model."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    return (
        _first_existing_file(path, TOKENIZER_FILES) is not None and _first_existing_file(path, WEIGHT_FILES) is not None
    )


def looks_like_static_embedding_reference(name):
    """
    Whether a name that is absent from the embeddings registry can be handed
    over to this backend: a local static embedding model, or a Hugging Face
    hub identifier of the form ``organisation/model``.
    """
    if not isinstance(name, str) or len(name.strip()) == 0:
        return False
    if os.path.isdir(name):
        return is_static_embedding_directory(name)
    parts = name.split("/")
    return len(parts) == 2 and all(len(part) > 0 for part in parts) and " " not in name


def _first_existing_file(directory, filenames):
    for filename in filenames:
        path = os.path.join(directory, *filename.split("/"))
        if os.path.isfile(path):
            return path
    return None


def _read_json_file(path):
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
