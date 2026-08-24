# Manage pre-trained embeddings
import gzip
import hashlib
import io
import json
import mmap
import os
import pickle
import shutil
import struct
import sys
import threading
import zipfile

import lmdb
import numpy as np
from tqdm import tqdm

# for fasttext binary embeddings
fasttext_support = True
try:
    import fastText
except ImportError:
    fasttext_support = False

from delft.utilities.StaticEmbeddings import (
    StaticTransformerEmbeddings,
    looks_like_static_embedding_reference,
)
from delft.utilities.Utilities import download_file

# this is the default init size of a lmdb database for embeddings
map_size = 100 * 1024 * 1024 * 1024

# modern static embeddings (sentence-transformers static embeddings, Model2Vec
# potion models) are not a vector file to be compiled into LMDB, they are a
# tokenizer plus an embedding matrix loaded from the HuggingFace hub, see
# delft/utilities/StaticEmbeddings.py
STATIC_TRANSFORMER_FORMAT = "static-transformer"
STATIC_TRANSFORMER_FORMATS = (STATIC_TRANSFORMER_FORMAT, "hf")


def is_static_transformer_description(description):
    """Whether an embeddings registry entry describes a modern static embedding model."""
    if not isinstance(description, dict):
        return False
    return (
        description.get("format") in STATIC_TRANSFORMER_FORMATS or description.get("type") in STATIC_TRANSFORMER_FORMATS
    )


# Since py-lmdb 1.0, opening the same environment twice in a process raises
#   lmdb.Error: The environment '<path>' is already open in this process.
# whatever the readonly/lock flags. Embedding databases are read-only and
# shareable, and hosts like GROBID legitimately need several Embeddings over the
# same database: one Sequence model is loaded per document structure, all in a
# single JEP SharedInterpreter. So environments are cached per resolved path and
# reused, which also keeps a single reader locktable for the whole process.
#
# The pid is stored next to the environment: a handle inherited through fork()
# is unusable in the child but still occupies a slot in py-lmdb's registry
# there, so it has to be closed before a fresh one can be opened.
_lmdb_env_lock = threading.RLock()
_lmdb_envs = {}


def _lmdb_env_key(path):
    return os.path.realpath(path)


def open_lmdb_env(path, **kwargs):
    """
    Open an LMDB environment, or reuse the one already open for this path in
    this process.

    Returns a (environment, opened) tuple, where `opened` is False when an
    existing environment was handed back instead of a newly opened one.
    """
    key = _lmdb_env_key(path)
    pid = os.getpid()
    with _lmdb_env_lock:
        cached = _lmdb_envs.get(key)
        if cached is not None:
            cached_pid, cached_env = cached
            if cached_pid == pid:
                return cached_env, False
            # inherited through fork(): release the slot before reopening
            del _lmdb_envs[key]
            try:
                cached_env.close()
            except Exception:
                pass
        env = lmdb.open(path, **kwargs)
        _lmdb_envs[key] = (pid, env)
        return env, True


def close_lmdb_env(path):
    """Close and forget the environment cached for this path, if any."""
    with _lmdb_env_lock:
        cached = _lmdb_envs.pop(_lmdb_env_key(path), None)
    if cached is not None:
        try:
            cached[1].close()
        except Exception:
            pass


def current_lmdb_env(path):
    """The environment currently cached for this path in this process, or None."""
    with _lmdb_env_lock:
        cached = _lmdb_envs.get(_lmdb_env_key(path))
    if cached is None or cached[0] != os.getpid():
        return None
    return cached[1]


class Embeddings(object):
    def __init__(
        self,
        name,
        resource_registry=None,
        lang="en",
        extension="vec",
        use_cache=True,
        load=True,
    ):
        self.name = name
        self.embed_size = 0
        self.static_embed_size = 0
        self.vocab_size = 0
        self.model = {}
        self.registry = resource_registry
        self.lang = lang
        self.extension = extension
        self.embedding_lmdb_path = None
        if self.registry is not None and "embedding-lmdb-path" in self.registry:
            self.embedding_lmdb_path = self.registry["embedding-lmdb-path"]
        self.env = None
        if load:
            self.make_embeddings_simple(name)
        self.static_embed_size = self.embed_size

        self.use_cache = use_cache

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.has_lmdb_env():
            self.env, _ = open_lmdb_env(
                self.lmdb_env_path(),
                readonly=True,
                max_readers=2048,
                max_spare_txns=2,
                lock=False,
            )
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for _key, value in cursor:
                    _check_lmdb_format(value)
                    break
                cursor.close()

    def lmdb_env_path(self):
        """Path of the LMDB database backing these embeddings, or None."""
        if self.extension == STATIC_TRANSFORMER_FORMAT:
            # a static embedding model is never compiled into LMDB, and its
            # name may be a path, which os.path.join below would take as the
            # database directory
            return None
        if not self.embedding_lmdb_path:
            return None
        return os.path.join(self.embedding_lmdb_path, self.name)

    def has_lmdb_env(self):
        """Whether an LMDB database exists on disk for these embeddings."""
        envFilePath = self.lmdb_env_path()
        return envFilePath is not None and os.path.isdir(envFilePath)

    def reopen_lmdb(self):
        """
        Reopen the LMDB environment for fork-safe multiprocessing. Call from a
        DataLoader worker_init_fn so each worker gets a fresh handle. No-op when
        embeddings aren't backed by LMDB.

        A handle inherited from the parent is recognised by pid and closed by
        open_lmdb_env, so when several Embeddings share one database and each
        calls this in the same worker, the database is reopened once and they
        all get the same handle.
        """
        if not self.has_lmdb_env():
            return
        self.env, _ = open_lmdb_env(
            self.lmdb_env_path(),
            readonly=True,
            max_readers=2048,
            max_spare_txns=2,
            lock=False,
        )

    def make_embeddings_simple_in_memory(self, name="fasttext-crawl"):
        nbWords = 0
        print("loading embeddings...")
        begin = True
        description = self.get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            self.lang = description["lang"]
            print("path:", embeddings_path)
            if self.extension == "bin":
                self.model = fastText.load_model(embeddings_path)
                nbWords = len(self.model.get_words())
                self.embed_size = self.model.get_dimension()
            else:
                with open(embeddings_path, encoding="utf8") as f:
                    for line in f:
                        line = line.strip().split(" ")
                        if begin:
                            begin = False
                            nb_words, embed_size = _fetch_header_if_available(line)

                            # we parse the header
                            if nb_words > 0 and embed_size > 0:
                                nbWords = nb_words
                                self.embed_size = embed_size
                                continue

                        word = line[0]
                        vector = np.array([float(val) for val in line[1 : len(line)]], dtype="float32")
                        if self.embed_size == 0:
                            self.embed_size = len(vector)
                        self.model[word] = vector
                if nbWords == 0:
                    nbWords = len(self.model)
            print(
                "embeddings loaded for",
                nbWords,
                "words and",
                self.embed_size,
                "dimensions",
            )

    def make_embeddings_lmdb(self, name="fasttext-crawl"):
        print("\nCompiling embeddings... (this is done only one time per embeddings at first usage)")
        description = self.get_description(name)

        if description is None:
            print("\nNo description found in embeddings registry for embeddings", name)
            return

        if description is not None:
            # the following method will possibly download the mebedding file if not available locally
            embeddings_path = self.get_embedding_path(description)
            if embeddings_path is None:
                print("\nCould not locate a usable resource for embeddings", name)
                return

            self.load_embeddings_from_file(embeddings_path)

        # cleaning possible downloaded embeddings
        self.clean_downloads()

    def load_embeddings_from_file(self, embeddings_path):
        begin = True
        nbWords = 0
        txn = self.env.begin(write=True)
        # batch_size = 1024
        i = 0
        nb_lines = 0

        # read number of lines first
        embedding_file = open_embedding_file(embeddings_path)
        if embedding_file is None:
            print("Error: could not open embeddings file", embeddings_path)
            return

        for line in embedding_file:
            nb_lines += 1
        embedding_file.close()

        embedding_file = open_embedding_file(embeddings_path)
        # with open(embeddings_path, encoding='utf8') as f:
        for line in tqdm(embedding_file, total=nb_lines):
            line = line.decode()
            line = line.split(" ")
            if begin:
                begin = False
                nb_words, embed_size = _fetch_header_if_available(line)

                if nb_words > 0 and embed_size > 0:
                    nbWords = nb_words
                    self.embed_size = embed_size
                    continue

            word = line[0]
            try:
                if line[len(line) - 1] == "\n":
                    vector = np.array([float(val) for val in line[1 : len(line) - 1]], dtype="float32")
                else:
                    vector = np.array([float(val) for val in line[1 : len(line)]], dtype="float32")

            except Exception:
                print(len(line))
                print(line[1 : len(line)])

            if self.embed_size == 0:
                self.embed_size = len(vector)

            if len(word.encode(encoding="UTF-8")) < self.env.max_key_size():
                txn.put(word.encode(encoding="UTF-8"), _serialize_float32(vector))
                i += 1

        embedding_file.close()

        # if i % batch_size != 0:
        txn.commit()
        if nbWords == 0:
            nbWords = i
        self.vocab_size = nbWords
        print("embeddings loaded for", nbWords, "words and", self.embed_size, "dimensions")

    def clean_downloads(self):
        # cleaning possible downloaded embeddings
        if (
            "embedding-download-path" in self.registry
            and os.path.exists(self.registry["embedding-download-path"])
            and os.path.isdir(self.registry["embedding-download-path"])
        ):
            for filename in os.listdir(self.registry["embedding-download-path"]):
                file_path = os.path.join(self.registry["embedding-download-path"], filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))

    def make_static_transformer_embeddings(self, name, description=None):
        """
        Load a modern static embedding model (sentence-transformers static
        embeddings, Model2Vec potion models), either described in the
        embeddings registry or given directly as a HuggingFace hub identifier
        or as a path to a local copy of such a model.
        """
        if description is None:
            description = {}
        model_reference = description.get("model") or description.get("path") or name
        self.lang = description.get("lang", self.lang)
        self.extension = STATIC_TRANSFORMER_FORMAT

        print("loading static embedding model", model_reference, "...")
        self.model = StaticTransformerEmbeddings(
            model_reference,
            dimensions=description.get("dimensions"),
            normalize=description.get("normalize"),
            lang=self.lang,
        )
        self.embed_size = self.model.embed_size
        self.vocab_size = self.model.vocab_size
        print(
            "embeddings loaded for",
            self.vocab_size,
            "sub-word units and",
            self.embed_size,
            "dimensions",
        )

    def make_embeddings_simple(self, name="fasttext-crawl"):
        description = self.get_description(name)
        if description is not None:
            self.extension = description.get("format", self.extension)

        if is_static_transformer_description(description) or (
            description is None and looks_like_static_embedding_reference(name)
        ):
            # a static embedding model is used as it is, there is nothing to
            # compile into LMDB: the vocabulary is made of sub-word units and
            # word vectors are pooled on the fly
            self.make_static_transformer_embeddings(name, description)

        elif self.extension == "bin":
            if fasttext_support:
                print("embeddings are of .bin format, so they will be loaded in memory...")
                self.make_embeddings_simple_in_memory(name)
            else:
                if not (sys.platform == "linux" or sys.platform == "darwin"):
                    raise ValueError("FastText .bin format not supported for your platform")
                else:
                    raise ValueError(
                        "Go to the documentation to get more information on how to install FastText .bin support"
                    )

        elif self.embedding_lmdb_path is None or self.embedding_lmdb_path == "None":
            print(
                "embedding_lmdb_path is not specified in the embeddings registry, so the embeddings will be loaded in memory..."
            )
            embeddings_path = None
            if "path" in description:
                embeddings_path = description["path"]
            self.lang = description["lang"]
            if embeddings_path is None or not os.path.isfile(embeddings_path):
                raise ValueError(
                    "Embedding path for",
                    description["name"],
                    "is not valid",
                    embeddings_path,
                )

            self.make_embeddings_simple_in_memory(name)

        else:
            # if the path to the lmdb database files does not exist, we create it
            if not os.path.isdir(self.embedding_lmdb_path):
                # conservative check (likely very useless)
                if not os.path.exists(self.embedding_lmdb_path):
                    os.makedirs(self.embedding_lmdb_path, exist_ok=True)

            # check if the lmdb database exists
            envFilePath = os.path.join(self.embedding_lmdb_path, name)
            load_db = True
            if os.path.isdir(envFilePath):
                description = self.get_description(name)
                if description is not None:
                    self.lang = description["lang"]

                # open the database in read mode, or reuse the environment
                # already open on this path in the current process
                self.env, opened = open_lmdb_env(envFilePath, readonly=True, max_readers=2048, max_spare_txns=4)
                if self.env:
                    try:
                        # we need to set self.embed_size and self.vocab_size
                        with self.env.begin() as txn:
                            stats = txn.stat()
                            size = stats["entries"]
                            self.vocab_size = size

                        with self.env.begin() as txn:
                            cursor = txn.cursor()
                            for key, value in cursor:
                                _check_lmdb_format(value)
                                vector = _deserialize_float32(value)
                                self.embed_size = vector.shape[0]
                                break
                            cursor.close()
                    except Exception:
                        # never leave an environment behind on the way out: the
                        # real cause (a legacy-format database, say) would then
                        # be masked by "already open in this process" on the
                        # next attempt
                        if opened:
                            close_lmdb_env(envFilePath)
                            self.env = None
                        raise

                    if self.vocab_size > 100 and self.embed_size > 10:
                        # lmdb database exists and looks valid
                        load_db = False

                        if opened:
                            # no idea why, but we need to close and reopen the environment to avoid
                            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
                            # when opening new transaction !
                            # Only for an environment we opened ourselves: a reused
                            # one already went through this and is read by others.
                            close_lmdb_env(envFilePath)
                            self.env, _ = open_lmdb_env(
                                envFilePath,
                                readonly=True,
                                max_readers=2048,
                                max_spare_txns=2,
                            )

            if load_db:
                # create and load the database in write mode. The check above may
                # have left a read-only environment open on this path, and write
                # mode needs an exclusive open.
                close_lmdb_env(envFilePath)
                self.env, _ = open_lmdb_env(envFilePath, map_size=map_size)
                self.make_embeddings_lmdb(name)

    def get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        for emb in self.registry["embeddings-contextualized"]:
            if emb["name"] == name:
                return emb
        for emb in self.registry["transformers"]:
            if emb["name"] == name:
                return emb
        return None

    def get_word_vector(self, word, _retry=True):
        """
        Get static embeddings (e.g. glove) for a given token
        """
        if self.extension == STATIC_TRANSFORMER_FORMAT:
            return self.model.get_word_vector(word)
        if (self.name == "wiki.fr") or (self.name == "wiki.fr.bin"):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.env is None or self.extension == "bin":
            # db not available or embeddings in bin format, the embeddings should be available in memory (normally!)
            return self.get_word_vector_in_memory(word)
        try:
            with self.env.begin() as txn:
                vector = txn.get(word.encode(encoding="UTF-8"))
                if vector:
                    word_vector = _deserialize_float32(vector)
                    vector = None
                else:
                    word_vector = np.zeros((self.static_embed_size,), dtype=np.float32)
                    # alternatively, initialize with random negative values
                    # word_vector = np.random.uniform(low=-0.5, high=0.0, size=(self.embed_size,))
                    # alternatively use fasttext OOV ngram possibilities (if ngram available)
        except lmdb.Error:
            if not _retry:
                # a second failure is a real problem, not a stale handle
                raise
            self.env = self.recover_lmdb_env()
            return self.get_word_vector(word, _retry=False)
        return word_vector

    def recover_lmdb_env(self):
        """
        Recover the LMDB environment after a failed transaction, typically
        mdb_txn_begin: MDB_BAD_RSLOT (invalid reuse of a reader locktable slot),
        or a handle closed by another holder.

        The environment is shared process-wide, so several Embeddings instances
        may hold the same object. Only the caller still holding the current
        handle reopens it; a caller holding a superseded one adopts the
        replacement, otherwise concurrent readers would close each other's
        freshly opened environment in turn.
        """
        envFilePath = self.lmdb_env_path()
        with _lmdb_env_lock:
            current = current_lmdb_env(envFilePath)
            if current is not None and current is not self.env:
                # already reopened by another instance or another thread
                return current
            close_lmdb_env(envFilePath)
            env, _ = open_lmdb_env(
                envFilePath,
                readonly=True,
                max_readers=2048,
                max_spare_txns=2,
                lock=False,
            )
            return env

    def get_word_vector_in_memory(self, word):
        if self.extension == STATIC_TRANSFORMER_FORMAT:
            return self.model.get_word_vector(word)
        if (self.name == "wiki.fr") or (self.name == "wiki.fr.bin"):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.extension == "bin":
            return self.model.get_word_vector(word)
        if word in self.model:
            return self.model[word]
        else:
            # for unknown word, we use a vector filled with 0.0
            return np.zeros((self.static_embed_size,), dtype=np.float32)

    def get_embedding_path(self, description):
        embeddings_path = None
        if "path" in description:
            embeddings_path = description["path"]
        self.lang = description["lang"]

        if embeddings_path is None or not os.path.isfile(embeddings_path):
            print(
                "error: embedding path for",
                description["name"],
                "is not valid",
                embeddings_path,
            )
            if "url" in description and len(description["url"]) > 0:
                url = description["url"]
                download_path = self.registry["embedding-download-path"]
                # if the download path does not exist, we create it
                if not os.path.isdir(download_path):
                    try:
                        os.mkdir(download_path)
                    except OSError:
                        print(
                            "Creation of the download directory",
                            download_path,
                            "failed",
                        )

                print("Downloading resource file for", description["name"], "...")
                embeddings_path = download_file(url, download_path)
                if embeddings_path is not None and os.path.isfile(embeddings_path):
                    print("Download sucessful:", embeddings_path)
            else:
                print(
                    "no download url available for this embeddings resource, please review the embedding registry for",
                    description["name"],
                )
        return embeddings_path


def _serialize_byteio(array):
    memfile = io.BytesIO()
    np.save(memfile, array)
    memfile.seek(0)
    return memfile.getvalue()


def _deserialize_byteio(serialized):
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile)


def _serialize_pickle(a):
    """Legacy: pickle serialization (for reading old databases)"""
    return pickle.dumps(a)


def _deserialize_pickle(serialized):
    """Legacy: pickle deserialization (for reading old databases)"""
    return pickle.loads(serialized)


def _serialize_float32(array):
    """
    Serialize numpy array to raw float32 bytes.
    This format is readable by both Python and Java.
    """
    return array.astype(np.float32).tobytes()


def _deserialize_float32(serialized):
    """
    Deserialize raw float32 bytes to numpy array.
    This format is readable by both Python and Java.
    """
    return np.frombuffer(serialized, dtype=np.float32)


def _check_lmdb_format(value):
    """
    Verify that an LMDB value contains raw float32 bytes, not the legacy
    serialized-numpy format. Two signals must both be present to avoid false
    positives on raw float32 data that happens to start with 0x80:

    1. pickle magic byte (0x80) followed by protocol version 2-5
    2. ``b"numpy"`` substring in the first 50 bytes (always present in
       serialized numpy arrays produced by older DeLFT versions)

    Raises ValueError with a clear, actionable message when both signals match.
    """
    if len(value) < 2:
        return

    if value[0] == 0x80 and value[1] in (2, 3, 4, 5) and b"numpy" in value[:50]:
        raise ValueError(
            "LMDB embedding database is in the legacy DeLFT format and is "
            "incompatible with the current raw float32 format. Reading it "
            "would produce garbage vectors. Delete the database directory "
            "and re-run; DeLFT will rebuild it from the source embedding "
            "file in the new format."
        )


def open_embedding_file(embeddings_path):
    # embeddings can be uncompressed or compressed with gzip or zip
    if embeddings_path.endswith(".gz"):
        embedding_file = gzip.open(embeddings_path, mode="rb")
    elif embeddings_path.endswith(".zip"):
        zip_file = zipfile.ZipFile(embeddings_path, "r")
        for filename in zip_file.namelist():
            print(filename)
            if filename.endswith("vec") or filename.endswith("txt"):
                embedding_file = zip_file.open(filename, mode="r")
    else:
        embedding_file = open(embeddings_path, mode="rb")
    return embedding_file


def _get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    fp.close()
    return lines


def list_digest(strings):
    hash = hashlib.sha1()
    for s in strings:
        hash.update(struct.pack("I", len(s)))
        hash.update(s.encode(encoding="UTF-8"))
    return hash.hexdigest()


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _fetch_header_if_available(line):
    """
    Fetch the header if the line is just composed by two elements the number of workds and the embedding size

    :param line: a splitted line (if not split, tried to split by spaces)
    :return: the number of words and the embedding size, if there is no header, they will be set to -1
    """
    if isinstance(line, str):
        line = line.split(" ")

    nb_words = -1
    embed_size = -1

    if len(line) == 2:
        # first line gives the nb of words and the embedding size
        nb_words = int(line[0])
        embed_size = int(line[1].replace("\n", ""))

    return nb_words, embed_size


def load_resource_registry(path="delft/resources-registry.json"):
    """
    Load the resource registry file in memory. Each description provides a name,
    a file path (used only if necessary) and an embeddings type (to take into account
    small variation of format)
    """
    registry_json = open(path).read()
    return json.loads(registry_json)
