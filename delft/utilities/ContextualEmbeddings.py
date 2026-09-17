"""
Contextual embeddings, i.e. the hidden states of a frozen transformer (BERT,
SciBERT, RoBERTa, ...) used as word vectors for the RNN architectures, the same
way ELMo embeddings were used in the past.

The transformer is *not* part of the trained model: it is a feature extractor.
This is what makes it different from the ``BERT*`` architectures, which
fine-tune the transformer and label its sub-word units. Here:

- vectors are produced **per word**: the sub-word units of a word are pooled
  back into one vector, so that a sentence of ``n`` tokens gives ``n`` vectors
  and the character, feature and label channels of the RNN architectures stay
  aligned, as with any static embeddings,
- sequences longer than what the transformer accepts are covered by
  **overlapping windows**, each sub-word unit taking its vector from the window
  where it is the furthest from an edge, i.e. where it has the most context on
  both sides. Sequence length is thus not bound by the transformer (the header
  model uses several thousands of tokens),
- as the transformer is frozen, the vectors of a sentence never change, so they
  are computed **once** and cached in LMDB (as float16). After the first pass,
  an epoch costs what it costs with static embeddings, and DataLoader workers
  only read the cache, they never load the transformer.

The vector of a word is, by default, the mean of the last four hidden layers of
its first sub-word unit.
"""

import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid

import numpy as np

# hidden layers used for the word vectors, as indices in the hidden states
# returned by the transformer (embedding layer first, last layer last)
DEFAULT_LAYERS = (-4, -3, -2, -1)

LAYER_POOLINGS = ("mean", "sum", "concat")
SUBWORD_POOLINGS = ("first", "last", "mean")

# size of a window in sub-word units, special tokens included, and step
# between two consecutive windows
DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 256

# number of windows sent to the transformer at once
DEFAULT_BATCH_SIZE = 32

# number of sentences embedded between two writes in the cache
PRECOMPUTE_CHUNK_SIZE = 256

# the vectors are cached as float16
CACHE_DTYPE = np.float16
CACHE_DTYPE_MAX = float(np.finfo(CACHE_DTYPE).max)

FINGERPRINT_FILE = "fingerprint.json"
SHARD_PREFIX = "shard-"
TEMPORARY_PREFIX = "tmp-"

# above this, the maximum length announced by a tokenizer is a placeholder
_UNBOUNDED_LENGTH = 100000


def _import_torch():
    import torch

    return torch


def _import_transformers():
    try:
        import transformers
    except ImportError as e:
        raise ImportError("the 'transformers' package is required to use contextual embeddings") from e
    return transformers


class ContextualEmbeddings:
    """
    Word embeddings taken from the hidden states of a frozen transformer.

    ``model_reference`` is a HuggingFace hub identifier or a local directory.
    ``cache_path`` is the directory under which the LMDB cache is created, when
    ``None`` the vectors are only kept in memory.
    """

    def __init__(
        self,
        model_reference,
        layers=DEFAULT_LAYERS,
        layer_pooling="mean",
        subword_pooling="first",
        window=DEFAULT_WINDOW,
        stride=DEFAULT_STRIDE,
        batch_size=DEFAULT_BATCH_SIZE,
        cache_path=None,
        device=None,
        local_files_only=False,
        lang="en",
    ):
        if layer_pooling not in LAYER_POOLINGS:
            raise ValueError("layer pooling must be one of %s, not %s" % (", ".join(LAYER_POOLINGS), layer_pooling))
        if subword_pooling not in SUBWORD_POOLINGS:
            raise ValueError(
                "sub-word pooling must be one of %s, not %s" % (", ".join(SUBWORD_POOLINGS), subword_pooling)
            )
        layers = tuple(int(layer) for layer in layers)
        if len(layers) == 0:
            raise ValueError("at least one hidden layer is required")

        self.model_reference = model_reference
        self.layers = layers
        self.layer_pooling = layer_pooling
        self.subword_pooling = subword_pooling
        self.requested_window = window
        self.requested_stride = stride
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.device_name = device
        self.local_files_only = local_files_only
        self.lang = lang

        self.embed_size = 0
        self.window = 0
        self.stride = 0

        self._tokenizer = None
        self._prefix_ids = []
        self._suffix_ids = []
        self._prefix_length = 0
        self._special_tokens = 0
        self._model = None
        self._device = None
        self._shards = {}
        self._shards_pid = None
        self._memory = {}
        self._warned_about_worker = False
        self._lock = threading.RLock()

        self._load_configuration()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _load_configuration(self):
        """
        Read what is needed to know the size of the vectors and to tokenize.
        The weights of the transformer are only loaded when some vectors have
        to be computed, which never happens when everything is in the cache.
        """
        transformers = _import_transformers()
        config = transformers.AutoConfig.from_pretrained(self.model_reference, local_files_only=self.local_files_only)
        hidden_size = config.hidden_size
        nb_hidden_states = config.num_hidden_layers + 1
        for layer in self.layers:
            if not -nb_hidden_states <= layer < nb_hidden_states:
                raise ValueError(
                    "%s has %d hidden states, layer %d does not exist" % (self.model_reference, nb_hidden_states, layer)
                )
        self.embed_size = hidden_size * len(self.layers) if self.layer_pooling == "concat" else hidden_size

        self._load_tokenizer()

        max_length = getattr(self._tokenizer, "model_max_length", None)
        if max_length is None or max_length > _UNBOUNDED_LENGTH:
            max_length = getattr(config, "max_position_embeddings", None) or DEFAULT_WINDOW
        self.window = min(self.requested_window, max_length)
        capacity = self.window - self._special_tokens
        if capacity < 1:
            raise ValueError("a window of %d sub-word units is too small for %s" % (self.window, self.model_reference))
        self.stride = max(1, min(self.requested_stride, capacity))

    def _load_tokenizer(self):
        transformers = _import_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_reference, local_files_only=self.local_files_only
        )
        try:
            tokenizer(["a"], is_split_into_words=True, add_special_tokens=False)
        except Exception:
            # byte-level BPE tokenizers (RoBERTa and friends) only accept
            # pre-tokenized input when they add a space before each word
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_reference, local_files_only=self.local_files_only, add_prefix_space=True
            )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "%s has no fast tokenizer, which is required to map sub-word units to words" % self.model_reference
            )
        self._tokenizer = tokenizer

        # special tokens the tokenizer puts before and after a sequence, the
        # windows are built by hand and need the same ones
        probe = tokenizer(["a"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        built = tokenizer(["a"], is_split_into_words=True, add_special_tokens=True)["input_ids"]
        self._prefix_ids, self._suffix_ids = [], []
        for position in range(len(built) - len(probe) + 1):
            if built[position : position + len(probe)] == probe:
                self._prefix_ids = list(built[:position])
                self._suffix_ids = list(built[position + len(probe) :])
                break
        self._prefix_length = len(self._prefix_ids)
        self._special_tokens = len(self._prefix_ids) + len(self._suffix_ids)

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return
            torch = _import_torch()
            transformers = _import_transformers()

            if torch.utils.data.get_worker_info() is not None and not self._warned_about_worker:
                self._warned_about_worker = True
                print(
                    "warning: contextual embeddings are computed in a DataLoader worker, "
                    "call precompute() beforehand to avoid loading the transformer in every worker"
                )

            if self.device_name is not None:
                device = torch.device(self.device_name)
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            print("loading transformer", self.model_reference, "for contextual embeddings on", device, "...")
            # fp32 is pinned, some checkpoints are shipped as fp16
            model = transformers.AutoModel.from_pretrained(
                self.model_reference, torch_dtype=torch.float32, local_files_only=self.local_files_only
            )
            model.eval()
            model.requires_grad_(False)
            model.to(device)
            self._model = model
            self._device = device

    def release_model(self):
        """Free the transformer, e.g. once the vectors of a corpus are cached."""
        with self._lock:
            self._model = None
            self._device = None

    def __getstate__(self):
        """
        Drop the transformer, the tokenizer and the LMDB handle when the object
        is serialized: a DataLoader worker only reads vectors that have been
        precomputed, from the LMDB cache or from the (kept) in-memory ones.
        """
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_model"] = None
        state["_device"] = None
        state["_shards"] = {}
        state["_shards_pid"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __repr__(self):
        return "ContextualEmbeddings(%s, layers=%s, layer_pooling=%s, subword_pooling=%s, dimensions=%d)" % (
            self.model_reference,
            list(self.layers),
            self.layer_pooling,
            self.subword_pooling,
            self.embed_size,
        )

    # ------------------------------------------------------------------
    # computing the vectors
    # ------------------------------------------------------------------

    def _tokenize(self, tokens):
        """Sub-word unit ids of a sentence and, for each of them, the index of its word."""
        if self._tokenizer is None:
            self._load_tokenizer()
        if len(tokens) == 0:
            return [], []
        # verbose=False: a sequence longer than the transformer accepts is
        # expected here, it is what the windows are for
        encoding = self._tokenizer(list(tokens), is_split_into_words=True, add_special_tokens=False, verbose=False)
        return encoding["input_ids"], encoding.word_ids()

    def _windows(self, nb_pieces):
        """Start of the windows covering a sequence of sub-word units."""
        capacity = self.window - self._special_tokens
        if nb_pieces <= capacity:
            return [0]
        starts = list(range(0, nb_pieces - capacity, self.stride))
        # the last window is aligned on the end of the sequence
        starts.append(nb_pieces - capacity)
        return starts

    def _pool_layers(self, hidden_states):
        torch = _import_torch()
        selected = [hidden_states[layer] for layer in self.layers]
        if self.layer_pooling == "concat":
            return torch.cat(selected, dim=-1)
        stacked = torch.stack(selected, dim=0)
        if self.layer_pooling == "sum":
            return stacked.sum(dim=0)
        return stacked.mean(dim=0)

    def _pool_subwords(self, piece_vectors, word_ids, nb_words):
        """One vector per word, a word without any sub-word unit gets a zero vector."""
        vectors = np.zeros((nb_words, self.embed_size), dtype=np.float32)
        if len(word_ids) == 0:
            return vectors
        word_ids = np.asarray(word_ids, dtype=np.int64)
        if self.subword_pooling == "mean":
            np.add.at(vectors, word_ids, piece_vectors)
            counts = np.bincount(word_ids, minlength=nb_words).astype(np.float32)
            vectors /= np.maximum(counts, 1.0)[:, None]
        elif self.subword_pooling == "last":
            # a later sub-word unit of the same word overwrites the earlier one
            vectors[word_ids] = piece_vectors
        else:
            # word ids are increasing: the first occurrence is the first unit
            _, first_positions = np.unique(word_ids, return_index=True)
            vectors[word_ids[first_positions]] = piece_vectors[first_positions]
        return vectors

    def embed_batch(self, token_lists):
        """
        Compute the vectors of several sentences, given as lists of tokens.
        Returns one float32 array of shape (number of tokens, embed_size) per
        sentence. The cache is neither read nor written.
        """
        torch = _import_torch()
        self._load_model()
        capacity = self.window - self._special_tokens

        tokenized = [self._tokenize(tokens) for tokens in token_lists]
        piece_vectors = []
        margins = []
        jobs = []
        for sentence_index, (piece_ids, _) in enumerate(tokenized):
            piece_vectors.append(np.zeros((len(piece_ids), self.embed_size), dtype=np.float32))
            margins.append(np.full((len(piece_ids),), -1, dtype=np.int64))
            if len(piece_ids) == 0:
                continue
            for start in self._windows(len(piece_ids)):
                jobs.append((sentence_index, start, piece_ids[start : start + capacity]))

        # windows of similar length together, to limit the padding
        jobs.sort(key=lambda job: len(job[2]), reverse=True)

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        for batch_start in range(0, len(jobs), self.batch_size):
            batch = jobs[batch_start : batch_start + self.batch_size]
            sequences = [self._prefix_ids + list(ids) + self._suffix_ids for _, _, ids in batch]
            max_length = max(len(sequence) for sequence in sequences)
            input_ids = np.full((len(batch), max_length), pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(batch), max_length), dtype=np.int64)
            for row, sequence in enumerate(sequences):
                input_ids[row, : len(sequence)] = sequence
                attention_mask[row, : len(sequence)] = 1

            with torch.inference_mode():
                outputs = self._model(
                    input_ids=torch.from_numpy(input_ids).to(self._device),
                    attention_mask=torch.from_numpy(attention_mask).to(self._device),
                    output_hidden_states=True,
                )
                pooled = self._pool_layers(outputs.hidden_states).float().cpu().numpy()

            for row, (sentence_index, start, ids) in enumerate(batch):
                length = len(ids)
                vectors = pooled[row, self._prefix_length : self._prefix_length + length]
                # distance of each sub-word unit to the closest edge of the
                # window: the window giving the most context wins
                positions = np.arange(length)
                margin = np.minimum(positions, length - 1 - positions)
                target_margin = margins[sentence_index][start : start + length]
                better = margin > target_margin
                piece_vectors[sentence_index][start : start + length][better] = vectors[better]
                target_margin[better] = margin[better]

        return [
            self._pool_subwords(piece_vectors[i], word_ids, len(token_lists[i]))
            for i, (_, word_ids) in enumerate(tokenized)
        ]

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------

    def fingerprint(self):
        """Everything the vectors depend on, two different fingerprints never share a cache."""
        return {
            "model": str(self.model_reference),
            "layers": list(self.layers),
            "layer_pooling": self.layer_pooling,
            "subword_pooling": self.subword_pooling,
            "window": self.window,
            "stride": self.stride,
            "dtype": np.dtype(CACHE_DTYPE).name,
        }

    def cache_env_path(self):
        """Directory of the LMDB cache, or None when the vectors are only kept in memory."""
        if not self.cache_path or self.cache_path == "None":
            return None
        fingerprint = json.dumps(self.fingerprint(), sort_keys=True)
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(os.path.normpath(str(self.model_reference))))
        return os.path.join(self.cache_path, "%s-%s" % (name, digest))

    @staticmethod
    def sentence_key(tokens):
        return hashlib.sha1("\x1f".join(tokens).encode("utf-8")).digest()

    # The cache is a directory of LMDB *shards*. A shard is written by a single
    # process in a temporary directory, and only becomes visible, through an
    # atomic rename, once it is complete and closed. It is never modified
    # afterwards. Several trainings can thus feed the same cache at the same
    # time, from different nodes and over a network file system (a SLURM job
    # array), where LMDB offers no protection against concurrent writers. At
    # worst two processes embed the same sentences at the same time, which
    # wastes some space.

    def _shard_names(self):
        path = self.cache_env_path()
        if path is None or not os.path.isdir(path):
            return []
        return sorted(
            name
            for name in os.listdir(path)
            if name.startswith(SHARD_PREFIX) and os.path.isdir(os.path.join(path, name))
        )

    def _close_shards(self):
        # handles inherited through fork() are closed too: they are unusable,
        # but they still prevent the shards from being opened in this process
        for env in self._shards.values():
            try:
                env.close()
            except Exception:
                pass
        self._shards = {}
        self._shards_pid = None

    def _open_shards(self):
        """Open the shards that are not open yet, returns whether there were some."""
        import lmdb

        if self._shards_pid != os.getpid():
            self._close_shards()
            self._shards_pid = os.getpid()
        path = self.cache_env_path()
        opened = False
        for name in self._shard_names():
            if name in self._shards:
                continue
            try:
                self._shards[name] = lmdb.open(
                    os.path.join(path, name), readonly=True, lock=False, max_readers=2048, max_spare_txns=2
                )
                opened = True
            except lmdb.Error as e:
                print("warning: cannot open the contextual embeddings cache shard", name, ":", e)
        return opened

    def reopen_lmdb(self):
        """Called by the DataLoader workers, see Embeddings.reopen_lmdb."""
        with self._lock:
            self._close_shards()
            self._open_shards()

    def _deserialize(self, value):
        return np.frombuffer(value, dtype=CACHE_DTYPE).reshape(-1, self.embed_size).astype(np.float32)

    def _serialize(self, vectors):
        return np.clip(vectors, -CACHE_DTYPE_MAX, CACHE_DTYPE_MAX).astype(CACHE_DTYPE).tobytes()

    def _read_shards(self, key):
        for env in self._shards.values():
            with env.begin() as txn:
                value = txn.get(key)
            if value is not None:
                return value
        return None

    def _lookup(self, key):
        vectors = self._memory.get(key)
        if vectors is not None:
            return vectors
        if self.cache_env_path() is None:
            return None
        with self._lock:
            if self._shards_pid != os.getpid():
                self._open_shards()
            value = self._read_shards(key)
            if value is None and self._open_shards():
                # a shard has been published in the meantime
                value = self._read_shards(key)
        if value is None:
            return None
        return self._deserialize(value)

    def _not_cached(self, pending):
        """The sentences of ``pending`` (key -> tokens) that are in no shard."""
        with self._lock:
            self._open_shards()
            for env in self._shards.values():
                if len(pending) == 0:
                    break
                with env.begin() as txn:
                    cursor = txn.cursor()
                    pending = {key: tokens for key, tokens in pending.items() if not cursor.set_key(key)}
        return pending

    def _write_fingerprint(self, env_path):
        fingerprint_path = os.path.join(env_path, FINGERPRINT_FILE)
        if os.path.isfile(fingerprint_path):
            return
        temporary_path = "%s.%s" % (fingerprint_path, uuid.uuid4().hex[:8])
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(self.fingerprint(), f, indent=4, sort_keys=True)
        os.replace(temporary_path, fingerprint_path)

    def precompute(self, token_lists, max_sequence_length=None, persist=True, verbose=True, distributed=False):
        """
        Compute and cache the vectors of the sentences that are not cached yet,
        in the process that owns the transformer (and the GPU), so that the
        DataLoader workers only have to read them. Returns the number of
        sentences embedded by this process.

        Sentences are truncated to ``max_sequence_length`` the way the datasets
        do it. With ``persist=False`` (prediction on arbitrary texts) the
        vectors only live in memory, until the next such call.

        ``distributed`` must only be set when every process of a distributed
        training makes the same call: the sentences are then shared out between
        the processes, which wait for each other.
        """
        env_path = self.cache_env_path() if persist else None
        if not persist:
            self._memory = {}

        pending = {}
        for tokens in token_lists:
            tokens = list(tokens)
            if max_sequence_length:
                tokens = tokens[:max_sequence_length]
            key = self.sentence_key(tokens)
            if key not in pending and key not in self._memory:
                pending[key] = tokens

        if env_path is not None:
            pending = self._not_cached(pending)

        world_size, rank = 1, 0
        if distributed and env_path is not None:
            torch = _import_torch()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
        if world_size > 1:
            # a sentence belongs to a process whatever the other sentences
            # are, the processes may not see the same state of the cache
            pending = {
                key: tokens for key, tokens in pending.items() if int.from_bytes(key[:4], "big") % world_size == rank
            }

        if len(pending) > 0:
            if verbose:
                print("computing contextual embeddings for", len(pending), "sequences with", self.model_reference)
            # longest first, so that a lack of memory shows up immediately
            items = sorted(pending.items(), key=lambda item: len(item[1]), reverse=True)
            if env_path is not None:
                self._write_shard(env_path, items, verbose)
            else:
                for chunk, vectors in self._embed_chunks(items, verbose):
                    for (key, _), sentence_vectors in zip(chunk, vectors):
                        # through float16 too, so that a vector is the same
                        # whether it comes from memory or from the cache
                        self._memory[key] = self._deserialize(self._serialize(sentence_vectors))

        if world_size > 1:
            _import_torch().distributed.barrier()
        if env_path is not None:
            with self._lock:
                self._open_shards()

        return len(pending)

    def _embed_chunks(self, items, verbose):
        progress = None
        if verbose:
            try:
                from tqdm import tqdm

                progress = tqdm(total=len(items))
            except ImportError:
                progress = None
        try:
            for start in range(0, len(items), PRECOMPUTE_CHUNK_SIZE):
                chunk = items[start : start + PRECOMPUTE_CHUNK_SIZE]
                yield chunk, self.embed_batch([tokens for _, tokens in chunk])
                if progress is not None:
                    progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

    def _write_shard(self, env_path, items, verbose):
        import lmdb

        os.makedirs(env_path, exist_ok=True)
        self._write_fingerprint(env_path)

        identifier = uuid.uuid4().hex[:8]
        temporary_path = os.path.join(
            env_path, "%s%s-%d-%s" % (TEMPORARY_PREFIX, socket.gethostname(), os.getpid(), identifier)
        )
        shard_path = os.path.join(
            env_path, "%s%s-%s" % (SHARD_PREFIX, time.strftime("%Y%m%dT%H%M%S", time.gmtime()), identifier)
        )

        # the size of the map is a maximum that cannot be exceeded: values
        # are stored in whole pages, plus the pages of the tree
        nb_tokens = sum(len(tokens) for _, tokens in items)
        map_size = int(nb_tokens * self.embed_size * np.dtype(CACHE_DTYPE).itemsize * 1.2)
        map_size += len(items) * 16384 + 64 * 1024 * 1024

        written = 0
        # no lock file: this process is the only one to ever write here, and
        # locks are not reliable on a network file system
        env = lmdb.open(temporary_path, map_size=map_size, lock=False)
        try:
            for chunk, vectors in self._embed_chunks(items, verbose):
                with env.begin(write=True) as txn:
                    for (key, _), sentence_vectors in zip(chunk, vectors):
                        txn.put(key, self._serialize(sentence_vectors))
                written += len(chunk)
        finally:
            env.close()
            if written > 0:
                # what has been computed is kept, even after a failure
                os.rename(temporary_path, shard_path)
            else:
                shutil.rmtree(temporary_path, ignore_errors=True)

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------

    def get_sentence_vectors(self, tokens):
        """
        Vectors of a sentence given as a list of tokens, as a float32 array of
        shape (number of tokens, embed_size). Taken from the cache when the
        sentence has been precomputed, computed on the fly otherwise.
        """
        tokens = list(tokens)
        if len(tokens) == 0:
            return np.zeros((0, self.embed_size), dtype=np.float32)
        vectors = self._lookup(self.sentence_key(tokens))
        if vectors is not None and vectors.shape[0] == len(tokens):
            return vectors
        vectors = self.embed_batch([tokens])[0]
        return self._deserialize(self._serialize(vectors))

    def get_word_vector(self, word):
        """
        Vector of a word out of any context, for the code that still works
        word by word. This is a poor use of a contextual model.
        """
        return self.get_sentence_vectors([word])[0]
