"""
Regression tests for sharing an LMDB embeddings database within one process.

Since py-lmdb 2.0, opening an environment that is already open in the process
raises lmdb.Error. GROBID runs sequence labelling and text classification on
two threads of a single JEP SharedInterpreter, both reading the same embeddings:
every DataGenerator calls reopen_lmdb(), so the handle must be shared, never
closed and reopened under another reader.
"""

import os
import pickle
import threading

import lmdb
import numpy as np

from delft.utilities.Embeddings import Embeddings, close_lmdb_env

EMBED_SIZE = 4
WORDS = ["the", "of", "university", "department", "laboratory"]


def _make_fake_lmdb(parent_dir, name):
    envpath = os.path.join(parent_dir, name)
    os.makedirs(envpath, exist_ok=True)
    env = lmdb.open(envpath, map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        for i, word in enumerate(WORDS):
            txn.put(word.encode("utf-8"), np.full(EMBED_SIZE, i, dtype=np.float32).tobytes())
    env.close()
    return envpath


def _make_embeddings(parent_dir, name):
    emb = Embeddings.__new__(Embeddings)
    emb.name = name
    emb.embed_size = EMBED_SIZE
    emb.static_embed_size = EMBED_SIZE
    emb.vocab_size = len(WORDS)
    emb.model = {}
    emb.registry = None
    emb.lang = "en"
    emb.extension = "vec"
    emb.embedding_lmdb_path = parent_dir
    emb.env = None
    emb.use_cache = False
    return emb


def test_reopen_reuses_the_environment_open_in_the_process(tmp_path):
    envpath = _make_fake_lmdb(str(tmp_path), "fake")
    try:
        first = _make_embeddings(str(tmp_path), "fake")
        second = _make_embeddings(str(tmp_path), "fake")
        first.reopen_lmdb()
        second.reopen_lmdb()
        first.reopen_lmdb()

        assert first.env is second.env
        np.testing.assert_array_equal(second.get_word_vector("of"), np.full(EMBED_SIZE, 1, dtype=np.float32))
    finally:
        close_lmdb_env(envpath)


def test_unpickled_embeddings_share_the_environment(tmp_path):
    envpath = _make_fake_lmdb(str(tmp_path), "fake")
    try:
        emb = _make_embeddings(str(tmp_path), "fake")
        emb.reopen_lmdb()
        restored = pickle.loads(pickle.dumps(emb))

        assert restored.env is emb.env
        np.testing.assert_array_equal(restored.get_word_vector("university"), np.full(EMBED_SIZE, 2, dtype=np.float32))
    finally:
        close_lmdb_env(envpath)


def test_concurrent_reopen_and_lookup_do_not_collide(tmp_path):
    envpath = _make_fake_lmdb(str(tmp_path), "fake")
    shared = _make_embeddings(str(tmp_path), "fake")
    shared.reopen_lmdb()
    errors = []
    start = threading.Barrier(4)

    def worker(seed):
        try:
            start.wait()
            for i in range(300):
                # what each DataGenerator does before reading the vectors
                shared.reopen_lmdb()
                word = WORDS[(seed + i) % len(WORDS)]
                vector = shared.get_word_vector(word)
                assert vector.shape == (EMBED_SIZE,)
                assert vector[0] == WORDS.index(word)
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    # daemon threads with a bounded join: an environment closed under a reader
    # can also wedge a thread, which must fail the test, not hang it
    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    stuck = [t for t in threads if t.is_alive()]
    if not stuck:
        close_lmdb_env(envpath)

    assert stuck == []
    assert errors == []
