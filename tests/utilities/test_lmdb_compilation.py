"""
The compilation of embeddings into their LMDB database, when several processes need
the same database at once, as the tasks of a SLURM array do, or find a damaged one.
"""

import multiprocessing
import os
import time

import numpy as np
import pytest

from delft.utilities import Embeddings as embeddings_module
from delft.utilities.Embeddings import BUILD_DIRECTORY_SUFFIX, BUILD_LOCK_SUFFIX, Embeddings, close_lmdb_env

NAME = "tiny-vectors"
NB_WORDS, DIMENSIONS = 150, 12  # above what a database found on disk must hold to be trusted


def _registry(tmp_path):
    path = tmp_path / "vectors.vec"
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            for i in range(NB_WORDS):
                f.write(f"word{i} " + " ".join(str(float(i + d)) for d in range(DIMENSIONS)) + "\n")
    return {
        "embedding-lmdb-path": str(tmp_path / "db"),
        "embedding-download-path": str(tmp_path / "download"),
        "embeddings": [{"name": NAME, "path": str(path), "type": "glove", "format": "vec", "lang": "en"}],
        "embeddings-contextualized": [],
        "transformers": [],
    }


@pytest.fixture
def small_and_quick(monkeypatch):
    monkeypatch.setattr(embeddings_module, "map_size", 10 * 1024 * 1024)
    monkeypatch.setattr(embeddings_module, "BUILD_WAIT_SECONDS", 0.05)


def _database(tmp_path):
    return str(tmp_path / "db" / NAME)


def _assert_usable(tmp_path, embeddings):
    assert embeddings.vocab_size == NB_WORDS and embeddings.embed_size == DIMENSIONS
    np.testing.assert_array_equal(embeddings.get_word_vector("word7"), [7.0 + d for d in range(DIMENSIONS)])
    assert not os.path.exists(_database(tmp_path) + BUILD_DIRECTORY_SUFFIX)
    assert not os.path.exists(_database(tmp_path) + BUILD_LOCK_SUFFIX)


def _load_and_report(registry, log, queue):
    """One process: compile or wait, then read; the pids that compiled go to ``log``."""
    original = Embeddings.load_embeddings_from_file

    def logged(self, path):
        with open(log, "a") as f:
            f.write(f"{os.getpid()}\n")
        time.sleep(0.5)  # keep the others waiting on the lock
        return original(self, path)

    Embeddings.load_embeddings_from_file = logged
    try:
        embeddings = Embeddings(NAME, resource_registry=registry)
        queue.put((embeddings.vocab_size, embeddings.embed_size, float(embeddings.get_word_vector("word7")[0])))
    except Exception as e:
        queue.put(("error", type(e).__name__, str(e)))


def test_several_processes_needing_the_same_database_compile_it_once(tmp_path, small_and_quick):
    """Each compiled it, all at once into the same directory: MDB_PAGE_NOTFOUND."""
    registry = _registry(tmp_path)
    log = str(tmp_path / "compiled-by.log")
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=_load_and_report, args=(registry, log, queue)) for _ in range(4)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert results == [(NB_WORDS, DIMENSIONS, 7.0)] * 4
    with open(log) as f:
        compiled_by = f.read().split()
    assert len(compiled_by) == 1, f"compiled by {compiled_by}: a process took a complete database for a damaged one"
    embeddings = Embeddings(NAME, resource_registry=registry)
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))


def test_a_damaged_database_is_compiled_again(tmp_path, small_and_quick):
    os.makedirs(_database(tmp_path))
    with open(os.path.join(_database(tmp_path), "data.mdb"), "wb") as f:
        f.write(os.urandom(4096))
    embeddings = Embeddings(NAME, resource_registry=_registry(tmp_path))
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))


def test_a_half_written_database_is_compiled_again(tmp_path, small_and_quick):
    import lmdb

    os.makedirs(_database(tmp_path))
    env = lmdb.open(_database(tmp_path), map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(b"word0", np.zeros(DIMENSIONS, dtype="float32").tobytes())
    env.close()
    embeddings = Embeddings(NAME, resource_registry=_registry(tmp_path))
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))


def test_leftovers_of_a_compilation_that_died_do_not_block(tmp_path, small_and_quick, monkeypatch):
    monkeypatch.setattr(embeddings_module, "STALE_BUILD_LOCK_SECONDS", 1)
    os.makedirs(_database(tmp_path) + BUILD_DIRECTORY_SUFFIX)
    os.makedirs(_database(tmp_path) + BUILD_LOCK_SUFFIX)
    old = time.time() - 10
    os.utime(_database(tmp_path) + BUILD_LOCK_SUFFIX, (old, old))
    embeddings = Embeddings(NAME, resource_registry=_registry(tmp_path))
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))


def test_a_process_waits_for_the_one_compiling(tmp_path, small_and_quick):
    """With the lock held, the process waits, and reads the database once it is there."""
    registry = _registry(tmp_path)
    lock = _database(tmp_path) + BUILD_LOCK_SUFFIX
    os.makedirs(lock)

    def compile_then_release():
        time.sleep(0.3)
        other = Embeddings.__new__(Embeddings)
        other.__init__(NAME, resource_registry=registry, load=False)
        other._compile_lmdb(NAME, _database(tmp_path))
        os.rmdir(lock)

    import threading

    thread = threading.Thread(target=compile_then_release)
    thread.start()
    embeddings = Embeddings(NAME, resource_registry=registry)
    thread.join()
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))
