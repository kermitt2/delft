"""
The compilation of embeddings into their LMDB database, when several processes need
the same database at once, as the tasks of a SLURM array do, or find a damaged one.
"""

import glob
import multiprocessing
import os
import threading
import time

import numpy as np
import pytest

from delft.utilities import Embeddings as embeddings_module
from delft.utilities.Embeddings import (
    BUILD_DIRECTORY_SUFFIX,
    BUILD_LOCK_SUFFIX,
    Embeddings,
    _acquire_build_lock,
    _build_lock_owner,
    _release_build_lock,
    close_lmdb_env,
)

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
    # nothing left of the compilations, each in a directory of its own
    assert glob.glob(_database(tmp_path) + BUILD_DIRECTORY_SUFFIX + "*") == []
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
    old = time.time() - 10
    for leftover in (BUILD_DIRECTORY_SUFFIX, BUILD_DIRECTORY_SUFFIX + "-node1-123-abc", BUILD_LOCK_SUFFIX):
        os.makedirs(_database(tmp_path) + leftover)
        os.utime(_database(tmp_path) + leftover, (old, old))
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


def _old(path, seconds=10):
    past = time.time() - seconds
    os.utime(path, (past, past))


class TestBuildLock:
    def test_taken_once(self, tmp_path):
        lock = str(tmp_path / "vectors.lock")
        token = _acquire_build_lock(lock)
        assert token is not None and _build_lock_owner(lock) == token
        assert _acquire_build_lock(lock) is None

    def test_released_by_its_owner_alone(self, tmp_path):
        lock = str(tmp_path / "vectors.lock")
        token = _acquire_build_lock(lock)
        _release_build_lock(lock, "another-process")
        assert _build_lock_owner(lock) == token
        _release_build_lock(lock, token)
        assert not os.path.exists(lock)

    def test_a_stale_lock_is_taken_over_by_one_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embeddings_module, "STALE_BUILD_LOCK_SECONDS", 1)
        lock = str(tmp_path / "vectors.lock")
        os.makedirs(lock)
        _old(lock)

        first = _acquire_build_lock(lock)
        assert first is not None and _build_lock_owner(lock) == first
        assert _acquire_build_lock(lock) is None
        assert glob.glob(lock + ".stale-*") == []

    def test_a_process_that_saw_the_stale_lock_does_not_take_the_one_that_replaced_it(self, tmp_path, monkeypatch):
        """
        Two processes found the lock stale: the first removed it and took a new one, which
        the second, acting on what it had seen, removed in turn and took as well.
        """
        monkeypatch.setattr(embeddings_module, "STALE_BUILD_LOCK_SECONDS", 1)
        lock = str(tmp_path / "vectors.lock")
        os.makedirs(lock)
        _old(lock)
        first = _acquire_build_lock(lock)  # takes over the stale lock

        # the second process looked at the lock while it was still the stale one
        really_stale = embeddings_module._build_lock_is_stale
        answers = iter([True])
        monkeypatch.setattr(
            embeddings_module, "_build_lock_is_stale", lambda path: next(answers, None) or really_stale(path)
        )

        assert _acquire_build_lock(lock) is None
        assert _build_lock_owner(lock) == first
        assert glob.glob(lock + ".stale-*") == []


def test_two_compilations_at_once_do_not_write_to_the_same_directory(tmp_path, small_and_quick):
    """Should two processes hold the lock, as after a takeover gone wrong, each compiles apart."""
    registry = _registry(tmp_path)
    os.makedirs(registry["embedding-lmdb-path"])
    directories = []
    original = Embeddings.load_embeddings_from_file

    def slow(self, path):
        directories.append(self.env.path())
        time.sleep(0.3)  # both are writing at the same time
        return original(self, path)

    errors = []

    def compile_():
        try:
            embeddings = Embeddings(NAME, resource_registry=registry, load=False)
            embeddings._compile_lmdb(NAME, _database(tmp_path))
        except Exception as e:  # reported by the assertion below
            errors.append(e)

    Embeddings.load_embeddings_from_file = slow
    try:
        threads = [threading.Thread(target=compile_) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        Embeddings.load_embeddings_from_file = original

    assert errors == []
    assert len(set(directories)) == 2
    embeddings = Embeddings(NAME, resource_registry=registry)
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))


def test_a_database_put_there_by_another_process_meanwhile_is_kept(tmp_path, small_and_quick):
    registry = _registry(tmp_path)
    os.makedirs(registry["embedding-lmdb-path"])
    original = Embeddings.make_embeddings_lmdb
    published = {}

    def compile_while_another_publishes(self, name):
        original(self, name)
        if not published:
            # the other process finishes first
            published["by-other"] = True
            other = Embeddings(NAME, resource_registry=registry, load=False)
            other._compile_lmdb(NAME, _database(tmp_path))
            published["inode"] = os.stat(_database(tmp_path)).st_ino

    Embeddings.make_embeddings_lmdb = compile_while_another_publishes
    try:
        Embeddings(NAME, resource_registry=registry, load=False)._compile_lmdb(NAME, _database(tmp_path))
    finally:
        Embeddings.make_embeddings_lmdb = original

    assert os.stat(_database(tmp_path)).st_ino == published["inode"]
    embeddings = Embeddings(NAME, resource_registry=registry)
    try:
        _assert_usable(tmp_path, embeddings)
    finally:
        close_lmdb_env(_database(tmp_path))
