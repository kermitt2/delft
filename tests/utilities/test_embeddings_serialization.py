"""
Regression tests for Embeddings serialization across DataLoader workers.

The PyTorch DataLoader serializes the dataset (and any objects it holds) to
spawn-mode worker processes. The LMDB Environment is not serializable, so
Embeddings must define __getstate__/__setstate__ to drop and re-open it.
"""

import multiprocessing as mp
import os
import struct

import lmdb

from delft.utilities.Embeddings import (
    Embeddings,
    _fetch_header_if_available,
    close_lmdb_env,
    current_lmdb_env,
    open_lmdb_env,
)


def _make_fake_lmdb(parent_dir: str, name: str, vector_dict: dict) -> None:
    envpath = os.path.join(parent_dir, name)
    os.makedirs(envpath, exist_ok=True)
    env = lmdb.open(envpath, map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        for word, vec in vector_dict.items():
            txn.put(word.encode("utf-8"), struct.pack(f"{len(vec)}f", *vec))
    env.close()


def _make_embeddings(tmpdir: str, name: str, embed_size: int) -> Embeddings:
    emb = Embeddings.__new__(Embeddings)
    emb.name = name
    emb.embed_size = embed_size
    emb.static_embed_size = embed_size
    emb.vocab_size = 1
    emb.model = {}
    emb.registry = None
    emb.lang = "en"
    emb.extension = "vec"
    emb.embedding_lmdb_path = tmpdir
    emb.env = None
    emb.use_cache = False
    return emb


def _spawn_worker(emb, q):
    try:
        v = emb.get_word_vector("hello")
        q.put(("ok", tuple(v.shape), float(v.sum())))
    except Exception as e:
        q.put(("err", type(e).__name__, str(e)))


def test_embeddings_spawn_multiprocessing(tmp_path):
    """
    The exact scenario a DataLoader(num_workers>0) hits on macOS / Python 3.8+:
    the spawn worker receives the Embeddings (which holds an LMDB Environment)
    and must be able to call get_word_vector. Regression: dev branch lost
    __getstate__/__setstate__ in the PyTorch migration cleanup.
    """
    _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
    emb = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_spawn_worker, args=(emb, q))
    p.start()
    p.join(timeout=30)

    assert p.exitcode == 0, f"spawn worker exited with {p.exitcode}"
    assert not q.empty(), "worker produced no result"
    result = q.get(timeout=2)
    assert result[0] == "ok", f"worker failed: {result[1:]}"
    shape, total = result[1], result[2]
    assert shape == (5,)
    assert abs(total - 1.5) < 1e-5


def test_reopen_lmdb_is_idempotent_when_no_lmdb():
    emb = Embeddings.__new__(Embeddings)
    emb.name = "no-lmdb"
    emb.embed_size = 0
    emb.static_embed_size = 0
    emb.vocab_size = 0
    emb.model = {}
    emb.registry = None
    emb.lang = "en"
    emb.extension = "vec"
    emb.embedding_lmdb_path = None
    emb.env = None
    emb.use_cache = False

    emb.reopen_lmdb()
    assert emb.env is None


def test_reopen_lmdb_opens_on_demand(tmp_path):
    _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
    emb = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)
    assert emb.env is None
    emb.reopen_lmdb()
    assert emb.env is not None
    vector = emb.get_word_vector("hello")
    assert vector.shape == (5,)


class TestSharedLmdbEnvironment:
    """
    py-lmdb >= 1.0 refuses to open the same environment twice in one process:
        lmdb.Error: The environment '<path>' is already open in this process.
    GROBID loads one Sequence model per document structure into a single JEP
    SharedInterpreter, so several Embeddings end up over the same database.
    """

    def test_several_embeddings_share_one_environment(self, tmp_path):
        _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
        envpath = os.path.join(str(tmp_path), "fake-embed")
        try:
            header = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)
            citation = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)

            header.reopen_lmdb()
            citation.reopen_lmdb()  # used to raise lmdb.Error

            assert header.env is citation.env
            assert header.get_word_vector("hello").shape == (5,)
            assert citation.get_word_vector("hello").shape == (5,)
        finally:
            close_lmdb_env(envpath)

    def test_open_lmdb_env_reports_reuse(self, tmp_path):
        _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
        envpath = os.path.join(str(tmp_path), "fake-embed")
        try:
            first, opened_first = open_lmdb_env(envpath, readonly=True)
            second, opened_second = open_lmdb_env(envpath, readonly=True)

            assert opened_first is True
            assert opened_second is False
            assert first is second

            # another spelling of the same path resolves to the same entry
            third, opened_third = open_lmdb_env(os.path.join(str(tmp_path), ".", "fake-embed"), readonly=True)
            assert opened_third is False
            assert third is first
        finally:
            close_lmdb_env(envpath)
        assert current_lmdb_env(envpath) is None

    def test_recovery_adopts_replacement_instead_of_reopening(self, tmp_path):
        _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
        envpath = os.path.join(str(tmp_path), "fake-embed")
        try:
            header = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)
            citation = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)
            header.reopen_lmdb()
            citation.reopen_lmdb()
            superseded = header.env

            # header hits MDB_BAD_RSLOT and reopens the shared environment
            recovered = header.recover_lmdb_env()
            assert recovered is not superseded

            # citation still holds the superseded handle: it must adopt the new
            # one rather than close it and open yet another
            adopted = citation.recover_lmdb_env()
            assert adopted is recovered

            citation.env = adopted
            assert citation.get_word_vector("hello").shape == (5,)
        finally:
            close_lmdb_env(envpath)

    def test_get_word_vector_recovers_from_closed_environment(self, tmp_path):
        _make_fake_lmdb(str(tmp_path), "fake-embed", {"hello": [0.1, 0.2, 0.3, 0.4, 0.5]})
        envpath = os.path.join(str(tmp_path), "fake-embed")
        try:
            emb = _make_embeddings(str(tmp_path), "fake-embed", embed_size=5)
            emb.reopen_lmdb()
            emb.env.close()  # a handle closed underneath us

            vector = emb.get_word_vector("hello")
            assert vector.shape == (5,)
            assert abs(float(vector.sum()) - 1.5) < 1e-5
        finally:
            close_lmdb_env(envpath)


class TestFetchHeaderIfAvailable:
    def test_raw_string_header_is_split_and_parsed(self):
        # Used to raise AttributeError because of `line.isinstance("str")`.
        assert _fetch_header_if_available("100 300") == (100, 300)

    def test_pre_split_header_list_is_parsed(self):
        assert _fetch_header_if_available(["100", "300"]) == (100, 300)

    def test_non_header_line_returns_sentinels(self):
        assert _fetch_header_if_available(["the", "0.1", "0.2", "0.3"]) == (-1, -1)

    def test_header_with_trailing_newline(self):
        assert _fetch_header_if_available(["100", "300\n"]) == (100, 300)
