"""
Tests for the modern static embeddings backend (sentence-transformers static
embeddings and Model2Vec potion models, see issue #178).

The tests build tiny local models with the two file layouts supported, so that
nothing is downloaded from the HuggingFace hub.
"""

import json
import os
import pickle

import numpy as np
import pytest

from delft.utilities.Embeddings import Embeddings
from delft.utilities.StaticEmbeddings import (
    StaticTransformerEmbeddings,
    is_static_embedding_directory,
    looks_like_static_embedding_reference,
)

pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

VOCAB = {"[UNK]": 0, "the": 1, "cat": 2, "##s": 3}

MATRIX = np.array(
    [
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ],
    dtype=np.float32,
)


def _write_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordPiece(vocab=dict(VOCAB), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)


def _write_matrix(path, tensor_name, matrix=MATRIX):
    from safetensors.numpy import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({tensor_name: matrix}, path)


def make_model2vec_model(directory, normalize=False, matrix=MATRIX):
    """A Model2Vec/potion layout: everything at the root of the directory."""
    directory = str(directory)
    _write_tokenizer(os.path.join(directory, "tokenizer.json"))
    _write_matrix(os.path.join(directory, "model.safetensors"), "embeddings", matrix)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "model2vec", "normalize": normalize}, f)
    return directory


def make_sentence_transformers_model(directory, normalize=False, matrix=MATRIX):
    """A sentence-transformers static embedding layout: 0_StaticEmbedding/."""
    directory = str(directory)
    _write_tokenizer(os.path.join(directory, "0_StaticEmbedding", "tokenizer.json"))
    _write_matrix(
        os.path.join(directory, "0_StaticEmbedding", "model.safetensors"),
        "embedding.weight",
        matrix,
    )
    modules = [
        {"idx": 0, "name": "0", "path": "0_StaticEmbedding", "type": "sentence_transformers.models.StaticEmbedding"}
    ]
    if normalize:
        modules.append({"idx": 1, "name": "1", "path": "1_Normalize", "type": "sentence_transformers.models.Normalize"})
    with open(os.path.join(directory, "modules.json"), "w", encoding="utf-8") as f:
        json.dump(modules, f)
    return directory


def _registry(tmp_path, entry):
    return {
        "embedding-lmdb-path": str(tmp_path / "db"),
        "embedding-download-path": str(tmp_path / "download"),
        "embeddings": [entry],
        "transformers": [],
        "embeddings-contextualized": [],
    }


class TestStaticTransformerEmbeddings:
    def test_word_vector_is_the_mean_of_its_sub_word_units(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))

        assert model.embed_size == 4
        assert model.vocab_size == 4
        # "cat" is a single unit, its vector is the matrix row
        np.testing.assert_allclose(model.get_word_vector("cat"), MATRIX[2])
        # "cats" is split into "cat" and "##s", the vectors are averaged
        np.testing.assert_allclose(model.get_word_vector("cats"), (MATRIX[2] + MATRIX[3]) / 2)

    def test_unknown_word_falls_back_on_the_unknown_unit(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))

        # unlike glove or word2vec, there is no zero vector for unseen words
        np.testing.assert_allclose(model.get_word_vector("zzzz"), MATRIX[0])

    def test_untokenizable_word_gets_a_zero_vector(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))

        np.testing.assert_allclose(model.get_word_vector(""), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(model.get_word_vector(" "), np.zeros(4, dtype=np.float32))

    def test_vectors_are_float32(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))

        assert model.get_word_vector("cats").dtype == np.float32
        assert model.get_word_vector("").dtype == np.float32

    def test_normalization_is_read_from_the_model_configuration(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion", normalize=True))

        assert model.normalize is True
        vector = model.get_word_vector("cats")
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        # normalization applies to the pooled vector, not to each unit
        expected = (MATRIX[2] + MATRIX[3]) / 2
        np.testing.assert_allclose(vector, expected / np.linalg.norm(expected), rtol=1e-6)

    def test_normalization_can_be_forced_from_the_registry(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion", normalize=True)

        assert StaticTransformerEmbeddings(directory, normalize=False).normalize is False

    def test_sentence_transformers_layout_is_supported(self, tmp_path):
        directory = make_sentence_transformers_model(tmp_path / "static-mrl", normalize=True)
        model = StaticTransformerEmbeddings(directory)

        assert model.embed_size == 4
        assert model.normalize is True
        np.testing.assert_allclose(model.get_word_vector("cat"), MATRIX[2])

    def test_matryoshka_truncation(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")
        model = StaticTransformerEmbeddings(directory, dimensions=2)

        assert model.embed_size == 2
        np.testing.assert_allclose(model.get_word_vector("cats"), ((MATRIX[2] + MATRIX[3]) / 2)[:2])

    def test_matryoshka_truncation_beyond_the_model_size_is_rejected(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")

        with pytest.raises(ValueError):
            StaticTransformerEmbeddings(directory, dimensions=8)

    def test_embedding_matrix_with_an_unknown_name_is_still_found(self, tmp_path):
        """A model of neither family, as long as it holds a single matrix."""
        directory = str(tmp_path / "custom")
        _write_tokenizer(os.path.join(directory, "tokenizer.json"))
        _write_matrix(os.path.join(directory, "model.safetensors"), "static_word_embeddings")

        model = StaticTransformerEmbeddings(directory)

        assert model.embed_size == 4
        np.testing.assert_allclose(model.get_word_vector("cat"), MATRIX[2])

    def test_ambiguous_embedding_matrix_is_reported(self, tmp_path):
        from safetensors.numpy import save_file

        directory = str(tmp_path / "custom")
        _write_tokenizer(os.path.join(directory, "tokenizer.json"))
        save_file(
            {"first": MATRIX, "second": MATRIX},
            os.path.join(directory, "model.safetensors"),
        )

        with pytest.raises(ValueError):
            StaticTransformerEmbeddings(directory)

    def test_a_directory_without_a_model_is_rejected(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError):
            StaticTransformerEmbeddings(str(empty))

    def test_serialization_drops_the_matrix_and_reloads_it(self, tmp_path):
        """
        A DataLoader worker receives the embeddings by pickle: the matrix must
        not travel through the pickle, it is reloaded from the model files.
        """
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))
        model.get_word_vector("cats")

        state = model.__getstate__()
        assert state["_matrix"] is None
        assert state["_tokenizer"] is None
        assert state["_cache"] == {}

        restored = pickle.loads(pickle.dumps(model))
        assert restored.embed_size == 4
        np.testing.assert_allclose(restored.get_word_vector("cats"), model.get_word_vector("cats"))

    def test_word_vectors_are_memoized(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"))

        assert model.get_word_vector("cats") is model.get_word_vector("cats")

    def test_cache_size_is_bounded(self, tmp_path):
        model = StaticTransformerEmbeddings(make_model2vec_model(tmp_path / "potion"), cache_size=1)

        model.get_word_vector("cat")
        model.get_word_vector("cats")

        assert len(model._cache) == 1
        np.testing.assert_allclose(model.get_word_vector("cats"), (MATRIX[2] + MATRIX[3]) / 2)


class TestReferenceDetection:
    def test_hub_identifiers_are_recognised(self):
        assert looks_like_static_embedding_reference("minishlab/potion-base-8M")
        assert looks_like_static_embedding_reference("sentence-transformers/static-retrieval-mrl-en-v1")

    def test_registry_names_are_not_hub_identifiers(self):
        assert not looks_like_static_embedding_reference("glove-840B")
        assert not looks_like_static_embedding_reference("word2vec")
        assert not looks_like_static_embedding_reference("")
        assert not looks_like_static_embedding_reference(None)

    def test_local_directories_are_recognised(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")

        assert is_static_embedding_directory(directory)
        assert looks_like_static_embedding_reference(directory)

    def test_unrelated_local_directory_is_rejected(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        assert not is_static_embedding_directory(str(empty))
        assert not looks_like_static_embedding_reference(str(empty))


class TestEmbeddingsIntegration:
    def test_loaded_from_the_embeddings_registry(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")
        registry = _registry(
            tmp_path,
            {
                "name": "test-static",
                "model": directory,
                "type": "model2vec",
                "format": "static-transformer",
                "lang": "en",
                "item": "word",
            },
        )

        embeddings = Embeddings("test-static", resource_registry=registry)

        assert embeddings.embed_size == 4
        assert embeddings.static_embed_size == 4
        assert embeddings.vocab_size == 4
        np.testing.assert_allclose(embeddings.get_word_vector("cats"), (MATRIX[2] + MATRIX[3]) / 2)
        # no LMDB database is compiled for a static embedding model
        assert not os.path.exists(os.path.join(str(tmp_path / "db"), "test-static"))

    def test_dimensions_from_the_registry(self, tmp_path):
        directory = make_sentence_transformers_model(tmp_path / "static-mrl")
        registry = _registry(
            tmp_path,
            {
                "name": "test-static",
                "model": directory,
                "type": "static-embedding",
                "format": "static-transformer",
                "dimensions": 3,
                "lang": "en",
                "item": "word",
            },
        )

        embeddings = Embeddings("test-static", resource_registry=registry)

        assert embeddings.embed_size == 3
        assert embeddings.get_word_vector("cat").shape == (3,)

    def test_model_path_can_also_be_given_as_path(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")
        registry = _registry(
            tmp_path,
            {
                "name": "test-static",
                "path": directory,
                "format": "static-transformer",
                "lang": "en",
                "item": "word",
            },
        )

        assert Embeddings("test-static", resource_registry=registry).embed_size == 4

    def test_local_model_used_without_any_registry_entry(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")
        registry = _registry(tmp_path, {"name": "glove-840B", "format": "vec", "lang": "en"})

        embeddings = Embeddings(directory, resource_registry=registry)

        assert embeddings.embed_size == 4
        np.testing.assert_allclose(embeddings.get_word_vector("cat"), MATRIX[2])

    def test_embeddings_survive_serialization(self, tmp_path):
        directory = make_model2vec_model(tmp_path / "potion")
        registry = _registry(
            tmp_path,
            {
                "name": "test-static",
                "model": directory,
                "format": "static-transformer",
                "lang": "en",
                "item": "word",
            },
        )

        embeddings = pickle.loads(pickle.dumps(Embeddings("test-static", resource_registry=registry)))

        assert embeddings.embed_size == 4
        np.testing.assert_allclose(embeddings.get_word_vector("cats"), (MATRIX[2] + MATRIX[3]) / 2)

    def test_no_lmdb_database_is_attached(self, tmp_path):
        """
        The name of a static embedding model may be a path; it must never be
        taken for the path of an LMDB database, which the DataLoader workers
        would then try to reopen.
        """
        directory = make_model2vec_model(tmp_path / "potion")
        registry = _registry(tmp_path, {"name": "glove-840B", "format": "vec", "lang": "en"})

        embeddings = Embeddings(directory, resource_registry=registry)

        assert embeddings.lmdb_env_path() is None
        assert embeddings.has_lmdb_env() is False
        embeddings.reopen_lmdb()
        assert embeddings.env is None
        np.testing.assert_allclose(pickle.loads(pickle.dumps(embeddings)).get_word_vector("cat"), MATRIX[2])


class TestUnreachableModel:
    def test_missing_model_reports_why_it_could_not_be_retrieved(self):
        """Offline, with nothing in the cache: the reason must reach the user."""
        with pytest.raises(ValueError) as error:
            StaticTransformerEmbeddings("delft-tests/not-a-static-embedding-model", local_files_only=True)

        assert "not-a-static-embedding-model" in str(error.value)
        assert "could not be retrieved" in str(error.value)

    def test_empty_model_reference_is_rejected(self):
        with pytest.raises(ValueError):
            StaticTransformerEmbeddings("")
