"""
Tests for stacked embeddings: several embeddings used together, typically a
static one and a contextual one, the vector of a word being the concatenation
of their vectors.

Everything is built locally (a small LMDB database of word vectors, a tiny
static embedding model, a tiny random BERT), nothing is downloaded.
"""

import multiprocessing as mp
import os
import pickle

import lmdb
import numpy as np
import pytest

from delft.sequenceLabelling.preprocess import to_vector_single
from delft.utilities.Embeddings import Embeddings
from delft.utilities.StackedEmbeddings import StackedEmbeddings
from tests.utilities.test_contextual_embeddings import HIDDEN_SIZE, LONG_SENTENCE, SENTENCE, WORDS, make_tiny_bert
from tests.utilities.test_static_embeddings import MATRIX, make_model2vec_model

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

WORD_SIZE = 12
STATIC_SIZE = MATRIX.shape[1]


def _word_vector(index):
    return np.random.RandomState(index).rand(WORD_SIZE).astype(np.float32) + 0.1


def make_word_database(lmdb_path, name="tiny-glove"):
    """A database of word vectors as DeLFT compiles them from a glove or word2vec file."""
    # "000" is what a number becomes once normalized, "123" is not in the vocabulary
    vocabulary = WORDS + ["000"] + ["filler%d" % i for i in range(150)]
    path = os.path.join(str(lmdb_path), name)
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, map_size=16 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for index, word in enumerate(vocabulary):
            txn.put(word.encode("utf-8"), _word_vector(index).tobytes())
    env.close()
    return {word: _word_vector(index) for index, word in enumerate(vocabulary)}


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    return make_tiny_bert(tmp_path_factory.mktemp("tiny-bert"))


@pytest.fixture
def resources(tiny_bert, tmp_path):
    """A registry with a word database, a static embedding model and contextual embeddings."""
    words = make_word_database(tmp_path / "db")
    static_model = make_model2vec_model(tmp_path / "tiny-static")
    registry = {
        "embedding-lmdb-path": str(tmp_path / "db"),
        "embedding-download-path": str(tmp_path / "download"),
        "embeddings": [
            {"name": "tiny-glove", "path": "/nowhere/tiny-glove.txt", "type": "glove", "format": "vec", "lang": "en"},
            {"name": "tiny-static", "model": static_model, "format": "static-transformer", "lang": "en"},
            {"name": "tiny-contextual", "model": tiny_bert, "format": "contextual-transformer", "lang": "en"},
        ],
        "embeddings-contextualized": [],
        "transformers": [],
    }
    return {"registry": registry, "words": words, "static_model": static_model, "tiny_bert": tiny_bert}


def _stack(resources, name="tiny-glove+tiny-contextual"):
    return Embeddings(name, resource_registry=resources["registry"])


class TestConcatenation:
    def test_size_is_the_sum_of_the_sizes(self, resources):
        embeddings = _stack(resources)
        assert embeddings.extension == "stacked"
        assert embeddings.embed_size == WORD_SIZE + HIDDEN_SIZE
        assert [component.name for component in embeddings.components] == ["tiny-glove", "tiny-contextual"]

    def test_vectors_are_those_of_each_embedding_used_alone(self, resources):
        stacked = to_vector_single(SENTENCE, _stack(resources), 10)
        assert stacked.shape == (10, WORD_SIZE + HIDDEN_SIZE)
        assert stacked.dtype == np.float32

        glove = Embeddings("tiny-glove", resource_registry=resources["registry"])
        contextual = Embeddings("tiny-contextual", resource_registry=resources["registry"])
        np.testing.assert_array_equal(stacked[:, :WORD_SIZE], to_vector_single(SENTENCE, glove, 10))
        np.testing.assert_array_equal(stacked[:, WORD_SIZE:], to_vector_single(SENTENCE, contextual, 10))

        # "the" twice: same static vector, different contextual vectors
        np.testing.assert_array_equal(stacked[0, :WORD_SIZE], resources["words"]["the"])
        np.testing.assert_array_equal(stacked[0, :WORD_SIZE], stacked[4, :WORD_SIZE])
        assert not np.allclose(stacked[0, WORD_SIZE:], stacked[4, WORD_SIZE:])
        # "cats" is not in the word database, the transformer still has a vector for it
        assert not stacked[1, :WORD_SIZE].any()
        assert stacked[1, WORD_SIZE:].any()
        # padding
        assert not stacked[len(SENTENCE) :].any()

    def test_whole_text_tokenization_reaches_the_contextual_component(self, resources):
        embeddings = Embeddings(
            "tiny-glove+tiny-contextual", resource_registry=resources["registry"], whole_text_tokenization=True
        )
        assert embeddings.components[1].model.whole_text_tokenization is True
        assert _stack(resources).components[1].model.whole_text_tokenization is False

    def test_order_is_the_order_of_the_name(self, resources):
        forward = to_vector_single(SENTENCE, _stack(resources, "tiny-glove+tiny-contextual"), 8)
        backward = to_vector_single(SENTENCE, _stack(resources, "tiny-contextual+tiny-glove"), 8)
        np.testing.assert_array_equal(forward[:, :WORD_SIZE], backward[:, HIDDEN_SIZE:])
        np.testing.assert_array_equal(forward[:, WORD_SIZE:], backward[:, :HIDDEN_SIZE])

    def test_each_embedding_keeps_its_own_handling_of_numbers(self, resources):
        embeddings = _stack(resources)
        contextual = embeddings.components[1].model
        seen = []
        get_sentence_vectors = contextual.get_sentence_vectors

        def spy(tokens):
            seen.append(list(tokens))
            return get_sentence_vectors(tokens)

        contextual.get_sentence_vectors = spy
        vectors = to_vector_single(["the", "123"], embeddings, 2)
        # normalized for the word database, as it is written for the transformer
        np.testing.assert_array_equal(vectors[1, :WORD_SIZE], resources["words"]["000"])
        assert seen == [["the", "123"]]

    def test_more_than_two_embeddings(self, resources):
        embeddings = _stack(resources, "tiny-glove+tiny-static+tiny-contextual")
        assert embeddings.embed_size == WORD_SIZE + STATIC_SIZE + HIDDEN_SIZE
        vectors = to_vector_single(["the", "cat"], embeddings, 2)
        np.testing.assert_allclose(vectors[0, WORD_SIZE : WORD_SIZE + STATIC_SIZE], MATRIX[1])

    def test_several_tokens_per_position(self, resources):
        # the text of a position can hold several tokens, taken from columns of the features
        # (see delft.sequenceLabelling.text_features): each embedding concatenates its own
        embeddings = _stack(resources, "tiny-glove+tiny-static")
        text = ["the cat", "cat the"]
        vectors = to_vector_single(text, embeddings, 3, tokens_per_position=2)
        assert vectors.shape == (3, embeddings.embed_size * 2)
        glove, static = embeddings.components
        np.testing.assert_array_equal(
            vectors[:, : WORD_SIZE * 2], to_vector_single(text, glove, 3, tokens_per_position=2)
        )
        np.testing.assert_array_equal(
            vectors[:, WORD_SIZE * 2 :], to_vector_single(text, static, 3, tokens_per_position=2)
        )
        np.testing.assert_array_equal(vectors[0, :WORD_SIZE], resources["words"]["the"])
        np.testing.assert_array_equal(vectors[0, WORD_SIZE : WORD_SIZE * 2], resources["words"]["cat"])
        assert not vectors[2].any()

    def test_several_tokens_per_position_with_contextual_embeddings(self, resources):
        """Their tokens are read as one sentence by the contextual part, whose vectors go back to their columns."""
        embeddings = _stack(resources)
        glove, contextual = embeddings.components
        text = ["the cat", "sat ", "on the"]  # a position without a second token
        assert embeddings.precompute([["the", "cat", "sat", "on", "the"]], verbose=False) == 1
        embeddings.release_model()

        vectors = to_vector_single(text, embeddings, 4, tokens_per_position=2)
        assert vectors.shape == (4, embeddings.embed_size * 2)
        assert contextual.model._model is None, "the sentence of the columns is the one that was computed"
        np.testing.assert_array_equal(
            vectors[:, : WORD_SIZE * 2], to_vector_single(text, glove, 4, tokens_per_position=2)
        )
        sentence = contextual.get_sentence_vectors(["the", "cat", "sat", "on", "the"])
        contextual_part = vectors[:, WORD_SIZE * 2 :]
        np.testing.assert_array_equal(contextual_part[0, :HIDDEN_SIZE], sentence[0])
        np.testing.assert_array_equal(contextual_part[0, HIDDEN_SIZE:], sentence[1])
        np.testing.assert_array_equal(contextual_part[1, :HIDDEN_SIZE], sentence[2])
        assert not contextual_part[1, HIDDEN_SIZE:].any()
        np.testing.assert_array_equal(contextual_part[2, HIDDEN_SIZE:], sentence[4])

    def test_word_vector_out_of_context(self, resources):
        vector = _stack(resources).get_word_vector("cat")
        assert vector.shape == (WORD_SIZE + HIDDEN_SIZE,)
        assert vector.dtype == np.float32
        np.testing.assert_array_equal(vector[:WORD_SIZE], resources["words"]["cat"])

    def test_no_word_database_is_attached_to_the_stack_itself(self, resources):
        embeddings = _stack(resources)
        assert embeddings.lmdb_env_path() is None
        assert embeddings.components[0].has_lmdb_env()


class TestNames:
    def test_described_in_the_embeddings_registry(self, resources):
        resources["registry"]["embeddings"].append(
            {"name": "tiny-both", "format": "stacked", "embeddings": ["tiny-glove", "tiny-contextual"], "lang": "en"}
        )
        embeddings = Embeddings("tiny-both", resource_registry=resources["registry"])
        assert embeddings.embed_size == WORD_SIZE + HIDDEN_SIZE
        np.testing.assert_array_equal(
            to_vector_single(SENTENCE, embeddings, 8), to_vector_single(SENTENCE, _stack(resources), 8)
        )

    def test_components_that_are_not_in_the_registry(self, resources):
        name = "%s+contextual:%s" % (resources["static_model"], resources["tiny_bert"])
        embeddings = Embeddings(name, resource_registry=resources["registry"])
        assert [component.extension for component in embeddings.components] == [
            "static-transformer",
            "contextual-transformer",
        ]
        assert embeddings.embed_size == STATIC_SIZE + HIDDEN_SIZE

    def test_existing_path_containing_the_separator_is_not_a_stack(self, resources, tmp_path):
        directory = make_model2vec_model(tmp_path / "a+b")
        embeddings = Embeddings(directory, resource_registry=resources["registry"])
        assert embeddings.extension == "static-transformer"

    @pytest.mark.parametrize("name", ["tiny-glove+", "+tiny-glove", "tiny-glove++tiny-contextual"])
    def test_empty_component_is_rejected(self, resources, name):
        with pytest.raises(ValueError):
            Embeddings(name, resource_registry=resources["registry"])

    def test_registry_entry_with_a_single_embedding_is_rejected(self, resources):
        resources["registry"]["embeddings"].append({"name": "alone", "format": "stacked", "embeddings": ["tiny-glove"]})
        with pytest.raises(ValueError):
            Embeddings("alone", resource_registry=resources["registry"])

    def test_stack_of_stacks_is_rejected(self, resources):
        resources["registry"]["embeddings"].append(
            {"name": "tiny-both", "format": "stacked", "embeddings": ["tiny-glove", "tiny-contextual"]}
        )
        with pytest.raises(ValueError):
            Embeddings("tiny-both+tiny-static", resource_registry=resources["registry"])
        resources["registry"]["embeddings"].append(
            {"name": "looping", "format": "stacked", "embeddings": ["looping", "tiny-glove"]}
        )
        with pytest.raises(ValueError):
            Embeddings("looping", resource_registry=resources["registry"])

    def test_at_least_two_embeddings(self, resources):
        with pytest.raises(ValueError):
            StackedEmbeddings([Embeddings("tiny-glove", resource_registry=resources["registry"])])


class TestPrecompute:
    def test_contextual_embeddings_are_precomputed(self, resources):
        embeddings = _stack(resources)
        assert hasattr(embeddings, "precompute")
        assert embeddings.precompute([SENTENCE, LONG_SENTENCE, SENTENCE], verbose=False) == 2
        assert embeddings.precompute([SENTENCE, LONG_SENTENCE], verbose=False) == 0

        contextual = embeddings.components[1].model
        contextual.release_model()
        to_vector_single(SENTENCE, embeddings, 8)
        assert contextual._model is None, "precomputed sentences must not load the transformer"

    def test_cache_is_the_one_of_the_contextual_embeddings_used_alone(self, resources):
        Embeddings("tiny-contextual", resource_registry=resources["registry"]).precompute([SENTENCE], verbose=False)
        assert _stack(resources).precompute([SENTENCE], verbose=False) == 0

    def test_options_are_passed_to_the_components(self, resources):
        embeddings = _stack(resources)
        embeddings.precompute([LONG_SENTENCE], max_sequence_length=8, persist=False, verbose=False)
        contextual = embeddings.components[1].model
        assert list(contextual._memory) == [contextual.sentence_key(LONG_SENTENCE[:8])]
        assert not os.path.isdir(contextual.cache_env_path())

    def test_nothing_to_precompute_with_static_embeddings_only(self, resources):
        embeddings = _stack(resources, "tiny-glove+tiny-static")
        assert embeddings.precompute([SENTENCE], verbose=False) == 0
        embeddings.release_model()  # no transformer to free
        assert to_vector_single(SENTENCE, embeddings, 8).shape == (8, WORD_SIZE + STATIC_SIZE)

    def test_the_transformer_of_the_stack_is_released(self, resources):
        """The data loaders free it once a corpus is embedded: the stack had no way to."""
        embeddings = _stack(resources)
        contextual = embeddings.components[1].model
        embeddings.precompute([SENTENCE], verbose=False)
        assert contextual._model is not None
        embeddings.release_model()
        assert contextual._model is None


def _worker(embeddings, tokens, queue):
    # what a DataLoader worker does: worker_init_fn, then the dataset lookups
    embeddings.reopen_lmdb()
    vectors = to_vector_single(tokens, embeddings, len(tokens))
    queue.put((vectors, embeddings.components[1].model._model is None))


class TestSerialization:
    def test_serialization_keeps_every_component_usable(self, resources):
        embeddings = _stack(resources)
        embeddings.precompute([SENTENCE], verbose=False)
        expected = to_vector_single(SENTENCE, embeddings, 8)

        restored = pickle.loads(pickle.dumps(embeddings))
        np.testing.assert_array_equal(to_vector_single(SENTENCE, restored, 8), expected)
        assert restored.components[1].model._model is None

    def test_reopening_the_databases_in_place(self, resources):
        embeddings = _stack(resources)
        embeddings.precompute([SENTENCE], verbose=False)
        expected = to_vector_single(SENTENCE, embeddings, 8)
        embeddings.reopen_lmdb()
        np.testing.assert_array_equal(to_vector_single(SENTENCE, embeddings, 8), expected)

    def test_spawned_worker_reads_the_word_database_and_the_cache(self, resources):
        embeddings = _stack(resources)
        embeddings.precompute([SENTENCE], verbose=False)
        expected = to_vector_single(SENTENCE, embeddings, len(SENTENCE))

        context = mp.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_worker, args=(embeddings, SENTENCE, queue))
        process.start()
        vectors, transformer_not_loaded = queue.get(timeout=120)
        process.join(timeout=120)
        np.testing.assert_array_equal(vectors, expected)
        assert transformer_not_loaded


class TestSequenceLabelling:
    def test_rnn_architecture_trains_tags_and_reloads_with_stacked_embeddings(self, resources, tmp_path, monkeypatch):
        import delft.sequenceLabelling.wrapper as wrapper

        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: resources["registry"])
        # checkpoints are written relatively to the working directory
        monkeypatch.chdir(tmp_path)

        rng = np.random.RandomState(0)
        x, y = [], []
        for _ in range(30):
            tokens = [WORDS[i] for i in rng.randint(0, len(WORDS), rng.randint(1, 30))]
            x.append(tokens)
            y.append(["B-ANIMAL" if token in ("cat", "dog") else "O" for token in tokens])
        x, y = np.array(x, dtype=object), np.array(y, dtype=object)

        model = wrapper.Sequence(
            "stacked-test",
            architecture="BidLSTM_CRF",
            embeddings_name="tiny-glove+tiny-contextual",
            max_epoch=1,
            batch_size=10,
            max_sequence_length=25,
            early_stop=False,
            nb_workers=0,
        )
        assert model.model_config.transformer_name is None
        assert model.model_config.word_embedding_size == WORD_SIZE + HIDDEN_SIZE

        model.train(x[:20], y[:20], x_valid=x[20:], y_valid=y[20:])
        assert model.embeddings.components[1].model._model is None, "the transformer is freed for the training"
        # the training and validation sets went to the cache of the contextual embeddings
        assert model.embeddings.precompute(list(x), max_sequence_length=25, verbose=False) == 0

        text = "the cat sat on the mat"
        expected = model.tag([text], "json")["texts"]

        model.save(str(tmp_path / "saved"))
        reloaded = wrapper.Sequence("stacked-test")
        reloaded.load(str(tmp_path / "saved"))
        assert reloaded.model_config.embeddings_name == "tiny-glove+tiny-contextual"
        assert reloaded.embeddings.embed_size == WORD_SIZE + HIDDEN_SIZE
        assert reloaded.tag([text], "json")["texts"] == expected


class TestTextClassification:
    """
    A text classifier took a stack word by word: the contextual part gave the vector of
    each word alone, computed by the dataset.
    """

    TEXTS = ["the cat sat on the mat", "the dog ran far away now", "cat 42 sat", "dog ran"] * 3
    CLASSES = np.array([[1, 0], [0, 1], [1, 0], [0, 1]] * 3, dtype=np.float32)

    def test_the_parts_embed_the_same_tokens(self, resources):
        from delft.textClassification.preprocess import to_vector_single as classification_vectors
        from delft.textClassification.preprocess import uses_sentence_vectors

        embeddings = _stack(resources)
        assert uses_sentence_vectors(embeddings)
        x = classification_vectors("the cat 42 sat", embeddings, maxlen=6)
        assert x.shape == (6, WORD_SIZE + HIDDEN_SIZE) and x.dtype == np.float32

        # the contextual part reads the text as it is written, as a whole
        tokens = ["the", "cat", "42", "sat"]
        contextual = embeddings.components[1]
        np.testing.assert_array_equal(x[:4, WORD_SIZE:], contextual.get_sentence_vectors(tokens))
        # the word database gives the vectors of the same tokens, and none for the number
        # the classifiers clean out of a text: "sat" stays next to its contextual vector
        words = resources["words"]
        np.testing.assert_array_equal(x[0, :WORD_SIZE], words["the"])
        np.testing.assert_array_equal(x[1, :WORD_SIZE], words["cat"])
        assert not x[2, :WORD_SIZE].any()
        np.testing.assert_array_equal(x[3, :WORD_SIZE], words["sat"])
        assert not x[4:].any()

    def test_a_stack_of_static_embeddings_is_used_as_each_of_them(self, resources):
        from delft.textClassification.preprocess import to_vector_single as classification_vectors
        from delft.textClassification.preprocess import uses_sentence_vectors

        embeddings = _stack(resources, "tiny-glove+tiny-static")
        assert not uses_sentence_vectors(embeddings)
        x = classification_vectors("the cat 42 sat", embeddings, maxlen=6)
        assert x.shape == (6, WORD_SIZE + STATIC_SIZE)
        # the text is cleaned, then cut into tokens: the number takes no position
        glove = embeddings.components[0]
        for position, word in enumerate(["the", "cat", "sat"]):
            np.testing.assert_array_equal(x[position, :WORD_SIZE], glove.get_word_vector(word))
        assert not x[3:].any()

    def test_classifier_trains_and_classifies_with_stacked_embeddings(self, resources, tmp_path, monkeypatch):
        import delft.textClassification.wrapper as wrapper

        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: resources["registry"])
        monkeypatch.chdir(tmp_path)

        classifier = wrapper.Classifier(
            "stacked-classifier",
            architecture="gru",
            embeddings_name="tiny-glove+tiny-contextual",
            list_classes=["a", "b"],
            maxlen=8,
            max_epoch=1,
            batch_size=4,
            early_stop=False,
            nb_workers=0,
            device="cpu",
        )
        assert classifier.model_config.word_embedding_size == WORD_SIZE + HIDDEN_SIZE
        contextual = classifier.embeddings.components[1].model

        classifier.train(self.TEXTS, self.CLASSES)
        assert contextual._model is None, "the transformer is freed once the texts are embedded"
        # the texts went to the cache of the contextual embeddings, as the sentences they are
        assert contextual.precompute([text.split() for text in set(self.TEXTS)], verbose=False) == 0

        scores = classifier.predict(["the cat sat", "dog ran far"], output_format="array")
        assert scores.shape == (2, 2)
        assert contextual.sentence_key(["the", "cat", "sat"]) in contextual._memory

        classifier.save(str(tmp_path / "saved"))
        reloaded = wrapper.Classifier("stacked-classifier", device="cpu", nb_workers=0)
        reloaded.load(str(tmp_path / "saved"))
        assert reloaded.model_config.embeddings_name == "tiny-glove+tiny-contextual"
        np.testing.assert_allclose(
            reloaded.predict(["the cat sat", "dog ran far"], output_format="array"), scores, rtol=1e-5
        )
