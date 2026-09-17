"""
Tests for the contextual embeddings backend (hidden states of a frozen
transformer used as word vectors by the RNN architectures).

The tests build a tiny random BERT locally, so that nothing is downloaded from
the HuggingFace hub.
"""

import json
import multiprocessing as mp
import os
import pickle

import numpy as np
import pytest

from delft.sequenceLabelling.preprocess import to_vector_single
from delft.utilities.ContextualEmbeddings import ContextualEmbeddings
from delft.utilities.Embeddings import Embeddings

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "far", "away", "now"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["##s", "##ing"]

HIDDEN_SIZE = 16
NB_LAYERS = 3
MAX_POSITIONS = 64


def make_tiny_bert(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    directory = str(directory)
    os.makedirs(directory, exist_ok=True)

    vocab = {piece: index for index, piece in enumerate(VOCAB)}
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    fast_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_POSITIONS,
    )
    fast_tokenizer.save_pretrained(directory)

    torch.manual_seed(7)
    config = transformers.BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NB_LAYERS,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=MAX_POSITIONS,
    )
    transformers.BertModel(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    return make_tiny_bert(tmp_path_factory.mktemp("tiny-bert"))


def _registry(tmp_path, entry):
    return {
        "embedding-lmdb-path": str(tmp_path / "db"),
        "embeddings": [entry],
        "embeddings-contextualized": [],
        "transformers": [],
    }


def _entry(model_directory, **options):
    entry = {
        "name": "tiny-contextual",
        "model": model_directory,
        "format": "contextual-transformer",
        "lang": "en",
    }
    entry.update(options)
    return entry


SENTENCE = ["the", "cats", "sat", "on", "the", "mat"]
LONG_SENTENCE = [WORDS[i % len(WORDS)] for i in range(40)]


class TestWordVectors:
    def test_one_vector_per_token(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        vectors = embeddings.embed_batch([SENTENCE])[0]
        assert vectors.shape == (len(SENTENCE), HIDDEN_SIZE)
        assert vectors.dtype == np.float32

    def test_vectors_depend_on_the_context(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        first, second = embeddings.embed_batch([["the", "cat", "sat"], ["the", "cat", "ran"]])
        assert not np.allclose(first[1], second[1])

    def test_vectors_do_not_depend_on_the_batch(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        alone = embeddings.embed_batch([SENTENCE])[0]
        batched = embeddings.embed_batch([LONG_SENTENCE, SENTENCE, ["dog"]])[1]
        np.testing.assert_allclose(alone, batched, atol=1e-5)

    def test_matches_the_hidden_states_of_the_transformer(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", layers=(-1,), subword_pooling="first")
        vectors = embeddings.embed_batch([SENTENCE])[0]

        tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_bert)
        model = transformers.AutoModel.from_pretrained(tiny_bert).eval()
        encoding = tokenizer(SENTENCE, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**encoding).last_hidden_state[0].numpy()
        word_ids = encoding.word_ids()
        for word_index in range(len(SENTENCE)):
            np.testing.assert_allclose(vectors[word_index], hidden[word_ids.index(word_index)], atol=1e-5)

    def test_subword_pooling(self, tiny_bert):
        # "cats" is split in "cat" + "##s"
        vectors = {}
        for pooling in ("first", "last", "mean"):
            embeddings = ContextualEmbeddings(tiny_bert, device="cpu", subword_pooling=pooling)
            vectors[pooling] = embeddings.embed_batch([SENTENCE])[0]
        np.testing.assert_allclose(vectors["mean"][1], (vectors["first"][1] + vectors["last"][1]) / 2, atol=1e-5)
        assert not np.allclose(vectors["first"][1], vectors["last"][1])
        # words made of a single unit are not affected
        np.testing.assert_allclose(vectors["first"][0], vectors["last"][0], atol=1e-6)

    def test_layer_pooling(self, tiny_bert):
        concat = ContextualEmbeddings(tiny_bert, device="cpu", layers=(-2, -1), layer_pooling="concat")
        mean = ContextualEmbeddings(tiny_bert, device="cpu", layers=(-2, -1), layer_pooling="mean")
        total = ContextualEmbeddings(tiny_bert, device="cpu", layers=(-2, -1), layer_pooling="sum")
        assert concat.embed_size == 2 * HIDDEN_SIZE
        assert mean.embed_size == HIDDEN_SIZE
        concat_vectors = concat.embed_batch([SENTENCE])[0]
        mean_vectors = mean.embed_batch([SENTENCE])[0]
        np.testing.assert_allclose(
            mean_vectors, (concat_vectors[:, :HIDDEN_SIZE] + concat_vectors[:, HIDDEN_SIZE:]) / 2, atol=1e-5
        )
        np.testing.assert_allclose(total.embed_batch([SENTENCE])[0], 2 * mean_vectors, atol=1e-5)

    def test_token_without_any_subword_unit_gets_a_zero_vector(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        vectors = embeddings.embed_batch([["the", " ", "cat"]])[0]
        assert vectors.shape == (3, HIDDEN_SIZE)
        assert not vectors[1].any()
        assert vectors[0].any() and vectors[2].any()

    def test_empty_sentence(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        assert embeddings.embed_batch([[]])[0].shape == (0, HIDDEN_SIZE)
        assert embeddings.get_sentence_vectors([]).shape == (0, HIDDEN_SIZE)

    def test_invalid_options_are_rejected(self, tiny_bert):
        with pytest.raises(ValueError):
            ContextualEmbeddings(tiny_bert, layer_pooling="max")
        with pytest.raises(ValueError):
            ContextualEmbeddings(tiny_bert, subword_pooling="max")
        with pytest.raises(ValueError):
            ContextualEmbeddings(tiny_bert, layers=(-12,))


class TestWindows:
    def test_window_is_bound_by_the_model(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", window=512)
        assert embeddings.window == MAX_POSITIONS

    def test_long_sequence_is_covered_by_windows(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", window=12, stride=5)
        assert len(embeddings._windows(len(LONG_SENTENCE))) > 1
        vectors = embeddings.embed_batch([LONG_SENTENCE])[0]
        assert vectors.shape == (len(LONG_SENTENCE), HIDDEN_SIZE)
        # every token got a vector
        assert np.abs(vectors).sum(axis=1).min() > 0

    def test_windows_cover_the_whole_sequence(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", window=12, stride=5)
        capacity = 10
        for nb_pieces in (1, 10, 11, 23, 40, 41):
            starts = embeddings._windows(nb_pieces)
            covered = set()
            for start in starts:
                covered.update(range(start, min(start + capacity, nb_pieces)))
            assert covered == set(range(nb_pieces))
            assert all(start + capacity <= max(nb_pieces, capacity) for start in starts)

    def test_short_sequence_is_not_affected_by_the_window(self, tiny_bert):
        windowed = ContextualEmbeddings(tiny_bert, device="cpu", window=12, stride=5)
        full = ContextualEmbeddings(tiny_bert, device="cpu")
        np.testing.assert_allclose(windowed.embed_batch([SENTENCE])[0], full.embed_batch([SENTENCE])[0], atol=1e-5)

    def test_long_sequence_vectors_do_not_depend_on_the_batch(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", window=12, stride=5, batch_size=3)
        alone = embeddings.embed_batch([LONG_SENTENCE])[0]
        batched = embeddings.embed_batch([SENTENCE, LONG_SENTENCE])[1]
        np.testing.assert_allclose(alone, batched, atol=1e-5)


class TestCache:
    def test_precompute_fills_the_cache_once(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        sentences = [SENTENCE, LONG_SENTENCE, SENTENCE]
        assert embeddings.precompute(sentences, verbose=False) == 2
        assert embeddings.precompute(sentences, verbose=False) == 0
        assert os.path.isdir(embeddings.cache_env_path())
        with open(os.path.join(embeddings.cache_env_path(), "fingerprint.json")) as f:
            assert json.load(f)["model"] == tiny_bert

    def test_cached_vectors_are_the_computed_ones(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embeddings.precompute([SENTENCE], verbose=False)
        embeddings.release_model()
        cached = embeddings.get_sentence_vectors(SENTENCE)
        assert embeddings._model is None, "a cached sentence must not load the transformer"
        assert cached.dtype == np.float32
        np.testing.assert_allclose(cached, embeddings.embed_batch([SENTENCE])[0], atol=1e-3)

    def test_cached_and_uncached_lookups_give_the_same_vectors(self, tiny_bert, tmp_path):
        cached = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        cached.precompute([SENTENCE], verbose=False)
        uncached = ContextualEmbeddings(tiny_bert, device="cpu")
        np.testing.assert_array_equal(cached.get_sentence_vectors(SENTENCE), uncached.get_sentence_vectors(SENTENCE))

    def test_truncation_matches_the_datasets(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embeddings.precompute([LONG_SENTENCE], max_sequence_length=8, verbose=False)
        embeddings.release_model()
        vectors = embeddings.get_sentence_vectors(LONG_SENTENCE[:8])
        assert vectors.shape == (8, HIDDEN_SIZE)
        assert embeddings._model is None

    def test_different_settings_do_not_share_a_cache(self, tiny_bert, tmp_path):
        first = ContextualEmbeddings(tiny_bert, cache_path=str(tmp_path), subword_pooling="first")
        mean = ContextualEmbeddings(tiny_bert, cache_path=str(tmp_path), subword_pooling="mean")
        assert first.cache_env_path() != mean.cache_env_path()

    def test_cache_is_shared_between_instances(self, tiny_bert, tmp_path):
        ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path)).precompute([SENTENCE], verbose=False)
        other = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        assert other.precompute([SENTENCE], verbose=False) == 0

    def test_cache_can_be_extended_after_being_read(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embeddings.precompute([SENTENCE], verbose=False)
        embeddings.get_sentence_vectors(SENTENCE)
        assert embeddings.precompute([LONG_SENTENCE], verbose=False) == 1
        assert embeddings.get_sentence_vectors(LONG_SENTENCE).shape == (len(LONG_SENTENCE), HIDDEN_SIZE)

    def test_without_a_cache_path_vectors_are_kept_in_memory(self, tiny_bert):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        assert embeddings.cache_env_path() is None
        assert embeddings.precompute([SENTENCE], verbose=False) == 1
        assert embeddings.precompute([SENTENCE], verbose=False) == 0

    def test_transient_vectors_are_not_written_on_disk(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embeddings.precompute([SENTENCE], persist=False, verbose=False)
        assert not os.path.isdir(embeddings.cache_env_path())
        assert len(embeddings._memory) == 1
        # forgotten with the next call
        embeddings.precompute([LONG_SENTENCE], persist=False, verbose=False)
        assert list(embeddings._memory) == [embeddings.sentence_key(LONG_SENTENCE)]


def _shards(embeddings):
    return sorted(name for name in os.listdir(embeddings.cache_env_path()) if name.startswith("shard-"))


class TestConcurrentCaches:
    """Several trainings (a SLURM job array) feed the same cache at the same time."""

    def test_each_writer_publishes_its_own_shard(self, tiny_bert, tmp_path):
        first = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        second = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        # both look at the cache before any of them has written anything
        first._open_shards()
        second._open_shards()
        first.precompute([SENTENCE], verbose=False)
        second.precompute([LONG_SENTENCE], verbose=False)
        assert len(_shards(first)) == 2
        # nothing else than complete shards is left behind
        assert sorted(os.listdir(first.cache_env_path())) == ["fingerprint.json"] + _shards(first)

    def test_a_shard_published_by_another_process_is_picked_up(self, tiny_bert, tmp_path):
        reader = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        reader.precompute([SENTENCE], verbose=False)
        reader.get_sentence_vectors(SENTENCE)
        reader.release_model()

        writer = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        writer.precompute([LONG_SENTENCE], verbose=False)

        assert reader.get_sentence_vectors(LONG_SENTENCE).shape == (len(LONG_SENTENCE), HIDDEN_SIZE)
        assert reader._model is None
        assert reader.precompute([SENTENCE, LONG_SENTENCE], verbose=False) == 0

    def test_same_sentences_embedded_twice_stay_readable(self, tiny_bert, tmp_path):
        first = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        second = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        pending = {first.sentence_key(SENTENCE): SENTENCE}
        first._write_shard(first.cache_env_path(), list(pending.items()), False)
        second._write_shard(second.cache_env_path(), list(pending.items()), False)
        assert len(_shards(first)) == 2
        np.testing.assert_array_equal(first.get_sentence_vectors(SENTENCE), second.get_sentence_vectors(SENTENCE))

    def test_what_was_computed_before_a_failure_is_kept(self, tiny_bert, tmp_path, monkeypatch):
        import delft.utilities.ContextualEmbeddings as module

        monkeypatch.setattr(module, "PRECOMPUTE_CHUNK_SIZE", 1)
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embed_batch = embeddings.embed_batch
        calls = []

        def failing_embed_batch(token_lists):
            calls.append(token_lists)
            if len(calls) == 2:
                raise RuntimeError("out of memory")
            return embed_batch(token_lists)

        embeddings.embed_batch = failing_embed_batch
        with pytest.raises(RuntimeError):
            embeddings.precompute([LONG_SENTENCE, SENTENCE], verbose=False)
        embeddings.embed_batch = embed_batch

        assert len(_shards(embeddings)) == 1
        # the longest sentence went first and is cached, the other one is not
        assert embeddings.precompute([LONG_SENTENCE, SENTENCE], verbose=False) == 1

    def test_nothing_is_left_when_nothing_could_be_computed(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))

        def failing_embed_batch(token_lists):
            raise RuntimeError("out of memory")

        embeddings.embed_batch = failing_embed_batch
        with pytest.raises(RuntimeError):
            embeddings.precompute([SENTENCE], verbose=False)
        assert os.listdir(embeddings.cache_env_path()) == ["fingerprint.json"]

    def test_large_corpus_fits_in_a_shard(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path), window=12, stride=5)
        rng = np.random.RandomState(1)
        sentences = [[WORDS[i] for i in rng.randint(0, len(WORDS), rng.randint(1, 60))] for _ in range(600)]
        embedded = embeddings.precompute(sentences, verbose=False)
        assert embedded == len({tuple(sentence) for sentence in sentences})
        assert embeddings.precompute(sentences, verbose=False) == 0


def _distributed_worker(rank, world_size, port, model_directory, cache_path, sentences, queue):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        embeddings = ContextualEmbeddings(model_directory, device="cpu", cache_path=cache_path)
        embedded = embeddings.precompute(sentences, verbose=False, distributed=True)
        # after the call, every process sees every sentence
        missing = len(embeddings._not_cached({embeddings.sentence_key(s): s for s in sentences}))
        queue.put((rank, embedded, missing))
    finally:
        dist.destroy_process_group()


class TestDistributed:
    def test_sentences_are_shared_out_between_the_processes(self, tiny_bert, tmp_path):
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        rng = np.random.RandomState(2)
        sentences = [[WORDS[i] for i in rng.randint(0, len(WORDS), rng.randint(2, 20))] for _ in range(40)]
        nb_distinct = len({tuple(sentence) for sentence in sentences})

        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_worker, args=(rank, 2, port, tiny_bert, str(tmp_path), sentences, queue)
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results = sorted(queue.get(timeout=180) for _ in processes)
        for process in processes:
            process.join(timeout=60)

        assert [missing for _, _, missing in results] == [0, 0]
        assert sum(embedded for _, embedded, _ in results) == nb_distinct
        assert all(embedded > 0 for _, embedded, _ in results)


def _worker(embeddings, tokens, queue):
    vectors = embeddings.get_sentence_vectors(tokens)
    queue.put((vectors, embeddings.model._model is None))


class TestSerialization:
    def test_serialization_drops_the_transformer(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        embeddings.precompute([SENTENCE], verbose=False)
        expected = embeddings.get_sentence_vectors(SENTENCE)

        payload = pickle.dumps(embeddings)
        assert len(payload) < 20000
        restored = pickle.loads(payload)
        assert restored._model is None and restored._tokenizer is None
        np.testing.assert_array_equal(restored.get_sentence_vectors(SENTENCE), expected)
        assert restored._model is None

    def test_spawned_worker_reads_the_cache(self, tiny_bert, tmp_path):
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        embeddings.precompute([SENTENCE], verbose=False)
        expected = embeddings.get_sentence_vectors(SENTENCE)

        context = mp.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_worker, args=(embeddings, SENTENCE, queue))
        process.start()
        vectors, transformer_not_loaded = queue.get(timeout=120)
        process.join(timeout=120)
        np.testing.assert_array_equal(vectors, expected)
        assert transformer_not_loaded


class TestEmbeddingsIntegration:
    def test_loaded_from_the_embeddings_registry(self, tiny_bert, tmp_path):
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        assert embeddings.embed_size == HIDDEN_SIZE
        assert embeddings.extension == "contextual-transformer"
        assert embeddings.lmdb_env_path() is None
        assert embeddings.model.cache_env_path().startswith(str(tmp_path / "db" / "contextual"))

    def test_options_from_the_registry(self, tiny_bert, tmp_path):
        entry = _entry(
            tiny_bert,
            **{"layers": [-2, -1], "layer-pooling": "concat", "subword-pooling": "mean", "window": 12, "stride": 5},
        )
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, entry))
        assert embeddings.embed_size == 2 * HIDDEN_SIZE
        assert embeddings.model.subword_pooling == "mean"
        assert embeddings.model.window == 12

    def test_transformer_used_without_any_registry_entry(self, tiny_bert, tmp_path):
        registry = _registry(tmp_path, _entry(tiny_bert))
        registry["embeddings"] = []
        embeddings = Embeddings("contextual:" + tiny_bert, resource_registry=registry)
        assert embeddings.extension == "contextual-transformer"
        assert embeddings.embed_size == HIDDEN_SIZE
        assert embeddings.model.model_reference == tiny_bert
        assert embeddings.lmdb_env_path() is None

    def test_cache_can_be_disabled(self, tiny_bert, tmp_path):
        entry = _entry(tiny_bert, cache=False)
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, entry))
        assert embeddings.model.cache_env_path() is None

    def test_to_vector_single_embeds_the_whole_sentence(self, tiny_bert, tmp_path):
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        x = to_vector_single(SENTENCE, embeddings, 10)
        assert x.shape == (10, HIDDEN_SIZE)
        assert x.dtype == np.float32
        np.testing.assert_array_equal(x[: len(SENTENCE)], embeddings.get_sentence_vectors(SENTENCE))
        assert not x[len(SENTENCE) :].any()
        # the same word has different vectors at different positions
        assert not np.allclose(x[0], x[4])

    def test_word_vector_out_of_context(self, tiny_bert, tmp_path):
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        assert embeddings.get_word_vector("cat").shape == (HIDDEN_SIZE,)


class TestSequenceLabelling:
    def test_rnn_architecture_trains_and_tags_with_contextual_embeddings(self, tiny_bert, tmp_path, monkeypatch):
        import delft.sequenceLabelling.wrapper as wrapper

        registry = _registry(tmp_path, _entry(tiny_bert, window=12, stride=5))
        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: registry)
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
            "contextual-test",
            architecture="BidLSTM_CRF",
            embeddings_name="tiny-contextual",
            max_epoch=1,
            batch_size=10,
            max_sequence_length=25,
            early_stop=False,
            nb_workers=0,
        )
        # a frozen transformer is an embedding, not a transformer architecture
        assert model.model_config.transformer_name is None
        assert model.model_config.word_embedding_size == HIDDEN_SIZE

        model.train(x[:20], y[:20], x_valid=x[20:], y_valid=y[20:])
        contextual = model.embeddings.model
        assert contextual.precompute(list(x), max_sequence_length=25, verbose=False) == 0

        result = model.tag(["the cat sat on the mat"], "json")
        assert len(result["texts"]) == 1
        # vectors of tagged texts stay in memory, they are not written in the cache
        tagged = ["the", "cat", "sat", "on", "the", "mat"]
        assert contextual.sentence_key(tagged) in contextual._memory
        other = ContextualEmbeddings(tiny_bert, device="cpu", window=12, stride=5, cache_path=contextual.cache_path)
        assert other.cache_env_path() == contextual.cache_env_path()
        assert other.precompute([tagged], verbose=False) == 1
