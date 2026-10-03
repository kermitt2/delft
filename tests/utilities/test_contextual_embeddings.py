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

import delft.utilities.ContextualEmbeddings as contextual_module
from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import Preprocessor, to_vector_single
from delft.utilities.ContextualEmbeddings import ContextualEmbeddings, local_model_revision
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

    def test_token_without_any_subword_unit_gets_a_zero_vector_and_a_rate_limited_warning(self, tiny_bert, capsys):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        vectors = embeddings.embed_batch([["the", " ", "cat"]])[0]
        assert vectors.shape == (3, HIDDEN_SIZE)
        assert not vectors[1].any()
        assert vectors[0].any() and vectors[2].any()
        warning = capsys.readouterr().out
        assert "warning:" in warning
        assert "' '" in warning
        embeddings.embed_batch([["", "the"]])
        assert "warning:" not in capsys.readouterr().out

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
    def test_ordinary_sentence_key_stays_compatible_with_existing_caches(self):
        assert ContextualEmbeddings.sentence_key(["the", "cat", "sat"]).hex() == (
            "c4a48330f76ba7e3d37c99c6166d5db429d177af"
        )

    def test_sentence_key_escapes_tokens_that_used_to_collide(self):
        assert ContextualEmbeddings.sentence_key(["a\x1fb", "c"]) != ContextualEmbeddings.sentence_key(["a", "b\x1fc"])

    @pytest.mark.parametrize(("cached", "read"), [([], [""]), ([""], [])])
    def test_empty_token_sequences_need_neither_cache_nor_model_in_a_worker(
        self, tiny_bert, tmp_path, monkeypatch, cached, read
    ):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        assert embeddings.precompute([cached], verbose=False) == 0
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())

        vectors = embeddings.get_sentence_vectors(read)

        assert vectors.shape == (len(read), HIDDEN_SIZE)
        assert not vectors.any()
        assert embeddings._model is None

    def test_cache_miss_in_a_dataloader_worker_is_an_error(self, tiny_bert, monkeypatch):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu")
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
        with pytest.raises(RuntimeError, match="cache miss.*DataLoader worker.*precompute.*main process"):
            embeddings.get_sentence_vectors(SENTENCE)
        assert embeddings._model is None

    def test_precompute_fills_the_cache_once(self, tiny_bert, tmp_path):
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        sentences = [SENTENCE, LONG_SENTENCE, SENTENCE]
        assert embeddings.precompute(sentences, verbose=False) == 2
        assert embeddings.precompute(sentences, verbose=False) == 0
        assert os.path.isdir(embeddings.cache_env_path())
        with open(os.path.join(embeddings.cache_env_path(), "fingerprint.json")) as f:
            fingerprint = json.load(f)
        # a local model is known by its type and its content, not by its path
        assert fingerprint["model"] == "bert" and fingerprint["revision"] == local_model_revision(tiny_bert)

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


def _retrained(directory, seed=11):
    """The tiny BERT of ``directory`` with other weights, written in its place."""
    torch.manual_seed(seed)
    config = transformers.AutoConfig.from_pretrained(directory)
    transformers.BertModel(config).save_pretrained(directory)


class TestRevision:
    """
    The cache was named after the reference of the model alone: a model updated under
    the same name kept the vectors of its previous weights for the sentences in the
    cache, and computed the others with the new ones.
    """

    def test_a_local_model_updated_in_place_gets_another_cache(self, tmp_path):
        directory = make_tiny_bert(tmp_path / "model")
        before = ContextualEmbeddings(directory, device="cpu", cache_path=str(tmp_path / "cache"))
        before.precompute([SENTENCE], verbose=False)
        assert before.fingerprint()["revision"] == before.revision == local_model_revision(directory)

        _retrained(directory)
        after = ContextualEmbeddings(directory, device="cpu", cache_path=str(tmp_path / "cache"))
        assert after.revision != before.revision
        assert after.cache_env_path() != before.cache_env_path()
        assert after.precompute([SENTENCE], verbose=False) == 1
        assert not np.allclose(after.get_sentence_vectors(SENTENCE), before.get_sentence_vectors(SENTENCE))

    def test_a_copy_of_a_local_model_has_its_revision(self, tiny_bert, tmp_path):
        """Whatever the dates of the files: a model copied to the disk of a node keeps its cache."""
        import shutil

        copy = str(tmp_path / "copy")
        shutil.copytree(tiny_bert, copy)
        for name in os.listdir(copy):
            os.utime(os.path.join(copy, name), (0, 0))
        assert local_model_revision(copy) == local_model_revision(tiny_bert)

    def test_a_copy_of_a_local_model_reads_its_cache(self, tiny_bert, tmp_path):
        """
        The path of the model was part of the name of the cache: a model staged under
        another path, or another name, on each node computed its vectors again.
        """
        import shutil

        original = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path / "cache"))
        original.precompute([SENTENCE], verbose=False)

        copy = str(tmp_path / "job-1234" / "staged")
        shutil.copytree(tiny_bert, copy)
        staged = ContextualEmbeddings(copy, device="cpu", cache_path=str(tmp_path / "cache"))
        assert staged.cache_env_path() == original.cache_env_path()
        assert os.path.basename(staged.cache_env_path()).startswith("bert-")
        assert staged.precompute([SENTENCE], verbose=False) == 0
        np.testing.assert_array_equal(staged.get_sentence_vectors(SENTENCE), original.get_sentence_vectors(SENTENCE))
        assert staged._model is None

    def test_a_model_of_the_hub_keeps_its_name(self, tiny_bert, tmp_path, monkeypatch):
        monkeypatch.setattr(ContextualEmbeddings, "_is_local_model", lambda self: False)
        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        assert embeddings.fingerprint()["model"] == tiny_bert
        assert os.path.basename(embeddings.cache_env_path()).startswith(os.path.basename(tiny_bert) + "-")

    def test_large_weights_are_sampled(self, tmp_path, monkeypatch):
        directory = make_tiny_bert(tmp_path / "model")
        monkeypatch.setattr(contextual_module, "REVISION_WHOLE_FILE_SIZE", 1024)
        monkeypatch.setattr(contextual_module, "REVISION_SAMPLES", 8)
        monkeypatch.setattr(contextual_module, "REVISION_SAMPLE_SIZE", 64)
        read = []
        real_open = open

        class Counting:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.file.close()

            def seek(self, offset):
                return self.file.seek(offset)

            def read(self, size=-1):
                data = self.file.read(size)
                read.append(len(data))
                return data

        monkeypatch.setattr(
            contextual_module, "open", lambda path, mode: Counting(real_open(path, mode)), raising=False
        )
        before = local_model_revision(directory)
        weights = os.path.getsize(os.path.join(directory, "model.safetensors"))
        large = [name for name in os.listdir(directory) if os.path.getsize(os.path.join(directory, name)) > 1024]
        assert "model.safetensors" in large
        assert read.count(64) == 8 * len(large), "8 samples of each large file, not the whole of it"
        assert max(read) <= 1024 < weights

        _retrained(directory)
        assert os.path.getsize(os.path.join(directory, "model.safetensors")) == weights
        assert local_model_revision(directory) != before

    def test_a_model_of_the_hub_is_pinned_to_the_commit_of_its_configuration(self, tiny_bert, tmp_path, monkeypatch):
        """The tokenizer and the weights are those of the commit the cache is named after."""
        calls = []

        def spy(loader, commit=None):
            real = loader.from_pretrained

            def from_pretrained(reference, **kwargs):
                calls.append((loader.__name__, kwargs.get("revision")))
                loaded = real(reference, **kwargs)  # the revision is ignored for a local directory
                if commit is not None:
                    # the other loaders ask for the configuration with what they did not use of their options
                    (loaded[0] if isinstance(loaded, tuple) else loaded)._commit_hash = commit
                return loaded

            monkeypatch.setattr(loader, "from_pretrained", from_pretrained)

        spy(transformers.AutoConfig, commit="0123abcd")
        spy(transformers.AutoTokenizer)
        spy(transformers.AutoModel)
        monkeypatch.setattr(ContextualEmbeddings, "_is_local_model", lambda self: False)

        embeddings = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        assert embeddings.revision == "0123abcd" and embeddings.fingerprint()["revision"] == "0123abcd"
        embeddings.precompute([SENTENCE], verbose=False)
        assert calls[0] == ("AutoConfig", None)
        assert ("AutoTokenizer", "0123abcd") in calls and ("AutoModel", "0123abcd") in calls
        assert all(revision == "0123abcd" for _, revision in calls[1:])

        # a worker, which loads the tokenizer again, takes the same commit
        del calls[:]
        restored = pickle.loads(pickle.dumps(embeddings))
        restored.embed_batch([SENTENCE])
        assert calls and all(revision == "0123abcd" for _, revision in calls)


class TestAccessToken:
    """HF_ACCESS_TOKEN, the token of DeLFT for the private models, reached none of the loaders."""

    @pytest.mark.parametrize("token", ["secret", None])
    def test_every_loader_is_given_the_token(self, tiny_bert, monkeypatch, token):
        if token is None:
            monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_ACCESS_TOKEN", token)
        tokens = {}
        for loader in (transformers.AutoConfig, transformers.AutoTokenizer, transformers.AutoModel):

            def from_pretrained(reference, real=loader.from_pretrained, name=loader.__name__, **kwargs):
                tokens.setdefault(name, []).append(kwargs.get("token"))
                return real(reference, **kwargs)

            monkeypatch.setattr(loader, "from_pretrained", from_pretrained)

        ContextualEmbeddings(tiny_bert, device="cpu").embed_batch([SENTENCE])
        assert sorted(tokens) == ["AutoConfig", "AutoModel", "AutoTokenizer"]
        assert all(given == [token] * len(given) for given in tokens.values())

    def test_the_token_is_not_kept_in_the_embeddings(self, tiny_bert, monkeypatch):
        """They are pickled for the DataLoader workers."""
        monkeypatch.setenv("HF_ACCESS_TOKEN", "secret-token")
        assert b"secret-token" not in pickle.dumps(ContextualEmbeddings(tiny_bert, device="cpu"))


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

    def test_instances_of_a_process_share_the_open_shards(self, tiny_bert, tmp_path):
        # py-lmdb 2 refuses to open an environment twice in a process: a second
        # instance reads the shards the first one opened, rather than embedding again
        first = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        first.precompute([SENTENCE], verbose=False)
        first.release_model()
        expected = first.get_sentence_vectors(SENTENCE)

        second = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=str(tmp_path))
        np.testing.assert_array_equal(second.get_sentence_vectors(SENTENCE), expected)
        assert first._model is None and second._model is None
        assert len(second._shards) == 1
        assert all(second._shards[name] is first._shards[name] for name in second._shards)

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
    def test_dataloader_workers_read_precomputed_contextual_vectors(self, tiny_bert, tmp_path):
        x = np.array([SENTENCE, SENTENCE[:3], SENTENCE[1:5], SENTENCE[2:]], dtype=object)
        y = np.array([["O"] * len(tokens) for tokens in x], dtype=object)
        preprocessor = Preprocessor()
        preprocessor.fit(x, y)
        config = ModelConfig(
            architecture="BidLSTM_CRF",
            embeddings_name="tiny-contextual",
            word_embedding_size=HIDDEN_SIZE,
            max_sequence_length=10,
        )
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        loader = create_dataloader(
            x,
            y,
            preprocessor=preprocessor,
            embeddings=embeddings,
            batch_size=1,
            shuffle=False,
            model_config=config,
            num_workers=2,
        )

        batches = list(loader)
        assert loader.num_workers == 2
        assert len(batches) == len(x)
        assert all(batch[0]["word_input"].abs().sum() > 0 for batch in batches)

    def test_nfold_training_with_contextual_embeddings(self, tiny_bert, tmp_path, monkeypatch):
        import delft.sequenceLabelling.wrapper as wrapper

        registry = _registry(tmp_path, _entry(tiny_bert, window=12, stride=5))
        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: registry)
        monkeypatch.chdir(tmp_path)
        x = np.array([SENTENCE, SENTENCE[:3], SENTENCE[1:5], SENTENCE[2:]], dtype=object)
        y = np.array([["B-ANIMAL" if token == "cat" else "O" for token in tokens] for tokens in x], dtype=object)
        model = wrapper.Sequence(
            "contextual-nfold-test",
            architecture="BidLSTM_CRF",
            embeddings_name="tiny-contextual",
            fold_number=2,
            max_epoch=1,
            batch_size=2,
            max_sequence_length=10,
            early_stop=False,
            nb_workers=0,
            device="cpu",
        )

        model.train_nfold(x, y)
        assert len(model.models) == 2
        assert model.embeddings.model.precompute(x, max_sequence_length=10, verbose=False) == 0

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

        model.save(str(tmp_path / "saved"))
        saved_settings = model.model_config.contextual_embedding_settings
        assert saved_settings == contextual.fingerprint()

        # A machine-specific local model path is not written to config.json: a
        # local registry entry is required to locate it when the model is loaded.
        empty_cache = tmp_path / "empty-cache"
        registry.clear()
        registry.update(_registry(empty_cache, _entry(tiny_bert)))
        registry["embeddings"] = []
        loaded = wrapper.Sequence("contextual-test", embeddings_name=None, nb_workers=0, device="cpu")
        with pytest.raises(ValueError, match="configure its path"):
            loaded.load(str(tmp_path / "saved"))

        # Vector settings still come from the saved model when the local entry
        # changes, while machine-local cache and batching choices follow it.
        import shutil

        local_cache = tmp_path / "local-cache"
        relocated_model = str(tmp_path / "relocated-model")
        shutil.copytree(tiny_bert, relocated_model)
        changed = _entry(
            relocated_model,
            layers=[-1],
            **{
                "layer-pooling": "sum",
                "subword-pooling": "last",
                "window": 32,
                "stride": 16,
                "cache-path": str(local_cache),
                "batch-size": 3,
            },
        )
        registry["embeddings"] = [changed]
        reloaded = wrapper.Sequence("contextual-test", embeddings_name=None, nb_workers=0, device="cpu")
        reloaded.load(str(tmp_path / "saved"))
        assert reloaded.embeddings.model.saved_settings() == saved_settings
        assert reloaded.embeddings.model.cache_path == str(local_cache)
        assert reloaded.embeddings.model.batch_size == 3

    def test_a_model_trained_with_the_path_of_a_local_transformer_loads_back(self, tiny_bert, tmp_path, monkeypatch):
        """
        ``contextual:<path>`` trained and saved a model that could not be loaded: the path,
        which the name of the embeddings holds, was only looked for in the registry.
        """
        import shutil

        import delft.sequenceLabelling.wrapper as wrapper

        registry = _registry(tmp_path, _entry(tiny_bert))
        registry["embeddings"] = []
        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: registry)
        monkeypatch.chdir(tmp_path)
        model_directory = str(tmp_path / "a-local-model")
        shutil.copytree(tiny_bert, model_directory)

        x = np.array([["the", "cat", "sat"], ["the", "dog", "ran", "far"]] * 6, dtype=object)
        y = np.array([["O", "B-ANIMAL", "O"], ["O", "B-ANIMAL", "O", "O"]] * 6, dtype=object)
        model = wrapper.Sequence(
            "contextual-path-test",
            architecture="BidLSTM_CRF",
            embeddings_name="contextual:" + model_directory,
            max_epoch=1,
            batch_size=4,
            early_stop=False,
            nb_workers=0,
            device="cpu",
        )
        model.train(x[:8], y[:8], x_valid=x[8:], y_valid=y[8:])
        expected = model.tag(["the cat sat"], "json")["texts"]
        model.save(str(tmp_path / "saved"))

        loaded = wrapper.Sequence("contextual-path-test", embeddings_name=None, nb_workers=0, device="cpu")
        loaded.load(str(tmp_path / "saved"))
        assert loaded.embeddings.model.model_reference == model_directory
        assert loaded.embeddings.model.saved_settings() == model.model_config.contextual_embedding_settings
        assert loaded.tag(["the cat sat"], "json")["texts"] == expected

        # another model in that directory is refused, and so is a directory that is gone
        _retrained(model_directory)
        with pytest.raises(ValueError, match="does not match the saved revision"):
            wrapper.Sequence("contextual-path-test", embeddings_name=None, nb_workers=0, device="cpu").load(
                str(tmp_path / "saved")
            )
        shutil.rmtree(model_directory)
        with pytest.raises(ValueError, match="there is no directory .*a-local-model here"):
            wrapper.Sequence("contextual-path-test", embeddings_name=None, nb_workers=0, device="cpu").load(
                str(tmp_path / "saved")
            )


class TestDataLoader:
    @staticmethod
    def _loader(embeddings, x, y, features=None, text_features_indices=None, max_sequence_length=None):
        from delft.sequenceLabelling.config import ModelConfig
        from delft.sequenceLabelling.data_loader import create_dataloader
        from delft.sequenceLabelling.preprocess import Preprocessor

        config = ModelConfig(architecture="BidLSTM_CRF", embeddings_name="tiny-contextual")
        config.max_sequence_length = max_sequence_length
        config.text_features_indices = text_features_indices
        preprocessor = Preprocessor()
        preprocessor.fit(x, y if y is not None else [["O"] * len(tokens) for tokens in x])
        return create_dataloader(
            x,
            y,
            preprocessor=preprocessor,
            embeddings=embeddings,
            features=features,
            model_config=config,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )

    def test_the_transformer_is_released_once_a_labelled_corpus_is_embedded(self, tiny_bert, tmp_path):
        """It stayed loaded, on the GPU, for the whole training of the RNN."""
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        x = [SENTENCE, LONG_SENTENCE[:9]]
        y = [["O"] * len(tokens) for tokens in x]
        released = []
        release = embeddings.model.release_model
        embeddings.model.release_model = lambda: (released.append(embeddings.model._model is not None), release())

        loader = self._loader(embeddings, x, y)
        assert released == [True] and embeddings.model._model is None
        batches = list(loader)
        assert sum(len(labels) for _, labels in batches) == 2
        assert embeddings.model._model is None, "the vectors of the corpus are read from the cache"

    def test_the_transformer_stays_loaded_for_the_texts_to_tag(self, tiny_bert, tmp_path):
        """Released, it would be loaded again at each call."""
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        self._loader(embeddings, [SENTENCE], None)
        assert embeddings.model._model is not None

    def test_several_tokens_per_position_are_embedded_as_one_sentence(self, tiny_bert, tmp_path):
        """
        Each token was embedded alone, without any context, and none of these vectors
        was computed before the training: the transformer was loaded by the dataset.
        """
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        lines = ["the cat", "sat ", "on the", " mat", "dog ran"]  # a line without a second token, one without a first
        words = ["the", "cat", "sat", "on", "the", "mat", "dog", "ran"]
        places = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (3, 1), (4, 0), (4, 1)]
        expected = embeddings.model.embed_batch([words])[0]

        x = to_vector_single(lines, embeddings, 6, tokens_per_position=2)
        assert x.shape == (6, 2 * HIDDEN_SIZE)
        filled = np.zeros((6, 2), dtype=bool)
        for vector, (position, column) in zip(expected, places):
            np.testing.assert_allclose(
                x[position, column * HIDDEN_SIZE : (column + 1) * HIDDEN_SIZE], vector, atol=1e-3
            )
            filled[position, column] = True
        for position, column in zip(*np.nonzero(~filled)):
            assert not x[position, column * HIDDEN_SIZE : (column + 1) * HIDDEN_SIZE].any()
        # in its sentence, not alone
        assert not np.allclose(x[0, :HIDDEN_SIZE], embeddings.get_word_vector("the"), atol=1e-3)

    def test_the_sentences_of_several_tokens_per_position_are_computed_before_the_training(self, tiny_bert, tmp_path):
        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        # the first two tokens of each line, as the columns of the features; a line has no second token
        features = [
            [["the", "cat"], ["sat", ""], ["on", "the"], ["mat", "dog"], ["ran", "far"]],
            [["far", "away"], ["now", "the"], ["cat", "sat"]],
        ]
        x = [[line[0] for line in lines] for lines in features]
        y = [["O"] * len(lines) for lines in x]
        loader = self._loader(embeddings, x, y, features=features, text_features_indices=[0, 1], max_sequence_length=4)
        assert embeddings.model._model is None
        inputs, _ = next(iter(loader))
        assert inputs["word_input"].shape[-1] == 2 * HIDDEN_SIZE
        assert embeddings.model._model is None, "the dataset found every sentence in the cache"
        # cut at 4 positions as the dataset cuts them, before the words are taken
        cached = embeddings.model
        assert cached.precompute([["the", "cat", "sat", "on", "the", "mat", "dog"]], verbose=False) == 0
        assert cached.precompute([["far", "away", "now", "the", "cat", "sat"]], verbose=False) == 0


class TestTextClassification:
    """
    The text classifiers took contextual embeddings word by word: every vector was the
    one of the word alone, computed by the dataset, in each of its worker processes.
    """

    TEXTS = ["the cat sat on the mat", "the dog ran far away now", "cat 42 sat", "dog ran"] * 3
    CLASSES = np.array([[1, 0], [0, 1], [1, 0], [0, 1]] * 3, dtype=np.float32)

    def test_a_text_is_embedded_as_a_whole(self, tiny_bert, tmp_path):
        from delft.textClassification.preprocess import to_vector_single as classification_vectors

        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        x = classification_vectors("the cat sat on the mat", embeddings, maxlen=8)
        assert x.shape == (8, HIDDEN_SIZE) and x.dtype == np.float32
        np.testing.assert_array_equal(x[:6], embeddings.get_sentence_vectors(["the", "cat", "sat", "on", "the", "mat"]))
        assert not x[6:].any()
        # the same word, at two places of the text
        assert not np.allclose(x[0], x[4])
        assert not np.allclose(x[0], embeddings.get_word_vector("the"), atol=1e-3)

    def test_the_tokens_are_given_as_they_are_written(self, tiny_bert, tmp_path):
        """The digits and the accents the classifiers clean out of a text are kept for the transformer."""
        from delft.textClassification.preprocess import tokens_to_embed

        assert tokens_to_embed("Le chat était là en 2020 !") == ["Le", "chat", "était", "là", "en", "2020", "!"]
        assert tokens_to_embed("the cat sat on the mat", maxlen=2) == ["the", "mat"]

    def test_classifier_trains_and_classifies_with_contextual_embeddings(self, tiny_bert, tmp_path, monkeypatch):
        import delft.textClassification.wrapper as wrapper

        registry = _registry(tmp_path, _entry(tiny_bert))
        monkeypatch.setattr(wrapper, "load_resource_registry", lambda path: registry)
        monkeypatch.chdir(tmp_path)

        classifier = wrapper.Classifier(
            "contextual-classifier",
            architecture="gru",
            embeddings_name="tiny-contextual",
            list_classes=["a", "b"],
            maxlen=8,
            max_epoch=1,
            batch_size=4,
            early_stop=False,
            nb_workers=0,
            device="cpu",
        )
        assert classifier.model_config.word_embedding_size == HIDDEN_SIZE
        contextual = classifier.embeddings.model

        classifier.train(self.TEXTS, self.CLASSES)
        assert classifier.preprocessor is None
        assert contextual._model is None, "the transformer is freed once the texts are embedded"
        # the texts of the training are in the cache, as the sentences they are
        sentences = [text.split() for text in set(self.TEXTS)]
        assert contextual.precompute(sentences, verbose=False) == 0

        scores = classifier.predict(["the cat sat", "dog ran far"], output_format="array")
        assert scores.shape == (2, 2)
        # the vectors of the texts to classify stay in memory, and the transformer loaded
        assert contextual.sentence_key(["the", "cat", "sat"]) in contextual._memory
        assert contextual._model is not None
        other = ContextualEmbeddings(tiny_bert, device="cpu", cache_path=contextual.cache_path)
        assert other.precompute([["the", "cat", "sat"]], verbose=False) == 1

    def test_the_dataset_reads_the_cache(self, tiny_bert, tmp_path):
        from delft.textClassification.config import ModelConfig
        from delft.textClassification.data_loader import create_dataloader

        embeddings = Embeddings("tiny-contextual", resource_registry=_registry(tmp_path, _entry(tiny_bert)))
        config = ModelConfig(architecture="gru", embeddings_name="tiny-contextual", list_classes=["a", "b"], maxlen=4)
        loader = create_dataloader(
            self.TEXTS,
            self.CLASSES,
            config,
            embeddings=embeddings,
            batch_size=4,
            shuffle=False,
            num_workers=2,
        )
        assert loader.num_workers > 0
        assert embeddings.model._model is None
        inputs, labels = next(iter(loader))
        assert inputs.shape == (4, 4, HIDDEN_SIZE) and labels.shape == (4, 2)
        assert embeddings.model._model is None, "the dataset found every text in the cache"
        # cut at the last 4 tokens, as the dataset cuts a text
        np.testing.assert_array_equal(inputs[0].numpy(), embeddings.get_sentence_vectors(["sat", "on", "the", "mat"]))

        prediction_loader = create_dataloader(
            self.TEXTS,
            None,
            config,
            embeddings=embeddings,
            batch_size=4,
            shuffle=False,
            num_workers=2,
        )
        assert prediction_loader.num_workers > 0
        assert sum(len(batch) for batch in prediction_loader) == len(self.TEXTS)
