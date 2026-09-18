"""Training sequences longer than the model takes are cut into windows, not truncated."""

from unittest.mock import patch

import numpy as np
import pytest

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.windows import (
    join_scored_windows,
    join_windows,
    split_into_windows,
    subtoken_costs,
    window_bounds,
)
from delft.sequenceLabelling.wrapper import Sequence

WORDS = ["Jim", "Hensonization", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
# [CLS] Jim He ##nson ##ization was a puppet ##eer in Mississippi today [SEP]
SUB_TOKENS_PER_WORD = [1, 3, 1, 1, 2, 1, 1, 1]


class TestWindowBounds:
    def test_a_sequence_that_fits_is_one_window(self):
        assert window_bounds([1] * 5, max_length=5, stride=2) == [(0, 5)]

    def test_an_empty_sequence_is_one_empty_window(self):
        assert window_bounds([], max_length=5, stride=2) == [(0, 0)]

    def test_a_stride_of_the_window_length_gives_windows_side_by_side(self):
        assert window_bounds([1] * 8, max_length=3, stride=3) == [(0, 3), (3, 6), (6, 8)]

    def test_a_smaller_stride_makes_the_windows_overlap(self):
        assert window_bounds([1] * 8, max_length=5, stride=3) == [(0, 5), (3, 8)]

    def test_lengths_are_counted_with_the_cost_of_each_token(self):
        # 1+3+1 = 5 sub-tokens in the first window; its first two words make the stride of 4
        assert window_bounds(SUB_TOKENS_PER_WORD, max_length=5, stride=4) == [(0, 3), (2, 6), (5, 8)]

    def test_a_token_longer_than_a_window_gets_its_own(self):
        assert window_bounds([1, 9, 1], max_length=3, stride=3) == [(0, 1), (1, 2), (2, 3)]

    @pytest.mark.parametrize("nb_tokens", [1, 2, 7, 50, 101])
    @pytest.mark.parametrize("max_length, stride", [(1, 1), (4, 1), (4, 3), (4, 4), (10, 5), (60, 17)])
    def test_windows_fit_advance_and_leave_no_gap(self, nb_tokens, max_length, stride):
        costs = [1 + (i % 3) // 2 for i in range(nb_tokens)]
        bounds = window_bounds(costs, max_length, stride)
        assert bounds[0][0] == 0 and bounds[-1][1] == nb_tokens
        for start, end in bounds:
            assert end > start
            assert sum(costs[start:end]) <= max_length or end == start + 1
        for (start, end), (next_start, next_end) in zip(bounds, bounds[1:]):
            assert start < next_start <= end < next_end

    @pytest.mark.parametrize("stride", [0, -1, 6])
    def test_rejects_a_stride_that_would_skip_tokens_or_not_advance(self, stride):
        with pytest.raises(ValueError, match="stride"):
            window_bounds([1] * 8, max_length=5, stride=stride)


class TestSplitIntoWindows:
    def test_labels_and_features_are_cut_with_their_tokens(self):
        features = [[f"f{i}"] for i in range(len(WORDS))]
        x, y, f, counts = split_into_windows([WORDS], [LABELS], [features], max_length=5, stride=3)
        assert counts == [2]
        assert x == [WORDS[0:5], WORDS[3:8]]
        assert y == [LABELS[0:5], LABELS[3:8]]
        assert f == [features[0:5], features[3:8]]

    def test_sequences_that_fit_are_left_alone(self):
        x, y, f, counts = split_into_windows(
            [WORDS[:2], WORDS, WORDS[:1]], [LABELS[:2], LABELS, LABELS[:1]], None, 5, 5
        )
        assert counts == [1, 2, 1]
        assert x == [WORDS[:2], WORDS[:5], WORDS[5:], WORDS[:1]]
        assert y == [LABELS[:2], LABELS[:5], LABELS[5:], LABELS[:1]]
        assert f is None

    def test_works_without_labels(self):
        x, y, f, _ = split_into_windows([WORDS], None, None, max_length=5, stride=5)
        assert x == [WORDS[:5], WORDS[5:]] and y is None and f is None

    def test_sub_token_costs(self, wordpiece_tokenizer):
        assert subtoken_costs(wordpiece_tokenizer)(WORDS) == SUB_TOKENS_PER_WORD
        assert subtoken_costs(wordpiece_tokenizer)([]) == []


class TestJoinWindows:
    @pytest.mark.parametrize("max_length", [1, 3, 5, 8, 20])
    def test_puts_back_together_windows_cut_side_by_side(self, max_length):
        sequences = [WORDS, WORDS[:3], [], WORDS[:5]]
        windows, _, _, counts = split_into_windows(sequences, None, None, max_length, stride=max_length)
        assert join_windows(windows, counts) == sequences

    def test_rejects_windows_that_do_not_match_the_counts(self):
        with pytest.raises(ValueError, match="3 windows"):
            join_windows([["a"], ["b"], ["c"]], [1, 1])

    def test_scores_are_left_alone_without_windows(self):
        loader, _ = _loader(5, window_stride=None)
        assert join_scored_windows(loader, [[1], [2]], [[3], [4]]) == ([[1], [2]], [[3], [4]])

    def test_scores_are_joined_per_sequence(self):
        loader, _ = _loader(5, window_stride=5)  # WORDS makes two windows, WORDS[:3] one
        predictions, labels = join_scored_windows(loader, [[1, 1], [2], [3]], [[4, 4], [5], [6]])
        assert predictions == [[1, 1, 2], [3]] and labels == [[4, 4, 5], [6]]

    def test_scores_are_left_alone_when_only_a_part_of_the_windows_was_seen(self):
        loader, _ = _loader(5, window_stride=5)
        assert join_scored_windows(loader, [[1, 1], [2]], [[4, 4], [5]]) == ([[1, 1], [2]], [[4, 4], [5]])


def _labels_seen(loader, preprocessor):
    """The labels of every example of a loader, padding left out."""
    index_to_tag = {index: tag for tag, index in preprocessor.vocab_tag.items()}
    seen = []
    for inputs, labels in loader:
        for row in labels.tolist():
            seen.append([index_to_tag[index] for index in row if index_to_tag[index] != "<PAD>"])
    return seen


def _loader(max_sequence_length, window_stride, transformer_name=None):
    preprocessor = Preprocessor(return_chars=transformer_name is None)
    preprocessor.fit([WORDS], [LABELS])
    model_config = ModelConfig(
        architecture="BERT" if transformer_name else "BidLSTM_CRF",
        embeddings_name=None,
        transformer_name=transformer_name,
        max_sequence_length=max_sequence_length,
    )
    loader = create_dataloader(
        [WORDS, WORDS[:3]],
        [LABELS, LABELS[:3]],
        batch_size=2,
        preprocessor=preprocessor,
        shuffle=False,
        model_config=model_config,
        window_stride=window_stride,
    )
    return loader, preprocessor


class TestTrainingLoader:
    def test_without_a_stride_what_follows_the_cut_is_lost(self):
        assert _labels_seen(*_loader(5, window_stride=None)) == [LABELS[:5], LABELS[:3]]

    def test_with_a_stride_the_whole_sequence_is_trained_on(self):
        assert _labels_seen(*_loader(5, window_stride=3)) == [LABELS[0:5], LABELS[3:8], LABELS[:3]]

    def test_rejects_a_stride_longer_than_the_sequence_length(self):
        with pytest.raises(ValueError, match="window_stride"):
            _loader(5, window_stride=6)

    def test_with_a_transformer_the_windows_are_measured_in_sub_tokens(self, wordpiece_tokenizer):
        # 7 sub-tokens, of which [CLS] and [SEP]: 5 are left for the words
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=wordpiece_tokenizer):
            loader, preprocessor = _loader(7, window_stride=4, transformer_name="in-memory")
            seen = _labels_seen(loader, preprocessor)
            for inputs, _ in loader:
                assert inputs["input_ids"].shape[1] <= 7
        assert seen == [LABELS[0:3], LABELS[2:6], LABELS[5:8], LABELS[:3]]

    def test_with_a_transformer_and_no_stride_the_cut_falls_in_the_sub_tokens(self, wordpiece_tokenizer):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=wordpiece_tokenizer):
            assert _labels_seen(*_loader(7, window_stride=None, transformer_name="in-memory"))[0] == LABELS[0:3]


def _windowed_sequence(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    return Sequence(
        "test-model",
        architecture="BidLSTM_CRF",
        embeddings_name=None,
        max_sequence_length=5,
        max_epoch=1,
        batch_size=2,
        early_stop=False,
        nb_workers=0,
        device="cpu",
        **kwargs,
    )


X = np.array([WORDS, WORDS[:3]], dtype=object)
Y = np.array([LABELS, LABELS[:3]], dtype=object)


def test_sequence_trains_on_the_windows_and_validates_on_whole_sequences(tmp_path, monkeypatch, capsys):
    sequence = _windowed_sequence(tmp_path, monkeypatch, window_stride=3)
    sequence.train(X, Y, x_valid=X, y_valid=Y)
    output = capsys.readouterr().out
    assert "[train] window stride 3: 2 sequences make 3 windows of at most 5" in output
    # side by side for the validation set, whatever the stride of the training set
    assert "[valid] window stride 5: 2 sequences make 3 windows of at most 5" in output


def _scored_labels(sequence):
    """The expected labels that eval() hands to the scoring, per sequence."""
    with patch("delft.sequenceLabelling.wrapper.classification_report", return_value=("", {})) as report:
        sequence.eval(X, Y)
    return report.call_args[0][0]


def test_evaluation_scores_whole_sequences(tmp_path, monkeypatch):
    sequence = _windowed_sequence(tmp_path, monkeypatch, window_stride=3)
    sequence.train(X, Y)
    assert _scored_labels(sequence) == [LABELS, LABELS[:3]]


def test_evaluation_truncates_without_a_stride(tmp_path, monkeypatch):
    sequence = _windowed_sequence(tmp_path, monkeypatch)
    sequence.train(X, Y)
    assert _scored_labels(sequence) == [LABELS[:5], LABELS[:3]]


def test_a_saved_model_is_evaluated_the_way_it_was_trained(tmp_path, monkeypatch):
    sequence = _windowed_sequence(tmp_path, monkeypatch, window_stride=3)
    sequence.train(X, Y)
    sequence.save(str(tmp_path))

    loaded = Sequence("test-model", nb_workers=0, device="cpu")
    loaded.load(str(tmp_path))
    assert loaded.model_config.window_stride == 3
    assert _scored_labels(loaded) == [LABELS, LABELS[:3]]
