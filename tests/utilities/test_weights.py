import os

import pytest
import torch

from delft.utilities.weights import SAFETENSORS_WEIGHT_FILE_NAME, find_weight_file, load_weights, save_weights


class TiedModel(torch.nn.Module):
    """Two parameters sharing one tensor, as the tied embeddings of a transformer do."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.output = torch.nn.Linear(4, 7, bias=False)
        self.output.weight = self.embedding.weight


def _assert_same_weights(model, other):
    state, other_state = model.state_dict(), other.state_dict()
    assert state.keys() == other_state.keys()
    for name in state:
        assert torch.equal(state[name], other_state[name]), name


@pytest.mark.parametrize("weight_file", ["model_weights.pt", SAFETENSORS_WEIGHT_FILE_NAME])
class TestSaveAndLoad:
    def test_round_trip(self, tmp_path, weight_file):
        model, other = torch.nn.LSTM(3, 5, bidirectional=True), torch.nn.LSTM(3, 5, bidirectional=True)
        save_weights(model, tmp_path / weight_file)
        load_weights(other, tmp_path / weight_file, device=torch.device("cpu"))
        _assert_same_weights(model, other)

    def test_round_trip_with_shared_tensors(self, tmp_path, weight_file):
        model, other = TiedModel(), TiedModel()
        save_weights(model, tmp_path / weight_file)
        load_weights(other, tmp_path / weight_file)
        _assert_same_weights(model, other)
        assert other.output.weight is other.embedding.weight


def test_safetensors_file_holds_no_pickle(tmp_path):
    from safetensors import safe_open

    save_weights(torch.nn.Linear(2, 3), tmp_path / SAFETENSORS_WEIGHT_FILE_NAME)
    with safe_open(str(tmp_path / SAFETENSORS_WEIGHT_FILE_NAME), framework="pt") as weights:
        assert sorted(weights.keys()) == ["bias", "weight"]


class TestFindWeightFile:
    @staticmethod
    def _touch(directory, name):
        (directory / name).write_bytes(b"")
        return str(directory / name)

    def test_requested_file_wins_when_both_formats_are_there(self, tmp_path):
        self._touch(tmp_path, SAFETENSORS_WEIGHT_FILE_NAME)
        requested = self._touch(tmp_path, "model_weights.pt")
        assert find_weight_file(str(tmp_path), "model_weights.pt") == requested

    def test_falls_back_to_safetensors(self, tmp_path):
        published = self._touch(tmp_path, SAFETENSORS_WEIGHT_FILE_NAME)
        assert find_weight_file(str(tmp_path), "model_weights.pt") == published
        assert find_weight_file(str(tmp_path), "model_weights.pth") == published

    @pytest.mark.parametrize("pickled", ["model_weights.pt", "model_weights.pth"])
    def test_falls_back_to_pickled_weights(self, tmp_path, pickled):
        saved = self._touch(tmp_path, pickled)
        assert find_weight_file(str(tmp_path), SAFETENSORS_WEIGHT_FILE_NAME) == saved

    def test_names_the_requested_file_when_there_are_no_weights(self, tmp_path):
        assert find_weight_file(str(tmp_path), "model_weights.pt") == os.path.join(str(tmp_path), "model_weights.pt")
        missing = str(tmp_path / "no-such-model")
        assert find_weight_file(missing, "model_weights.pt") == os.path.join(missing, "model_weights.pt")
