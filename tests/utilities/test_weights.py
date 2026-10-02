import os

import pytest
import torch

from delft.utilities.weights import (
    SAFETENSORS_WEIGHT_FILE_NAME,
    find_fold_weight_file,
    find_weight_file,
    fold_weight_file,
    load_weights,
    save_weights,
)


class TiedModel(torch.nn.Module):
    """Two parameters sharing one tensor, as the tied embeddings of a transformer do."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.output = torch.nn.Linear(4, 7, bias=False)
        self.output.weight = self.embedding.weight


def _flatten_like_cudnn(lstm):
    """
    Make every weight of ``lstm`` a view into one buffer, as cuDNN's flatten_parameters
    does on a GPU: the tensors share a storage that none of them covers whole.
    """
    params = list(lstm.parameters())
    buffer = torch.cat([p.detach().flatten() for p in params])
    offset = 0
    for p in params:
        p.data = buffer[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
    assert params[0].untyped_storage().data_ptr() == params[1].untyped_storage().data_ptr()


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

    def test_round_trip_of_an_lstm_flattened_by_cudnn(self, tmp_path, weight_file):
        """Saving as safetensors failed on every model trained on a GPU, after the training."""
        model, other = torch.nn.LSTM(3, 5, bidirectional=True), torch.nn.LSTM(3, 5, bidirectional=True)
        _flatten_like_cudnn(model)
        save_weights(model, tmp_path / weight_file)
        load_weights(other, tmp_path / weight_file, device=torch.device("cpu"))
        _assert_same_weights(model, other)


def test_a_file_that_holds_a_tied_parameter_once_loads(tmp_path):
    """What safetensors' save_model wrote, up to now, for the tied parameters."""
    from safetensors.torch import save_model

    model, other = TiedModel(), TiedModel()
    save_model(model, str(tmp_path / SAFETENSORS_WEIGHT_FILE_NAME))
    load_weights(other, tmp_path / SAFETENSORS_WEIGHT_FILE_NAME)
    _assert_same_weights(model, other)


def test_a_file_with_weights_missing_or_unexpected_is_refused(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / SAFETENSORS_WEIGHT_FILE_NAME
    save_file({"bias": torch.zeros(3)}, str(path))
    with pytest.raises(RuntimeError, match="Missing.*weight"):
        load_weights(torch.nn.Linear(2, 3), path)
    save_file({"bias": torch.zeros(3), "weight": torch.zeros(3, 2), "extra": torch.zeros(1)}, str(path))
    with pytest.raises(RuntimeError, match="Unexpected.*extra"):
        load_weights(torch.nn.Linear(2, 3), path)


@pytest.mark.parametrize("weight_file", ["model_weights.pt", SAFETENSORS_WEIGHT_FILE_NAME])
def test_round_trip_of_weights_that_are_not_contiguous(tmp_path, weight_file):
    """
    As the weights of SciBERT, converted from TensorFlow and transposed: safetensors
    refuses them, and a fine-tuned SciBERT could not be saved after its training.
    """
    model, other = torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)
    model.weight = torch.nn.Parameter(torch.randn(2, 3).t())
    assert not model.weight.is_contiguous()
    save_weights(model, tmp_path / weight_file)
    load_weights(other, tmp_path / weight_file)
    _assert_same_weights(model, other)


class WeightedLossModel(torch.nn.Module):
    """A model whose loss holds a tensor, as a text classifier trained with class weights."""

    def __init__(self, class_weights=None):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights)


@pytest.mark.parametrize("weight_file", ["model_weights.pth", SAFETENSORS_WEIGHT_FILE_NAME])
class TestLossTensors:
    def test_a_model_trained_with_class_weights_loads_back(self, tmp_path, weight_file):
        """Its loss weights were saved with it, and the model built to load them has none: an error."""
        model, other = WeightedLossModel(torch.tensor([1.0, 2.0, 3.0])), WeightedLossModel()
        assert "loss_fn.weight" in model.state_dict()
        save_weights(model, tmp_path / weight_file)
        load_weights(other, tmp_path / weight_file)
        assert torch.equal(other.linear.weight, model.linear.weight)

    def test_the_loss_of_the_model_keeps_its_own_weights(self, tmp_path, weight_file):
        """As when training goes on from a saved model, with the class weights of the new training."""
        save_weights(WeightedLossModel(torch.tensor([1.0, 2.0, 3.0])), tmp_path / weight_file)
        other = WeightedLossModel(torch.tensor([5.0, 5.0, 5.0]))
        load_weights(other, tmp_path / weight_file)
        assert other.loss_fn.weight.tolist() == [5.0, 5.0, 5.0]

    def test_a_file_saved_with_the_loss_weights_still_loads(self, tmp_path, weight_file):
        """The models saved so far hold them."""
        model, other = WeightedLossModel(torch.tensor([1.0, 2.0, 3.0])), WeightedLossModel()
        state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        assert "loss_fn.weight" in state
        if weight_file == SAFETENSORS_WEIGHT_FILE_NAME:
            from safetensors.torch import save_file

            save_file(state, str(tmp_path / weight_file))
        else:
            torch.save(state, tmp_path / weight_file)
        load_weights(other, tmp_path / weight_file)
        assert torch.equal(other.linear.bias, model.linear.bias)


def test_the_loss_weights_are_not_written(tmp_path):
    from safetensors import safe_open

    save_weights(WeightedLossModel(torch.tensor([1.0, 2.0, 3.0])), tmp_path / SAFETENSORS_WEIGHT_FILE_NAME)
    with safe_open(str(tmp_path / SAFETENSORS_WEIGHT_FILE_NAME), framework="pt") as weights:
        assert sorted(weights.keys()) == ["linear.bias", "linear.weight"]


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


class TestFindFoldWeightFile:
    """The weights of the model of a fold, of a text classifier trained over several folds."""

    @staticmethod
    def _touch(directory, name):
        (directory / name).write_bytes(b"")
        return str(directory / name)

    def test_name_of_the_weights_of_a_fold(self):
        assert fold_weight_file("model.safetensors", 0) == "model_fold0.safetensors"
        assert fold_weight_file("model_weights.pt", 12) == "model_weights_fold12.pt"

    def test_safetensors_win_when_both_formats_are_there(self, tmp_path):
        self._touch(tmp_path, "model_weights_fold1.pt")
        published = self._touch(tmp_path, "model_fold1.safetensors")
        assert find_fold_weight_file(str(tmp_path), 1) == published

    @pytest.mark.parametrize("pickled", ["model_weights.pt", "model_weights.pth", "weights.pt"])
    def test_falls_back_to_pickled_weights_whatever_their_name(self, tmp_path, pickled):
        """model_weights.pt is a name the wrappers save under: only the .pth one was looked for."""
        saved = [self._touch(tmp_path, fold_weight_file(pickled, fold_id)) for fold_id in range(11)]
        assert [find_fold_weight_file(str(tmp_path), fold_id) for fold_id in (0, 1, 10)] == [
            saved[0],
            saved[1],
            saved[10],
        ]

    def test_names_the_safetensors_file_when_the_fold_has_no_weights(self, tmp_path):
        self._touch(tmp_path, "model_fold0.safetensors")
        self._touch(tmp_path, "model.safetensors")  # the weights of a single model are not those of a fold
        assert find_fold_weight_file(str(tmp_path), 1) == os.path.join(str(tmp_path), "model_fold1.safetensors")
        missing = str(tmp_path / "no-such-model")
        assert find_fold_weight_file(missing, 0) == os.path.join(missing, "model_fold0.safetensors")
