"""Incremental training goes on from a loaded model, and n-fold models save and load back as single models."""

import numpy as np
import pytest

from delft.sequenceLabelling.wrapper import Sequence

X = np.array([["March", "2019"], ["June", "2020"]] * 10, dtype=object)
Y = np.array([["B-<month>", "B-<year>"]] * 20, dtype=object)


def _trained(tmp_path, **kwargs):
    model = Sequence(
        "incremental-test",
        architecture="BidLSTM_CRF",
        embeddings_name=None,
        max_epoch=1,
        batch_size=4,
        device="cpu",
        nb_workers=0,
        **kwargs,
    )
    if model.model_config.fold_number > 1:
        model.train_nfold(X, Y)
        model.eval(X, Y)
    else:
        model.train(X, Y)
    model.save(str(tmp_path))
    return model


def _loaded(tmp_path, **kwargs):
    model = Sequence("incremental-test", max_epoch=1, batch_size=4, device="cpu", nb_workers=0, **kwargs)
    model.load(str(tmp_path))
    return model


def test_characters_the_model_does_not_know_are_added_to_it(tmp_path):
    _trained(tmp_path)
    model = _loaded(tmp_path)
    known = len(model.p.vocab_char)

    model.train(np.array([["Ζεύς", "2019"]] * 20, dtype=object), Y, incremental=True)

    assert len(model.p.vocab_char) > known
    assert model.model.char_encoder.char_embeddings.num_embeddings == len(model.p.vocab_char)
    assert model.tag([["Ζεύς", "2019"]], None)[0][1] == ("2019", "B-<year>")

    model.save(str(tmp_path))
    assert len(_loaded(tmp_path).p.vocab_char) == len(model.p.vocab_char)


def test_a_label_the_model_does_not_know_is_refused(tmp_path):
    _trained(tmp_path)
    model = _loaded(tmp_path)
    with pytest.raises(ValueError, match="B-<day>"):
        model.train(X, np.array([["B-<month>", "B-<day>"]] * 20, dtype=object), incremental=True)


def test_each_fold_goes_on_from_the_loaded_model(tmp_path):
    _trained(tmp_path)
    model = _loaded(tmp_path, fold_number=2)

    model.train_nfold(X, Y, incremental=True)

    assert len(model.models) == 2
    assert all(fold_model is not model.model for fold_model in model.models)
    model.eval(X, Y)


def test_the_model_of_an_n_fold_training_loads_as_a_single_model(tmp_path):
    _trained(tmp_path, fold_number=2)
    model = _loaded(tmp_path)

    assert model.model_config.fold_number == 1
    model.eval(X, Y)


def test_n_fold_training_needs_two_folds_at_least(tmp_path):
    model = Sequence("incremental-test", architecture="BidLSTM_CRF", embeddings_name=None, device="cpu")
    with pytest.raises(ValueError, match="at least 2 folds"):
        model.train_nfold(X, Y)


@pytest.mark.parametrize("fold_number", [1, 2])
def test_without_a_model_nothing_is_saved(tmp_path, fold_number):
    model = Sequence("incremental-test", architecture="BidLSTM_CRF", embeddings_name=None, fold_number=fold_number)
    with pytest.raises(ValueError, match="No model to save"):
        model.save(str(tmp_path))
    assert not (tmp_path / "incremental-test").exists()
