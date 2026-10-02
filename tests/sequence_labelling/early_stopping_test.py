"""
Early stopping does not count the epochs whose validation f1 is 0, for every
architecture: a model that has not predicted an entity yet is not stopped for it.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from delft.sequenceLabelling.models import MODEL_REGISTRY
from delft.sequenceLabelling.trainer import ModelCheckpoint, Trainer
from delft.sequenceLabelling.wrapper import Sequence

WORDS = ["Jim", "Henson", "was", "a", "puppeteer", "in", "Mississippi", "today"]
LABELS = ["B-per", "I-per", "O", "O", "O", "O", "B-loc", "O"]
X = [WORDS, WORDS[:3], WORDS[2:], WORDS[:5]]
Y = [LABELS, LABELS[:3], LABELS[2:], LABELS[:5]]
FEATURES = [[[word, "UP" if word[0].isupper() else "LOW"] for word in sequence] for sequence in X]
X_ARRAY, Y_ARRAY, FEATURES_ARRAY = (np.array(item, dtype=object) for item in (X, Y, FEATURES))

PATIENCE = 2
# the f1 of every epoch: no entity predicted for four epochs, then a score that does
# not improve. With the zeros counted, the training stopped at epoch 3; without them
# the patience runs from epoch 5, the first score, and it stops at epoch 7.
SCORES = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
EPOCHS_WITH_THE_ZEROS_COUNTED = PATIENCE + 1
EPOCHS_EXPECTED = 7


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


def _tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + WORDS
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


def _train(tmp_path, monkeypatch, architecture, scores):
    """Train ``architecture`` with the validation f1 of every epoch scripted; the epochs run."""
    monkeypatch.chdir(tmp_path)
    scripted = iter(scores)
    epochs = []
    options = dict(
        max_epoch=len(scores),
        patience=PATIENCE,
        early_stop=True,
        batch_size=2,
        nb_workers=0,
        device="cpu",
        embeddings_name=None,
    )
    if architecture.startswith("BERT"):
        options.update(transformer_name="in-memory", max_sequence_length=16)
    sequence = Sequence("test-model", architecture=architecture, **options)
    features = FEATURES_ARRAY if architecture.endswith("FEATURES") else None

    def evaluate(self, data_loader):
        f1 = next(scripted)
        return {"f1": f1, "precision": f1, "recall": f1, "loss": 1.0}

    with (
        patch.object(Trainer, "evaluate", evaluate),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
    ):
        sequence.train(
            X_ARRAY,
            Y_ARRAY,
            f_train=features,
            x_valid=X_ARRAY,
            y_valid=Y_ARRAY,
            f_valid=features,
            callbacks=[lambda epoch, logs: epochs.append(epoch)],
        )
    return epochs


@pytest.mark.parametrize("architecture", sorted(MODEL_REGISTRY))
def test_a_model_with_no_entity_predicted_yet_is_not_stopped(tmp_path, monkeypatch, architecture):
    epochs = _train(tmp_path, monkeypatch, architecture, SCORES)
    assert len(epochs) == EPOCHS_EXPECTED, f"stopped after {len(epochs)} epochs"
    assert len(epochs) > EPOCHS_WITH_THE_ZEROS_COUNTED


def test_a_model_whose_score_stops_improving_is_still_stopped(tmp_path, monkeypatch):
    epochs = _train(tmp_path, monkeypatch, "BidLSTM_CRF", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert len(epochs) == PATIENCE + 1


def test_every_architecture_is_covered():
    assert len(MODEL_REGISTRY) == 15


class TestCheckpoint:
    """
    The weights kept as the best ones follow the same rule: those of the first epoch were
    kept as long as the score stayed at 0, and put back at the end of a training that
    early stopping had let go on.
    """

    @staticmethod
    def _kept(tmp_path, scores, mode="max"):
        """The epoch whose weights are kept after epochs with these scores, counted from 1."""
        model = torch.nn.Linear(1, 1)
        checkpoint = ModelCheckpoint(str(tmp_path / "best.pt"), mode=mode)
        for epoch, score in enumerate(scores, start=1):
            torch.nn.init.constant_(model.weight, float(epoch))
            checkpoint(model, score)
        return int(torch.load(tmp_path / "best.pt")["weight"].item())

    def test_the_latest_weights_are_kept_while_there_is_no_score(self, tmp_path):
        assert self._kept(tmp_path, [0.0, 0.0, 0.0]) == 3

    def test_the_best_weights_are_kept_once_there_is_a_score(self, tmp_path):
        assert self._kept(tmp_path, [0.0, 0.0, 0.5, 0.4, 0.0]) == 3
        assert self._kept(tmp_path, [0.5, 0.0, 0.0]) == 1
        assert self._kept(tmp_path, [0.2, 0.2, 0.3, 0.3]) == 3

    def test_a_loss_is_not_concerned(self, tmp_path):
        assert self._kept(tmp_path, [0.0, 0.0, 0.0], mode="min") == 1


def test_a_training_that_never_scores_ends_with_its_latest_weights(tmp_path, monkeypatch):
    """Its weights were those of its first epoch, whatever the number of epochs it ran."""
    saved = []
    save = ModelCheckpoint._save

    def spy(self, model):
        saved.append(len(saved) + 1)
        save(self, model)

    monkeypatch.setattr(ModelCheckpoint, "_save", spy)
    epochs = _train(tmp_path, monkeypatch, "BidLSTM_CRF", [0.0] * 5)
    assert len(epochs) == 5 and len(saved) == 5
