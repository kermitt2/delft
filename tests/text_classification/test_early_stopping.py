"""
Early stopping of the text classifiers: the epochs whose ROC-AUC is 0 do not count,
as the epochs with an f1 of 0 do not in sequence labelling, and a validation loss of
0 is a score. Every architecture goes through the same trainer.
"""

from unittest.mock import patch

import numpy as np
import pytest

from delft.textClassification.models import MODEL_REGISTRY
from delft.textClassification.trainer import Trainer
from delft.textClassification.wrapper import Classifier

TEXTS = [
    "good work",
    "bad work",
    "good good work",
    "bad bad work",
    "work good",
    "work bad",
    "good",
    "bad",
]
CLASSES = np.array([[1, 0], [0, 1]] * 4, dtype=np.float32)

PATIENCE = 2
ROC_AUCS = [0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6]
EPOCHS_EXPECTED = 7  # the patience runs from epoch 5, the first score


class _Embeddings:
    """Word vectors that need no download: a word always gets the same random vector."""

    embed_size = 300

    def get_word_vector(self, word):
        return np.random.RandomState(hash(word) % (2**32)).randn(self.embed_size).astype("float32")


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "work"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


def _train(tmp_path, monkeypatch, architecture, scores, use_roc_auc=True):
    """Train ``architecture`` with the validation metrics of every epoch scripted; the epochs run."""
    monkeypatch.chdir(tmp_path)
    scripted = iter(scores)
    epochs = []
    options = dict(
        list_classes=["a", "b"],
        maxlen=100,  # dpcnn pools a shorter sequence to nothing
        max_epoch=len(scores),
        patience=PATIENCE,
        early_stop=True,
        use_roc_auc=use_roc_auc,
        batch_size=4,
        nb_workers=0,
        device="cpu",
        embeddings_name=None,
    )
    if architecture == "bert":
        options.update(transformer_name="in-memory", maxlen=8)
    classifier = Classifier("test-classifier", architecture=architecture, **options)
    classifier.embeddings = _Embeddings()

    def evaluate(self, data_loader):
        score = next(scripted)
        epochs.append(score)
        return {"loss": score, "roc_auc": score}

    with (
        patch.object(Trainer, "evaluate", evaluate),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer),
    ):
        classifier.train(TEXTS, CLASSES)
    return epochs


@pytest.mark.parametrize("architecture", sorted(MODEL_REGISTRY))
def test_a_classifier_with_a_roc_auc_of_zero_is_not_stopped_for_it(tmp_path, monkeypatch, architecture):
    epochs = _train(tmp_path, monkeypatch, architecture, ROC_AUCS)
    assert len(epochs) == EPOCHS_EXPECTED, f"stopped after {len(epochs)} epochs"


def test_a_classifier_whose_roc_auc_stops_improving_is_still_stopped(tmp_path, monkeypatch):
    epochs = _train(tmp_path, monkeypatch, "gru", [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
    assert len(epochs) == PATIENCE + 1


def test_on_the_validation_loss_a_zero_is_a_score(tmp_path, monkeypatch):
    """The loss is minimised: a 0 is the best possible score, and the patience counts."""
    epochs = _train(tmp_path, monkeypatch, "gru", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], use_roc_auc=False)
    assert len(epochs) == PATIENCE + 1


def test_every_architecture_is_covered():
    assert len(MODEL_REGISTRY) == 11
