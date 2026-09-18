"""Transformer inputs of the text classifiers, with a tokenizer built in memory."""

from unittest.mock import patch

import pytest
import torch

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.data_loader import TextClassificationDataset
from delft.textClassification.preprocess import create_batch_input_bert, create_single_input_bert


def _tokenizer(with_segment_ids):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "work"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    names = ["input_ids", "token_type_ids", "attention_mask"] if with_segment_ids else ["input_ids", "attention_mask"]
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_input_names=names,
    )


@pytest.mark.parametrize("with_segment_ids", [True, False], ids=["BERT-like", "RoBERTa-like"])
class TestTokenization:
    """encode_plus and batch_encode_plus are gone from transformers 5: every transformer classifier failed."""

    def test_dataset_item(self, with_segment_ids):
        model_config = ModelConfig(
            model_name="test", architecture="bert", list_classes=["a", "b"], transformer_name="in-memory", maxlen=6
        )
        dataset = TextClassificationDataset(
            ["good work"], [[1, 0]], model_config, transformer_tokenizer=_tokenizer(with_segment_ids)
        )
        inputs, labels = dataset[0]
        assert inputs["input_ids"].tolist() == [2, 4, 6, 3, 0, 0]
        assert inputs["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]
        assert ("token_type_ids" in inputs) == with_segment_ids

    def test_single_and_batch_helpers(self, with_segment_ids):
        tokenizer = _tokenizer(with_segment_ids)
        ids, segment_ids, mask = create_single_input_bert("good work", maxlen=6, transformer_tokenizer=tokenizer)
        assert ids == [2, 4, 6, 3, 0, 0] and mask == [1, 1, 1, 1, 0, 0]
        assert (segment_ids is not None) == with_segment_ids

        batch_ids, _, batch_mask = create_batch_input_bert(["good work", "bad"], 6, transformer_tokenizer=tokenizer)
        assert batch_ids == [[2, 4, 6, 3, 0, 0], [2, 5, 3, 0, 0, 0]]


class _NoSegmentIds(torch.nn.Module):
    """A transformer that takes no segment ids, as DistilBERT and ModernBERT."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.embeddings = torch.nn.Embedding(10, 8)

    def forward(self, input_ids, attention_mask=None):
        return type("Output", (), {"last_hidden_state": self.embeddings(input_ids)})()


def test_a_transformer_that_takes_no_segment_ids_is_not_given_any():
    from delft.textClassification.models import getModel

    model_config = ModelConfig(
        model_name="test", architecture="bert", list_classes=["a", "b"], transformer_name="in-memory"
    )
    with patch("transformers.AutoModel.from_pretrained", return_value=_NoSegmentIds()):
        model = getModel(model_config, TrainingConfig(learning_rate=1e-3))
    inputs = {
        "input_ids": torch.tensor([[2, 4, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "token_type_ids": torch.zeros(1, 3, dtype=torch.long),
    }
    assert model(inputs, labels=torch.tensor([[1.0, 0.0]]))["logits"].shape == (1, 2)
