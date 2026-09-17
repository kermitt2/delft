"""Every transformer architecture trains and decodes, with a tiny random transformer in place of a download."""

from unittest.mock import patch

import pytest
import torch

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import MODEL_REGISTRY, get_model

NTAGS = 5
BATCH, LENGTH, NB_FEATURES = 2, 6, 3
BERT_ARCHITECTURES = sorted(name for name in MODEL_REGISTRY if name.startswith("BERT"))


def _tiny_transformer(*args, **kwargs):
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )


def _model(architecture):
    config = ModelConfig(
        architecture=architecture,
        embeddings_name=None,
        transformer_name="tiny-random",
        max_sequence_length=LENGTH,
        features_indices=list(range(NB_FEATURES)),
    )
    with patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_transformer):
        return get_model(config, NTAGS, load_pretrained_weights=True)


def _inputs():
    generator = torch.Generator().manual_seed(0)
    attention_mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    attention_mask[1, 4:] = 0  # the second sequence is shorter
    return {
        "input_ids": torch.randint(1, 50, (BATCH, LENGTH), generator=generator),
        "token_type_ids": torch.zeros(BATCH, LENGTH, dtype=torch.long),
        "attention_mask": attention_mask,
        "features_input": torch.randint(1, 10, (BATCH, LENGTH, NB_FEATURES), generator=generator),
    }


def test_there_are_transformer_architectures_to_check():
    assert {"BERT", "BERT_CRF", "BERT_ChainCRF", "BERT_FEATURES", "BERT_CRF_FEATURES", "BERT_ChainCRF_FEATURES"} <= set(
        BERT_ARCHITECTURES
    )


@pytest.mark.parametrize("architecture", BERT_ARCHITECTURES)
def test_scores_every_label_and_trains(architecture):
    model = _model(architecture)
    inputs = _inputs()
    labels = torch.randint(1, NTAGS, (BATCH, LENGTH), generator=torch.Generator().manual_seed(1))
    labels = labels * inputs["attention_mask"]

    outputs = model(inputs, labels=labels)
    assert outputs["logits"].shape == (BATCH, LENGTH, NTAGS)
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()


@pytest.mark.parametrize("architecture", [name for name in BERT_ARCHITECTURES if "CRF" in name])
def test_decodes_to_labels(architecture):
    model = _model(architecture).eval()
    predictions = model.decode(_inputs())
    assert len(predictions) == BATCH
    assert all(0 <= tag < NTAGS for sequence in predictions for tag in sequence)
