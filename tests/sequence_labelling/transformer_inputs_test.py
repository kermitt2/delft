"""What the loader gives a transformer, with a tokenizer whose padding id is not 0 (RoBERTa: 1)."""

from unittest.mock import patch

import numpy as np
import pytest

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import Preprocessor

WORDS = ["Deep", "John", "xy"]
LABELS = ["B-title", "B-author", "O"]
PAD_ID = 1


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["<s>", "<pad>", "</s>", "<unk>"] + WORDS
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    assert fast.pad_token_id == PAD_ID
    return fast


def _batch(tokenizer, x, y, max_sequence_length=32):
    preprocessor = Preprocessor(return_chars=False)
    preprocessor.fit([WORDS], [LABELS])
    config = ModelConfig(
        architecture="BERT_CRF",
        embeddings_name=None,
        transformer_name="in-memory",
        max_sequence_length=max_sequence_length,
    )
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        loader = create_dataloader(x, y, preprocessor=preprocessor, shuffle=False, model_config=config, batch_size=4)
        return next(iter(loader))


def test_a_batch_is_as_wide_as_its_longest_sequence_not_as_max_sequence_length(tokenizer):
    inputs, labels = _batch(tokenizer, [WORDS, WORDS[:1]], [LABELS, LABELS[:1]])
    # <s> Deep John xy </s>
    assert inputs["input_ids"].shape == (2, 5)
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert labels.shape == (2, 5)


def test_the_sentence_start_token_is_not_taken_for_padding(tokenizer):
    """Its id is 0, which is the padding id of other tokenizers."""
    inputs, _ = _batch(tokenizer, [WORDS], [LABELS])
    assert inputs["input_ids"][0].tolist()[0] == 0
    assert inputs["word_start_mask"][0].tolist() == [0, 1, 1, 1, 0]


def test_segment_ids_are_zero_on_every_position(tokenizer):
    """The padding id is not a segment: RoBERTa has a single one, and fails on the id 1."""
    inputs, _ = _batch(tokenizer, [WORDS, WORDS[:1]], [LABELS, LABELS[:1]])
    assert inputs["token_type_ids"].sum().item() == 0


def test_sequences_given_as_the_rows_of_an_array(tokenizer):
    """What the readers return for sequences of the same length, a file with a single one for instance."""
    x = np.array([WORDS, WORDS], dtype=object)
    y = np.array([LABELS, LABELS], dtype=object)
    assert x.ndim == 2
    inputs, _ = _batch(tokenizer, x, y)
    assert inputs["input_ids"].shape == (2, 5)
