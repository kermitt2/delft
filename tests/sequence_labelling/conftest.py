import os

import pytest

from .test_data import TEST_DATA_PATH


@pytest.fixture
def preprocessor1():
    return os.path.join(TEST_DATA_PATH, "preprocessor.json")


@pytest.fixture
def preprocessor2():
    return os.path.join(TEST_DATA_PATH, "preprocessor2.json")


@pytest.fixture
def wordpiece_tokenizer():
    """A BERT-like tokenizer built in memory, so that no model is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Jim", "He", "##nson", "##ization", "was", "a"]
    vocabulary += ["puppet", "##eer", "in", "Mississippi", "today"]
    tokenizer = Tokenizer(models.WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )
