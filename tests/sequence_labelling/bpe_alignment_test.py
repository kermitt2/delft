"""Labels are aligned on the words of a byte-level BPE tokenizer (RoBERTa, GPT2...)."""

from unittest.mock import patch

import pytest

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.preprocess import BERTPreprocessor, Preprocessor

WORDS = ["Deep", "John", "xy"]
LABELS = ["B-title", "B-author", "O"]


def _byte_level_tokenizer(add_prefix_space):
    """A RoBERTa-like tokenizer built in memory: a word is one sub-token, 'Ġ' standing for a leading space."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["<pad>", "<s>", "</s>", "<unk>"] + WORDS + ["Ġ" + word for word in WORDS]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


def _aligned_labels(add_prefix_space):
    preprocessor = BERTPreprocessor(_byte_level_tokenizer(add_prefix_space))
    preprocessor.is_BPE_SP = True  # inferred from the class name of a real tokenizer
    *_, labels, _ = preprocessor.tokenize_and_align_features_and_labels([WORDS], [None], None, [LABELS], maxlen=24)
    return [label for label in labels[0] if label != "<PAD>"]


def test_every_word_keeps_its_label_with_a_prefix_space():
    assert _aligned_labels(add_prefix_space=True) == LABELS


def test_without_a_prefix_space_words_lose_their_first_sub_token():
    """Why the tokenizer has to be loaded with add_prefix_space: this is what happens otherwise."""
    assert _aligned_labels(add_prefix_space=False) != LABELS


@pytest.mark.parametrize("architecture", ["BERT", "BERT_CRF"])
def test_the_loader_asks_for_a_prefix_space(architecture):
    preprocessor = Preprocessor(return_chars=False)
    preprocessor.fit([WORDS], [LABELS])
    config = ModelConfig(
        architecture=architecture, embeddings_name=None, transformer_name="in-memory", max_sequence_length=24
    )
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=_byte_level_tokenizer(True)
    ) as from_pretrained:
        create_dataloader([WORDS], [LABELS], preprocessor=preprocessor, shuffle=False, model_config=config)
    from_pretrained.assert_called_once_with("in-memory", add_prefix_space=True)
