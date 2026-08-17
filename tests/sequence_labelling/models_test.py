import pytest
import torch

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import MODEL_REGISTRY, CharacterEncoder

CHAR_VOCAB_SIZE = 10
CHAR_EMBEDDING_SIZE = 4
HIDDEN_SIZE = 3

# What the Keras implementations set on the character embedding, per architecture
# (delft 0.4.3, delft/sequenceLabelling/models.py). The architectures using the
# CNN character encoder are absent: they have no character LSTM to mask.
CHARACTER_MASK_ZERO_BY_ARCHITECTURE = {
    "BidLSTM": True,
    "BidLSTM_CRF": True,
    "BidGRU_CRF": True,
    "BidLSTM_CRF_CASING": True,
    "BidLSTM_CRF_FEATURES": True,
    "BidLSTM_ChainCRF": False,
    "BidLSTM_ChainCRF_FEATURES": False,
}


def get_model_config(architecture: str) -> ModelConfig:
    config = ModelConfig(
        architecture=architecture,
        word_embedding_size=8,
        char_emb_size=CHAR_EMBEDDING_SIZE,
        char_lstm_units=HIDDEN_SIZE,
        word_lstm_units=6,
        dropout=0.0,
        recurrent_dropout=0.0,
    )
    config.char_vocab_size = CHAR_VOCAB_SIZE
    config.case_vocab_size = 8
    return config


class TestCharacterEncoder:
    def test_should_not_mask_padding_by_default(self):
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE)
        assert encoder.mask_zero is False

    @pytest.mark.parametrize("mask_zero", [False, True])
    def test_should_return_the_expected_shape(self, mask_zero: bool):
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE, mask_zero=mask_zero)
        encoder.eval()
        with torch.no_grad():
            output = encoder(torch.tensor([[[1, 2, 0], [3, 0, 0]]]))
        assert output.shape == (1, 2, HIDDEN_SIZE * 2)

    def test_should_encode_a_token_the_same_whatever_the_character_window(self):
        # the same token, padded to a wider character window
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE, mask_zero=True)
        encoder.eval()
        with torch.no_grad():
            narrow = encoder(torch.tensor([[[1, 2, 0]]]))
            wide = encoder(torch.tensor([[[1, 2, 0, 0, 0, 0]]]))
        assert torch.allclose(narrow, wide, atol=1e-6)

    def test_should_encode_each_token_independently_of_the_others(self):
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE, mask_zero=True)
        encoder.eval()
        tokens = torch.tensor([[[5, 6, 7, 8], [2, 0, 0, 0], [3, 4, 0, 0]]])
        with torch.no_grad():
            together = encoder(tokens)
            separately = torch.cat(
                [encoder(tokens[:, index : index + 1, :]) for index in range(tokens.shape[1])], dim=1
            )
        assert torch.allclose(together, separately, atol=1e-6)

    def test_should_encode_a_token_without_padding_the_same_as_without_masking(self):
        masked = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE, mask_zero=True)
        unmasked = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE)
        unmasked.load_state_dict(masked.state_dict())
        masked.eval()
        unmasked.eval()
        full_window = torch.tensor([[[1, 2, 3]]])
        with torch.no_grad():
            assert torch.allclose(masked(full_window), unmasked(full_window), atol=1e-6)

    def test_should_return_zeros_for_a_token_that_is_entirely_padding(self):
        # as Keras did, leaving the initial state rather than encoding the padding
        encoder = CharacterEncoder(CHAR_VOCAB_SIZE, CHAR_EMBEDDING_SIZE, HIDDEN_SIZE, mask_zero=True)
        encoder.eval()
        with torch.no_grad():
            output = encoder(torch.tensor([[[0, 0, 0]]]))
        assert torch.equal(output, torch.zeros_like(output))


class TestCharacterMaskZeroByArchitecture:
    @pytest.mark.parametrize("architecture,expected_mask_zero", sorted(CHARACTER_MASK_ZERO_BY_ARCHITECTURE.items()))
    def test_should_mask_padded_characters_where_the_keras_embedding_did(
        self, architecture: str, expected_mask_zero: bool
    ):
        model = MODEL_REGISTRY[architecture](get_model_config(architecture), ntags=5)
        assert model.char_encoder.mask_zero is expected_mask_zero
