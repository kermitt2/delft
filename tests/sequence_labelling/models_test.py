import pytest
import torch
import torch.nn as nn

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import (
    MODEL_REGISTRY,
    BidLSTM_CRF,
    CharacterEncoder,
    get_token_mask,
    run_masked_lstm,
)

CHAR_VOCAB_SIZE = 10
CHAR_EMBEDDING_SIZE = 4
HIDDEN_SIZE = 3
WORD_EMBEDDING_SIZE = 8
NTAGS = 5

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

# two documents, the first a token shorter than the second, so that batching
# them together pads the first
CHAR_INPUT = torch.tensor(
    [
        [[1, 2], [3, 1], [0, 0]],
        [[4, 5], [6, 7], [8, 9]],
    ]
)

SHORTER_DOCUMENT_LENGTH = 2


def get_model_config(architecture: str) -> ModelConfig:
    config = ModelConfig(
        architecture=architecture,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        char_emb_size=CHAR_EMBEDDING_SIZE,
        char_lstm_units=HIDDEN_SIZE,
        word_lstm_units=6,
        dropout=0.0,
        recurrent_dropout=0.0,
    )
    config.char_vocab_size = CHAR_VOCAB_SIZE
    config.case_vocab_size = 8
    return config


@pytest.fixture(name="model")
def _model():
    # an untrained model may decode the same tags either way by chance; this
    # seed is one where the unmasked implementation decodes different ones
    torch.manual_seed(4)
    model = BidLSTM_CRF(get_model_config("BidLSTM_CRF"), NTAGS)
    model.eval()
    return model


@pytest.fixture(name="word_input")
def _word_input():
    torch.manual_seed(1)
    return torch.randn(CHAR_INPUT.shape[0], CHAR_INPUT.shape[1], WORD_EMBEDDING_SIZE)


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
        model = MODEL_REGISTRY[architecture](get_model_config(architecture), ntags=NTAGS)
        assert model.char_encoder.mask_zero is expected_mask_zero


class TestGetTokenMask:
    def test_should_mark_a_token_with_at_least_one_character_as_real(self):
        mask = get_token_mask(torch.tensor([[[1, 0], [0, 2]]]))
        assert mask.tolist() == [[True, True]]

    def test_should_mark_a_token_that_is_entirely_padding(self):
        mask = get_token_mask(torch.tensor([[[1, 2], [0, 0]]]))
        assert mask.tolist() == [[True, False]]

    def test_should_keep_the_first_position_unmasked_where_the_crf_requires_it(self):
        mask = get_token_mask(torch.zeros(1, 2, 2, dtype=torch.long))
        assert mask.tolist() == [[True, False]]


class TestRunMaskedLstm:
    def test_should_return_the_full_padded_length(self):
        lstm = nn.LSTM(4, 3, batch_first=True, bidirectional=True)
        x = torch.randn(2, 5, 4)
        mask = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
        assert run_masked_lstm(lstm, x, mask).shape == (2, 5, 6)

    def test_should_not_let_padding_reach_the_real_positions(self):
        torch.manual_seed(42)
        lstm = nn.LSTM(4, 3, batch_first=True, bidirectional=True)
        x = torch.randn(1, 5, 4)
        padded = run_masked_lstm(lstm, x, torch.tensor([[True] * 3 + [False] * 2]))
        unpadded = run_masked_lstm(lstm, x[:, :3], torch.tensor([[True] * 3]))
        # without the mask the backward direction would start in the padding and
        # run back through it, reaching every position
        assert torch.allclose(padded[:, :3], unpadded, atol=1e-6)


class TestBidLSTMCRF:
    def test_should_give_the_same_logits_whatever_the_document_is_batched_with(self, model, word_input):
        with torch.no_grad():
            batched = model({"word_input": word_input, "char_input": CHAR_INPUT})
            on_its_own = model(
                {
                    "word_input": word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                    "char_input": CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
                }
            )
        assert torch.allclose(
            batched["logits"][:1, :SHORTER_DOCUMENT_LENGTH],
            on_its_own["logits"],
            atol=1e-6,
        )

    def test_should_give_the_same_tags_whatever_the_document_is_batched_with(self, model, word_input):
        batched = model.decode({"word_input": word_input, "char_input": CHAR_INPUT})
        on_its_own = model.decode(
            {
                "word_input": word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                "char_input": CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
            }
        )
        assert batched[0][:SHORTER_DOCUMENT_LENGTH] == on_its_own[0]

    def test_should_give_the_same_loss_whatever_the_document_is_batched_with(self, model, word_input):
        labels = torch.tensor([[1, 2, 0], [3, 1, 2]])
        with torch.no_grad():
            batched = model(
                {"word_input": word_input[:1], "char_input": CHAR_INPUT[:1]},
                labels=labels[:1],
            )
            on_its_own = model(
                {
                    "word_input": word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                    "char_input": CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
                },
                labels=labels[:1, :SHORTER_DOCUMENT_LENGTH],
            )
        assert torch.allclose(batched["loss"], on_its_own["loss"], atol=1e-6)

    def test_should_decode_one_tag_per_position(self, model, word_input):
        # a masked decode returns only the real positions, where callers expect
        # a rectangular result
        predictions = model.decode({"word_input": word_input, "char_input": CHAR_INPUT})
        assert [len(tags) for tags in predictions] == [CHAR_INPUT.shape[1]] * CHAR_INPUT.shape[0]

    def test_should_decode_padded_positions_as_the_padding_tag(self, model, word_input):
        predictions = model.decode({"word_input": word_input, "char_input": CHAR_INPUT})
        assert predictions[0][SHORTER_DOCUMENT_LENGTH:] == [0]


# The architectures whose Keras counterparts masked padded tokens, via the
# character embedding's mask_zero=True propagating into the word-level RNN
TOKEN_MASKED_ARCHITECTURES = [
    "BidLSTM",
    "BidLSTM_CRF",
    "BidGRU_CRF",
    "BidLSTM_CRF_CASING",
    "BidLSTM_CRF_FEATURES",
]


def _get_masked_model(architecture: str):
    # an untrained model may decode the same tags either way by chance; this
    # seed is one where the unmasked implementation decodes different ones
    torch.manual_seed(5)
    model = MODEL_REGISTRY[architecture](get_model_config(architecture), ntags=NTAGS)
    model.eval()
    return model


def _get_inputs(architecture: str, word_input: torch.Tensor, char_input: torch.Tensor) -> dict:
    inputs = {"word_input": word_input, "char_input": char_input}
    batch_size, sequence_length = char_input.shape[:2]
    if architecture == "BidLSTM_CRF_CASING":
        inputs["casing_input"] = torch.ones(batch_size, sequence_length, dtype=torch.long)
    if architecture == "BidLSTM_CRF_FEATURES":
        inputs["features_input"] = torch.ones(batch_size, sequence_length, 1, dtype=torch.long)
    return inputs


class TestTokenMaskedArchitectures:
    @pytest.mark.parametrize("architecture", TOKEN_MASKED_ARCHITECTURES)
    def test_should_give_the_same_logits_whatever_the_document_is_batched_with(self, architecture, word_input):
        model = _get_masked_model(architecture)
        with torch.no_grad():
            batched = model(_get_inputs(architecture, word_input, CHAR_INPUT))
            on_its_own = model(
                _get_inputs(
                    architecture,
                    word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                    CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
                )
            )
        assert torch.allclose(
            batched["logits"][:1, :SHORTER_DOCUMENT_LENGTH],
            on_its_own["logits"],
            atol=1e-6,
        )

    @pytest.mark.parametrize("architecture", TOKEN_MASKED_ARCHITECTURES)
    def test_should_give_the_same_tags_whatever_the_document_is_batched_with(self, architecture, word_input):
        model = _get_masked_model(architecture)
        batched = model.decode(_get_inputs(architecture, word_input, CHAR_INPUT))
        on_its_own = model.decode(
            _get_inputs(
                architecture,
                word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
            )
        )
        assert batched[0][:SHORTER_DOCUMENT_LENGTH] == on_its_own[0][:SHORTER_DOCUMENT_LENGTH]

    @pytest.mark.parametrize("architecture", TOKEN_MASKED_ARCHITECTURES)
    def test_should_give_the_same_loss_whatever_the_document_is_batched_with(self, architecture, word_input):
        model = _get_masked_model(architecture)
        labels = torch.tensor([[1, 2, 0], [3, 1, 2]])
        with torch.no_grad():
            batched = model(
                _get_inputs(architecture, word_input[:1], CHAR_INPUT[:1]),
                labels=labels[:1],
            )
            on_its_own = model(
                _get_inputs(
                    architecture,
                    word_input[:1, :SHORTER_DOCUMENT_LENGTH],
                    CHAR_INPUT[:1, :SHORTER_DOCUMENT_LENGTH],
                ),
                labels=labels[:1, :SHORTER_DOCUMENT_LENGTH],
            )
        assert torch.allclose(batched["loss"], on_its_own["loss"], atol=1e-6)

    @pytest.mark.parametrize("architecture", TOKEN_MASKED_ARCHITECTURES)
    def test_should_decode_one_tag_per_position(self, architecture, word_input):
        model = _get_masked_model(architecture)
        predictions = model.decode(_get_inputs(architecture, word_input, CHAR_INPUT))
        assert [len(tags) for tags in predictions] == [CHAR_INPUT.shape[1]] * CHAR_INPUT.shape[0]
