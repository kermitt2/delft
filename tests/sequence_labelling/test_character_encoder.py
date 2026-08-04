"""
Character encoder masking.

Character rows are padded to ``max_char_length`` while the median token is a
few characters long. The encoder must skip that padding, the way the Keras
original did with ``mask_zero=True`` on the character embedding: the forward
state is read at the last real character, and the backward pass starts there
instead of running in from the padding.

The reference below is the definition of that behaviour — each token's row
run on its own, trimmed to its real length.
"""

import pytest
import torch

from delft.sequenceLabelling.models import CharacterEncoder

VOCAB_SIZE = 12
EMB_SIZE = 5
HIDDEN = 4


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    enc = CharacterEncoder(VOCAB_SIZE, EMB_SIZE, HIDDEN)
    enc.eval()
    return enc


def reference(enc, x):
    """Encode each token's characters on their own, trimmed to the real length."""
    batch_size, seq_len, _ = x.shape
    rows = []
    for row in x.reshape(batch_size * seq_len, -1):
        length = int((row != 0).sum())
        if length == 0:
            # an entirely masked sequence never updates the LSTM state
            rows.append(torch.zeros(enc.output_size))
            continue
        emb = enc.char_embeddings(row[:length].unsqueeze(0))
        _, (hidden, _) = enc.bilstm(emb)
        rows.append(torch.cat([hidden[0, 0], hidden[1, 0]], dim=-1))
    return torch.stack(rows).view(batch_size, seq_len, -1)


def _rows(words, width):
    """Lay tokens out as a [1, len(words), width] padded character batch."""
    padded = [list(w) + [0] * (width - len(w)) for w in words]
    return torch.tensor([padded], dtype=torch.long)


class TestMatchesPerWordReference:
    @pytest.mark.parametrize(
        "words",
        [
            [[3, 4, 5]],  # one ordinary token
            [[3, 4, 5], [7], [9, 2, 8, 1, 6]],  # mixed lengths
            [[1] * 30],  # a token filling the row, no padding at all
            [[5], [0]],  # a padded token slot next to a real one
            [[0]],  # nothing but padding
            [[2, 2], [2, 2]],  # repeated tokens
        ],
    )
    def test_matches(self, encoder, words):
        x = _rows(words, width=30)
        with torch.no_grad():
            assert torch.allclose(encoder(x), reference(encoder, x), atol=1e-6)

    def test_matches_on_a_random_batch(self, encoder):
        torch.manual_seed(7)
        lengths = torch.randint(0, 31, (3, 11))
        x = torch.zeros(3, 11, 30, dtype=torch.long)
        for b in range(3):
            for t in range(11):
                n = int(lengths[b, t])
                x[b, t, :n] = torch.randint(1, VOCAB_SIZE, (n,))
        with torch.no_grad():
            assert torch.allclose(encoder(x), reference(encoder, x), atol=1e-6)


class TestPaddingIsIgnored:
    def test_padding_width_does_not_change_the_encoding(self, encoder):
        """The same token encoded in a wider row must give the same vector."""
        with torch.no_grad():
            narrow = encoder(_rows([[3, 4, 5]], width=6))
            wide = encoder(_rows([[3, 4, 5]], width=30))
        assert torch.allclose(narrow, wide, atol=1e-6)

    def test_a_neighbour_token_does_not_leak(self, encoder):
        """Each token is encoded independently of the rest of the batch."""
        with torch.no_grad():
            alone = encoder(_rows([[3, 4, 5]], width=30))
            crowded = encoder(_rows([[3, 4, 5], [9, 9, 9, 9, 9], [0]], width=30))
        assert torch.allclose(alone[0, 0], crowded[0, 0], atol=1e-6)

    def test_an_all_padding_row_encodes_to_zero(self, encoder):
        with torch.no_grad():
            out = encoder(_rows([[7, 7], [0]], width=30))
        assert torch.equal(out[0, 1], torch.zeros(encoder.output_size))
        assert not torch.equal(out[0, 0], torch.zeros(encoder.output_size))


class TestShapeAndTraining:
    def test_preserves_shape_and_dtype(self, encoder):
        x = _rows([[3, 4], [5], [0]], width=30)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (1, 3, encoder.output_size)
        assert out.dtype == torch.float32

    def test_gradients_reach_the_lstm_and_the_embeddings(self, encoder):
        encoder.train()
        out = encoder(_rows([[3, 4, 5], [6], [0]], width=30))
        out.sum().backward()
        assert encoder.bilstm.weight_ih_l0.grad is not None
        assert encoder.bilstm.weight_ih_l0.grad.abs().sum() > 0
        assert encoder.char_embeddings.weight.grad.abs().sum() > 0

    def test_works_under_inference_mode(self, encoder):
        """The tagger runs inside inference_mode, which forbids some in-place work."""
        with torch.inference_mode():
            out = encoder(_rows([[3, 4, 5], [0]], width=30))
        assert out.shape == (1, 2, encoder.output_size)
