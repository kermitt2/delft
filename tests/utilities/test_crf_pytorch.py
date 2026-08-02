"""The scripted Viterbi decode must stay bit-for-bit identical to pytorch-crf.

``CRF.decode`` no longer delegates to ``torchcrf.CRF.decode`` — it runs a
TorchScript reimplementation instead, for speed. That is only a safe swap as
long as it returns exactly the same tags, so pin it against the reference.
"""

import pytest
import torch

from delft.utilities.crf_pytorch import CRF, HAS_TORCHCRF, viterbi_decode

pytestmark = pytest.mark.skipif(not HAS_TORCHCRF, reason="pytorch-crf not installed")

# (seq_length, batch_size, num_tags), covering the single-timestep edge case,
# single-item batches (what GROBID sends) and long sequences.
SHAPES = [(1, 1, 3), (1, 8, 5), (2, 1, 2), (5, 4, 9), (17, 3, 4), (50, 32, 12), (500, 2, 8)]


def _reference(num_tags, batch_first, seed):
    from torchcrf import CRF as TorchCRF

    torch.manual_seed(seed)
    ref = TorchCRF(num_tags, batch_first=batch_first)
    with torch.no_grad():
        ref.start_transitions.uniform_(-2, 2)
        ref.end_transitions.uniform_(-2, 2)
        ref.transitions.uniform_(-2, 2)
    return ref


def _emissions_and_mask(seq_length, batch_size, num_tags, variable_length):
    emissions = torch.randn(seq_length, batch_size, num_tags)
    if variable_length and seq_length > 1:
        lengths = torch.randint(1, seq_length + 1, (batch_size,))
        # torchcrf requires the first timestep to be unmasked for every item
        lengths[0] = seq_length
    else:
        lengths = torch.full((batch_size,), seq_length)
    mask = torch.arange(seq_length).unsqueeze(1) < lengths.unsqueeze(0)
    return emissions, mask


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("variable_length", [False, True])
def test_scripted_decode_matches_pytorch_crf(shape, variable_length):
    seq_length, batch_size, num_tags = shape
    ref = _reference(num_tags, batch_first=False, seed=seq_length)
    emissions, mask = _emissions_and_mask(seq_length, batch_size, num_tags, variable_length)

    decoded = viterbi_decode(emissions, mask, ref.start_transitions, ref.transitions, ref.end_transitions)

    assert decoded == ref.decode(emissions, mask=mask)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("batch_first", [False, True])
def test_crf_wrapper_decode_matches_pytorch_crf(shape, batch_first):
    """The wrapper has to transpose into the time-first layout when batch_first."""
    seq_length, batch_size, num_tags = shape
    ref = _reference(num_tags, batch_first, seed=seq_length)
    crf = CRF(num_tags, batch_first=batch_first)
    crf.crf.load_state_dict(ref.state_dict())

    emissions, mask = _emissions_and_mask(seq_length, batch_size, num_tags, variable_length=True)
    if batch_first:
        emissions, mask = emissions.transpose(0, 1), mask.transpose(0, 1)

    assert crf.decode(emissions, mask=mask) == ref.decode(emissions, mask=mask)
    # mask=None must decode every timestep, like torchcrf does
    assert crf.decode(emissions) == ref.decode(emissions)
