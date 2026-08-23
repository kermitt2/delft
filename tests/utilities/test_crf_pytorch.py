"""Tests for the CRF layers in delft/utilities/crf_pytorch.py

The Viterbi decodes must stay bit-for-bit identical to pytorch-crf. ``CRF.decode``
no longer delegates to ``torchcrf.CRF.decode``. It runs one of two
reimplementations instead, for speed — a TorchScript one and a numpy twin,
chosen by batch size. Both are only safe swaps as long as they return exactly
the same tags, so pin them against the reference and against each other.
"""

import logging

import pytest
import torch
from torch.optim import Adam

from delft.utilities.crf_pytorch import (
    CRF,
    HAS_TORCHCRF,
    NUMPY_DECODE_MAX_BATCH,
    ChainCRF,
    viterbi_decode,
    viterbi_decode_numpy,
)

LOGGER = logging.getLogger(__name__)

IMPLEMENTATIONS = [("scripted", viterbi_decode), ("numpy", viterbi_decode_numpy)]

# Only the CRF suite needs pytorch-crf as a reference; ChainCRF is standalone.
requires_torchcrf = pytest.mark.skipif(not HAS_TORCHCRF, reason="pytorch-crf not installed")

# (seq_length, batch_size, num_tags), covering the single-timestep edge case,
# single-item batches (what GROBID sends) and long sequences.
SHAPES = [(1, 1, 3), (1, 8, 5), (2, 1, 2), (5, 4, 9), (17, 3, 4), (50, 32, 12), (500, 2, 8)]

NUM_TAGS = 5
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4

PARAMETER_NAMES = {"U", "b_start", "b_end"}


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


@requires_torchcrf
@pytest.mark.parametrize("name,decode", IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("variable_length", [False, True])
def test_decode_matches_pytorch_crf(name, decode, shape, variable_length):
    seq_length, batch_size, num_tags = shape
    ref = _reference(num_tags, batch_first=False, seed=seq_length)
    emissions, mask = _emissions_and_mask(seq_length, batch_size, num_tags, variable_length)

    decoded = decode(emissions, mask, ref.start_transitions, ref.transitions, ref.end_transitions)

    assert decoded == ref.decode(emissions, mask=mask)


@requires_torchcrf
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_the_two_implementations_agree(shape, tie_heavy):
    """
    The pair is only interchangeable if it agrees on tie-breaking too.

    Both argmaxes return the first of several equal maxima, so small-integer
    scores — which produce exact ties constantly, unlike random floats — are
    the case that would expose a divergence.
    """
    seq_length, batch_size, num_tags = shape
    torch.manual_seed(seq_length + batch_size)
    if tie_heavy:
        emissions = torch.randint(0, 3, (seq_length, batch_size, num_tags)).float()
        start = torch.randint(0, 3, (num_tags,)).float()
        transitions = torch.randint(0, 3, (num_tags, num_tags)).float()
        end = torch.randint(0, 3, (num_tags,)).float()
    else:
        emissions = torch.randn(seq_length, batch_size, num_tags)
        start, transitions, end = (
            torch.randn(num_tags),
            torch.randn(num_tags, num_tags),
            torch.randn(num_tags),
        )
    _, mask = _emissions_and_mask(seq_length, batch_size, num_tags, variable_length=True)

    assert viterbi_decode_numpy(emissions, mask, start, transitions, end) == viterbi_decode(
        emissions, mask, start, transitions, end
    )


@requires_torchcrf
@pytest.mark.parametrize(
    "batch_size,expected",
    [(1, "numpy"), (NUMPY_DECODE_MAX_BATCH, "numpy"), (NUMPY_DECODE_MAX_BATCH + 1, "scripted")],
)
def test_wrapper_picks_the_implementation_by_batch_size(batch_size, expected, monkeypatch):
    """The split is a measured performance crossover — keep it wired up."""
    called = []
    for name, decode in IMPLEMENTATIONS:
        monkeypatch.setattr(
            f"delft.utilities.crf_pytorch.viterbi_decode{'_numpy' if name == 'numpy' else ''}",
            lambda *a, _name=name, _decode=decode: (called.append(_name), _decode(*a))[1],
        )

    crf = CRF(4, batch_first=True)
    crf.decode(torch.randn(batch_size, 6, 4))

    assert called == [expected]


@requires_torchcrf
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


def _emissions_and_tags():
    torch.manual_seed(42)
    emissions = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, NUM_TAGS, requires_grad=True)
    tags = torch.randint(0, NUM_TAGS, (BATCH_SIZE, SEQUENCE_LENGTH))
    return emissions, tags


class TestChainCRF:
    """Tests for the ChainCRF layer."""

    def test_parameters_registered_when_num_tags_is_known(self):
        chain_crf = ChainCRF(NUM_TAGS)
        assert set(chain_crf.state_dict().keys()) == PARAMETER_NAMES
        assert chain_crf.U.shape == (NUM_TAGS, NUM_TAGS)
        assert chain_crf.b_start.shape == (NUM_TAGS,)
        assert chain_crf.b_end.shape == (NUM_TAGS,)

    def test_parameters_built_lazily_without_num_tags(self):
        chain_crf = ChainCRF()
        assert not chain_crf.state_dict()
        emissions, tags = _emissions_and_tags()
        chain_crf(emissions, tags)
        assert set(chain_crf.state_dict().keys()) == PARAMETER_NAMES

    def test_optimizer_created_before_first_forward_updates_transitions(self):
        chain_crf = ChainCRF(NUM_TAGS)
        optimizer = Adam(chain_crf.parameters(), lr=0.1)
        emissions, tags = _emissions_and_tags()
        transitions_before = chain_crf.U.detach().clone()
        for _ in range(5):
            optimizer.zero_grad()
            chain_crf(emissions, tags).backward()
            optimizer.step()
        assert not torch.equal(transitions_before, chain_crf.U.detach())

    def test_state_dict_round_trip(self):
        chain_crf = ChainCRF(NUM_TAGS)
        emissions, tags = _emissions_and_tags()
        chain_crf(emissions, tags)
        loaded_chain_crf = ChainCRF(NUM_TAGS)
        loaded_chain_crf.load_state_dict(chain_crf.state_dict())
        assert torch.equal(loaded_chain_crf.U.detach(), chain_crf.U.detach())

    def test_decode_returns_a_tag_per_token(self):
        chain_crf = ChainCRF(NUM_TAGS)
        emissions, _ = _emissions_and_tags()
        decoded = chain_crf.decode(emissions)
        assert torch.as_tensor(decoded).shape == (BATCH_SIZE, SEQUENCE_LENGTH)
