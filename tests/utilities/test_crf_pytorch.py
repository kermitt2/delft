"""
Tests for the CRF layers in delft/utilities/crf_pytorch.py
"""

import logging

import torch
from torch.optim import Adam

from delft.utilities.crf_pytorch import ChainCRF

LOGGER = logging.getLogger(__name__)

NUM_TAGS = 5
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4

PARAMETER_NAMES = {"U", "b_start", "b_end"}


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
