"""Tests for the CRF layers in delft/utilities/crf_pytorch.py

The Viterbi decodes must stay bit-for-bit identical to pytorch-crf. ``CRF.decode``
no longer delegates to ``torchcrf.CRF.decode``. It runs one of two
reimplementations instead, for speed — a TorchScript one and a numpy twin,
chosen by batch size. Both are only safe swaps as long as they return exactly
the same tags, so pin them against the reference and against each other.

The loss is not delegated to ``torchcrf.CRF.forward`` either. It cannot be
bit-for-bit identical — the same sums are taken in another order — so it is
pinned to rounding instead, gradients included, since a loss that matches while
its gradients do not would train another model.
"""

import logging

import pytest
import torch
from torch.optim import Adam

from delft.utilities.crf_pytorch import (
    CPU_LOSS_MAX_SCORES_PER_STEP,
    CRF,
    HAS_TORCHCRF,
    NUMPY_DECODE_MAX_BATCH,
    ChainCRF,
    log_partition,
    loss_device,
    viterbi_decode,
    viterbi_decode_numpy,
)

LOGGER = logging.getLogger(__name__)

IMPLEMENTATIONS = [("scripted", viterbi_decode), ("numpy", viterbi_decode_numpy)]

# Only the CRF suite needs pytorch-crf as a reference; ChainCRF is standalone.
requires_torchcrf = pytest.mark.skipif(not HAS_TORCHCRF, reason="pytorch-crf not installed")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")

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


REDUCTIONS = ["none", "sum", "mean", "token_mean"]

# double precision leaves rounding far below any real difference
DOUBLE_TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}


def _loss_inputs(shape, batch_first, variable_length, dtype=torch.float64):
    """A reference pytorch-crf layer, a CRF with its parameters, and a batch for both."""
    seq_length, batch_size, num_tags = shape
    ref = _reference(num_tags, batch_first, seed=seq_length).to(dtype)
    crf = CRF(num_tags, batch_first=batch_first).to(dtype)
    crf.crf.load_state_dict(ref.state_dict())

    emissions, mask = _emissions_and_mask(seq_length, batch_size, num_tags, variable_length)
    emissions = emissions.to(dtype)
    tags = torch.randint(0, num_tags, (seq_length, batch_size))
    if batch_first:
        emissions, mask, tags = emissions.transpose(0, 1), mask.transpose(0, 1), tags.transpose(0, 1)
    return ref, crf, emissions.contiguous(), tags.contiguous(), mask.contiguous()


def _loss_and_gradients(layer, emissions, tags, sign=1.0, **kwargs):
    """The loss of a layer, with its gradients for the emissions and for each parameter."""
    emissions = emissions.detach().clone().requires_grad_(True)
    layer.zero_grad()
    loss = sign * layer(emissions, tags, **kwargs)
    loss.sum().backward()
    # a parameter the loss does not reach has no gradient, which is a gradient of zero:
    # the transitions of pytorch-crf over a single timestep
    gradients = {
        name: torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.clone()
        for name, parameter in layer.named_parameters()
    }
    gradients["emissions"] = emissions.grad
    return loss.detach(), gradients


def _assert_same_loss_and_gradients(ref, crf, emissions, tags, tolerance=DOUBLE_TOLERANCE, reduction="mean", **kwargs):
    # pytorch-crf returns the log-likelihood, the layer its negation, and the two do not
    # default to the same reduction
    kwargs["reduction"] = reduction
    expected_loss, expected_gradients = _loss_and_gradients(ref, emissions, tags, sign=-1.0, **kwargs)
    loss, gradients = _loss_and_gradients(crf, emissions, tags, **kwargs)

    torch.testing.assert_close(loss, expected_loss, **tolerance)
    assert set(gradients) == {"emissions"} | {f"crf.{name}" for name in expected_gradients if name != "emissions"}
    for name, expected in expected_gradients.items():
        key = name if name == "emissions" else f"crf.{name}"
        torch.testing.assert_close(gradients[key], expected, **tolerance)


@requires_torchcrf
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("variable_length", [False, True])
@pytest.mark.parametrize("batch_first", [False, True])
def test_loss_and_gradients_match_pytorch_crf(shape, variable_length, batch_first):
    ref, crf, emissions, tags, mask = _loss_inputs(shape, batch_first, variable_length)

    _assert_same_loss_and_gradients(ref, crf, emissions, tags, mask=mask)


@requires_torchcrf
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_loss_matches_pytorch_crf_for_every_reduction(reduction):
    ref, crf, emissions, tags, mask = _loss_inputs((17, 3, 4), batch_first=True, variable_length=True)

    _assert_same_loss_and_gradients(ref, crf, emissions, tags, mask=mask, reduction=reduction)


@requires_torchcrf
@pytest.mark.parametrize("batch_first", [False, True])
def test_loss_without_a_mask_takes_every_timestep_like_pytorch_crf(batch_first):
    ref, crf, emissions, tags, _ = _loss_inputs((17, 3, 4), batch_first, variable_length=True)

    _assert_same_loss_and_gradients(ref, crf, emissions, tags)


@requires_torchcrf
def test_loss_in_single_precision_matches_pytorch_crf_to_rounding():
    """What training runs: the two differ by the order of their sums and by no more."""
    ref, crf, emissions, tags, mask = _loss_inputs(
        (50, 32, 12), batch_first=True, variable_length=True, dtype=torch.float32
    )

    _assert_same_loss_and_gradients(ref, crf, emissions, tags, tolerance={"rtol": 1e-4, "atol": 1e-5}, mask=mask)


@requires_torchcrf
def test_loss_takes_the_mask_of_a_transformer():
    """BERT_CRF passes its attention mask as floats."""
    ref, crf, emissions, tags, mask = _loss_inputs((17, 3, 4), batch_first=True, variable_length=True)

    loss = crf(emissions, tags, mask=mask.float())

    torch.testing.assert_close(loss, -ref(emissions, tags, mask=mask, reduction="mean"), **DOUBLE_TOLERANCE)


@requires_torchcrf
def test_loss_matches_pytorch_crf_without_pytorch_crf_installed(monkeypatch):
    """Without pytorch-crf the layer holds the parameters itself, and computes the same loss."""
    ref, _, emissions, tags, mask = _loss_inputs((17, 3, 4), batch_first=True, variable_length=True)
    monkeypatch.setattr("delft.utilities.crf_pytorch.HAS_TORCHCRF", False)
    crf = CRF(4, batch_first=True).double()
    assert not hasattr(crf, "crf")
    crf.load_state_dict(ref.state_dict())

    loss = crf(emissions, tags, mask=mask)

    torch.testing.assert_close(loss, -ref(emissions, tags, mask=mask, reduction="mean"), **DOUBLE_TOLERANCE)


def test_log_partition_does_not_depend_on_what_is_padded():
    """Padding a batch must not move the loss of the sequences in it."""
    torch.manual_seed(7)
    num_tags, length, padding = 4, 6, 5
    start, transitions, end = torch.randn(num_tags), torch.randn(num_tags, num_tags), torch.randn(num_tags)
    emissions = torch.randn(3, length, num_tags, dtype=torch.float64)
    padded = torch.cat([emissions, 100 * torch.randn(3, padding, num_tags, dtype=torch.float64)], dim=1)
    mask = torch.arange(length + padding).unsqueeze(0).expand(3, -1) < length
    start, transitions, end = start.double(), transitions.double(), end.double()

    torch.testing.assert_close(
        log_partition(padded, mask, start, transitions, end),
        log_partition(emissions, None, start, transitions, end),
        **DOUBLE_TOLERANCE,
    )


def test_backward_of_the_loss_does_not_index_the_emissions_per_timestep():
    """
    Indexing the emissions inside the loop is what made the loss slow: the
    backward pass of each ``emissions[i]`` builds a gradient of the size of the
    whole tensor. The timesteps are unbound once instead, so the graph of the
    loss must hold as many selects for a long sequence as for a short one.
    """
    crf = CRF(4, batch_first=True)

    def node_counts(seq_length):
        emissions = torch.randn(3, seq_length, 4, requires_grad=True)
        tags = torch.randint(0, 4, (3, seq_length))
        mask = torch.arange(seq_length).unsqueeze(0) < torch.tensor([seq_length, 3, 2]).unsqueeze(1)

        pending, seen, counts = [crf(emissions, tags, mask=mask).grad_fn], set(), {}
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            name = type(node).__name__.rstrip("0123456789")
            counts[name] = counts.get(name, 0) + 1
            pending.extend(parent for parent, _ in node.next_functions)
        return counts

    short, long = node_counts(5), node_counts(40)

    assert long["LogsumexpBackward"] > short["LogsumexpBackward"]  # the recurrence is there
    assert long.get("SelectBackward", 0) == short.get("SelectBackward", 0)
    assert long.get("IndexBackward", 0) == short.get("IndexBackward", 0)
    assert long["UnbindBackward"] == short["UnbindBackward"] == 1


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"reduction": "median"}, "invalid reduction"),
        ({"mask": torch.tensor([[False, True, True], [True, True, False]])}, "first timestep"),
        ({"mask": torch.ones(2, 4, dtype=torch.bool)}, "emissions and mask must match"),
        ({"tags": torch.zeros(2, 4, dtype=torch.long)}, "emissions and tags must match"),
        ({"emissions": torch.zeros(2, 3, 7)}, "last dimension of emissions"),
    ],
)
def test_loss_refuses_what_pytorch_crf_refuses(arguments, message):
    crf = CRF(NUM_TAGS, batch_first=True)
    call = {"emissions": torch.zeros(2, 3, NUM_TAGS), "tags": torch.zeros(2, 3, dtype=torch.long), **arguments}

    with pytest.raises(ValueError, match=message):
        crf(call.pop("emissions"), call.pop("tags"), **call)


@pytest.mark.parametrize(
    "device,batch_size,num_tags,expected",
    [
        ("cuda", 30, 5, "cpu"),  # long sequences of few tags: reference-segmenter
        ("cuda", 20, 34, "cpu"),
        ("cuda", 200, 37, "cuda"),  # batches of short sequences: citation
        ("cuda:1", 200, 37, "cuda:1"),
        ("cuda", CPU_LOSS_MAX_SCORES_PER_STEP, 1, "cpu"),
        ("cuda", CPU_LOSS_MAX_SCORES_PER_STEP + 1, 1, "cuda"),
        ("cpu", 30, 5, "cpu"),
        ("cpu", 200, 37, "cpu"),
    ],
)
def test_loss_of_a_small_batch_on_a_gpu_is_computed_on_the_cpu(device, batch_size, num_tags, expected):
    """The split is a measured performance crossover — keep it wired up."""
    assert loss_device(torch.device(device), batch_size, num_tags) == torch.device(expected)


@requires_torchcrf
@pytest.mark.parametrize("with_mask", [False, True])
def test_loss_computed_on_another_device_matches_pytorch_crf(with_mask, monkeypatch):
    """
    The path a small batch takes on a GPU, without one: ``cpu:0`` is the CPU under another
    name, which is enough for the batch and the parameters to go through the move.
    """
    elsewhere = torch.device("cpu:0")
    moved = []
    monkeypatch.setattr(
        "delft.utilities.crf_pytorch.loss_device",
        lambda device, batch_size, num_tags: (moved.append((batch_size, num_tags)), elsewhere)[1],
    )
    ref, crf, emissions, tags, mask = _loss_inputs((17, 3, 4), batch_first=False, variable_length=True)
    kwargs = {"mask": mask} if with_mask else {}

    _assert_same_loss_and_gradients(ref, crf, emissions, tags, **kwargs)
    assert moved and set(moved) == {(3, 4)}  # the batch size and not the length, whatever the layout


@requires_torchcrf
@requires_cuda
@pytest.mark.parametrize("shape", [(50, 4, 12), (50, 128, 12)])
def test_loss_on_a_gpu_matches_pytorch_crf_wherever_it_is_computed(shape):
    """One batch under the crossover and one over it: the same loss, returned on the GPU either way."""
    _, batch_size, num_tags = shape
    ref, crf, emissions, tags, mask = _loss_inputs(shape, batch_first=True, variable_length=True)
    ref, crf = ref.cuda(), crf.cuda()
    emissions, tags, mask = emissions.cuda(), tags.cuda(), mask.cuda()
    on_cpu = batch_size * num_tags <= CPU_LOSS_MAX_SCORES_PER_STEP
    assert (loss_device(emissions.device, batch_size, num_tags).type == "cpu") == on_cpu

    assert crf(emissions, tags, mask=mask).device == emissions.device
    _assert_same_loss_and_gradients(ref, crf, emissions, tags, mask=mask)


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

    def test_decode_returns_lists_of_tag_indices_like_the_crf_layer(self):
        """A tensor element is not the integer it holds: looked up in an index-to-tag
        mapping it is a KeyError, which is how tagging with a ChainCRF model failed."""
        chain_crf = ChainCRF(NUM_TAGS)
        emissions, _ = _emissions_and_tags()
        for decoded in (chain_crf.decode(emissions), chain_crf(emissions)):
            assert isinstance(decoded, list)
            assert all(isinstance(tags, list) and all(type(tag) is int for tag in tags) for tags in decoded)
            assert [len(tags) for tags in decoded] == [SEQUENCE_LENGTH] * BATCH_SIZE

    def test_masked_positions_decode_to_the_padding_tag(self):
        chain_crf = ChainCRF(NUM_TAGS)
        emissions, _ = _emissions_and_tags()
        mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH)
        mask[0, 2:] = 0
        decoded = chain_crf.decode(emissions, mask=mask)
        assert decoded[0][2:] == [0] * (SEQUENCE_LENGTH - 2)
        assert len(decoded[0]) == SEQUENCE_LENGTH
