"""
PyTorch CRF implementation for DeLFT sequence labeling models.

This module provides:
- CRF: Standard Conditional Random Field layer using pytorch-crf
- ChainCRF: Custom Linear Chain CRF with Viterbi decoding (ported from Keras)

References:
- Original Keras CRF: https://github.com/phipleg/keras/blob/crf/keras/layers/crf.py
- pytorch-crf: https://github.com/kmkurn/pytorch-crf
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

# Try to import TorchCRF from pytorch-crf package
try:
    from torchcrf import CRF as TorchCRF

    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False


@torch.jit.script
def viterbi_decode(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> List[List[int]]:
    """
    TorchScript Viterbi decoding, output-identical to ``torchcrf.CRF.decode``.

    pytorch-crf's decode is an interpreted per-timestep Python loop, which
    dominates inference time on the long sequences GROBID sends. Scripting it
    keeps the arithmetic identical while lifting the loop out of the
    interpreter.

    Args:
        emissions: [seq_length, batch_size, num_tags] — time-first, the
            torchcrf-internal layout.
        mask: [seq_length, batch_size] bool, same layout.
        start_transitions, transitions, end_transitions: the CRF's own
            parameters.

    Returns:
        One list of tag indices per batch item, each truncated to its own
        length as given by the mask.
    """
    seq_length = emissions.size(0)
    batch_size = emissions.size(1)

    # [T,B,1,K], so the per-step broadcast operand is a plain index rather than
    # an index plus an unsqueeze. Every op in the loop is on the critical path:
    # each step depends on the previous one, so nothing overlaps and the cost is
    # the number of dispatches, not the arithmetic (K is a few dozen).
    emissions_bcast = emissions.unsqueeze(2)

    score = start_transitions + emissions[0]
    history = torch.jit.annotate(List[torch.Tensor], [])

    if bool(mask.all().item()):
        # Nothing is padded — the usual case when a host tags one sequence at a
        # time. torch.where would then copy next_score onto itself, so drop it.
        for i in range(1, seq_length):
            next_score = score.unsqueeze(2) + transitions + emissions_bcast[i]  # [B,K,K]
            score, indices = next_score.max(dim=1)  # [B,K]
            history.append(indices)
    else:
        for i in range(1, seq_length):
            next_score = score.unsqueeze(2) + transitions + emissions_bcast[i]  # [B,K,K]
            next_score, indices = next_score.max(dim=1)  # [B,K]
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

    score = score + end_transitions

    # Backtracking is pure integer bookkeeping. Reading it off nested Python
    # lists costs one bulk copy, where indexing the tensors costs two selects
    # and a device sync per timestep.
    seq_ends: List[int] = mask.long().sum(dim=0).sub(1).tolist()
    best_last: List[int] = score.argmax(dim=1).tolist()
    backpointers = torch.jit.annotate(List[List[List[int]]], [])
    if len(history) > 0:
        backpointers = torch.jit.annotate(List[List[List[int]]], torch.stack(history).tolist())

    out = torch.jit.annotate(List[List[int]], [])

    for idx in range(batch_size):
        best = [best_last[idx]]
        for j in range(seq_ends[idx] - 1, -1, -1):
            best.append(backpointers[j][idx][best[-1]])
        best.reverse()
        out.append(best)

    return out


# Above this batch size the scripted decode wins and below it numpy does.
# numpy has the lower per-op overhead, which is what a narrow batch is bound
# by, while torch's max() returns values and indices in a single pass, which
# pays off once the arrays are big enough for the arithmetic to matter.
# Measured at T=300, K=44 — numpy/torch: B=1 1.38x, B=2 1.16x, B=3 1.10x,
# B=4 1.00x, B=6 0.90x, B=9 0.88x, B=20 0.77x. A host tagging one sequence at
# a time sits at the left end; training and evaluation batches sit at the right.
NUMPY_DECODE_MAX_BATCH = 3


def viterbi_decode_numpy(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> List[List[int]]:
    """
    numpy twin of :func:`viterbi_decode`, for narrow batches.

    Same operations in the same order, so the results are bit-identical: the
    broadcasts are IEEE float32 either way, and both argmax implementations
    return the first of several equal maxima. ``tests/utilities/
    test_crf_pytorch.py`` pins the two together, tie-heavy cases included.

    Arguments are as in :func:`viterbi_decode` — time-first emissions
    [seq_length, batch_size, num_tags] and a [seq_length, batch_size] mask.
    """
    em = emissions.detach().numpy()
    mk = mask.detach().numpy()
    start = start_transitions.detach().numpy()
    trans = transitions.detach().numpy()
    end = end_transitions.detach().numpy()

    seq_length, batch_size, num_tags = em.shape

    score = start + em[0]
    backpointers = np.empty((max(seq_length - 1, 0), batch_size, num_tags), dtype=np.int64)

    all_valid = bool(mk.all())
    for i in range(1, seq_length):
        next_score = score[:, :, None] + trans + em[i][:, None, :]  # [B,K,K]
        indices = next_score.argmax(axis=1)  # [B,K]
        best = next_score.max(axis=1)  # [B,K]
        score = best if all_valid else np.where(mk[i][:, None], best, score)
        backpointers[i - 1] = indices

    score = score + end

    seq_ends = mk.sum(axis=0) - 1
    best_last = score.argmax(axis=1)
    history = backpointers.tolist()

    out = []
    for idx in range(batch_size):
        best_path = [int(best_last[idx])]
        for j in range(int(seq_ends[idx]) - 1, -1, -1):
            best_path.append(history[j][idx][best_path[-1]])
        best_path.reverse()
        out.append(best_path)

    return out


class CRF(nn.Module):
    """
    Conditional Random Field layer using pytorch-crf.

    This is the primary CRF implementation for models like BidLSTM_CRF.

    Args:
        num_tags: Number of tags/labels
        batch_first: Whether the batch dimension is first (default: True)
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        if HAS_TORCHCRF:
            self.crf = TorchCRF(num_tags, batch_first=batch_first)
        else:
            # Fallback to custom implementation
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self._reset_parameters()

    def _reset_parameters(self):
        """Initialize transitions with uniform distribution."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the CRF loss or decode the best sequence.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            tags: Gold tag sequence [batch_size, seq_len] (required for training)
            mask: Mask tensor [batch_size, seq_len] (1 = valid, 0 = pad)
            reduction: Loss reduction method ('mean', 'sum', 'none')

        Returns:
            If tags is provided: negative log-likelihood loss
            If tags is None: best tag sequence
        """
        if HAS_TORCHCRF:
            if tags is not None:
                # Training: compute negative log-likelihood
                if mask is not None:
                    mask = mask.bool()
                return -self.crf(emissions, tags, mask=mask, reduction=reduction)
            else:
                # Inference: decode best sequence
                return self.decode(emissions, mask=mask)
        else:
            return self._forward_custom(emissions, tags, mask, reduction)

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Decode the best tag sequence using Viterbi algorithm.

        Decodes with one of this module's own implementations rather than
        ``torchcrf.CRF.decode``: same arithmetic, same output, but without the
        interpreted per-timestep loop. The loop is control-flow and
        latency-bound, so it runs on CPU whatever device the model sits on —
        on GPU it would cost a kernel launch plus a sync per timestep.

        Which of the two runs is decided by batch size alone, on the measured
        crossover recorded at ``NUMPY_DECODE_MAX_BATCH``. They are
        bit-identical, so this only ever changes how long the call takes.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            mask: Mask tensor [batch_size, seq_len]

        Returns:
            List of best tag sequences for each item in batch
        """
        if HAS_TORCHCRF:
            emissions = emissions.cpu()
            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool)
            else:
                mask = mask.bool().cpu()

            if self.batch_first:
                # torchcrf works time-first internally; match that layout.
                emissions = emissions.transpose(0, 1)
                mask = mask.transpose(0, 1)

            decode = viterbi_decode_numpy if emissions.size(1) <= NUMPY_DECODE_MAX_BATCH else viterbi_decode
            return decode(
                emissions,
                mask,
                self.crf.start_transitions.detach().cpu(),
                self.crf.transitions.detach().cpu(),
                self.crf.end_transitions.detach().cpu(),
            )
        else:
            return self._viterbi_decode_custom(emissions, mask)

    def _forward_custom(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor],
        reduction: str,
    ) -> torch.Tensor:
        """Custom forward pass without pytorch-crf."""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        else:
            mask = mask.bool()

        # Compute log-likelihood
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        else:
            return -llh

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the score of a tag sequence."""
        batch_size, seq_length = tags.shape

        # Start transition score
        score = self.start_transitions[tags[:, 0]]

        # Emission and transition scores
        for i in range(seq_length):
            score += emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1) * mask[:, i].float()
            if i < seq_length - 1:
                transition_score = self.transitions[tags[:, i], tags[:, i + 1]]
                score += transition_score * mask[:, i + 1].float()

        # End transition score (at last valid position)
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the partition function using forward algorithm."""
        batch_size, seq_length, num_tags = emissions.shape

        # Initialize alpha with start transitions
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            emit_score = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch, num_tags, 1]

            next_alpha = alpha_expand + trans_score + emit_score
            next_alpha = torch.logsumexp(next_alpha, dim=1)

            # Apply mask
            alpha = torch.where(mask[:, i : i + 1].bool(), next_alpha, alpha)

        # Add end transitions
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode_custom(self, emissions: torch.Tensor, mask: Optional[torch.Tensor]) -> List[List[int]]:
        """Viterbi decoding without pytorch-crf."""
        batch_size, seq_length, num_tags = emissions.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        # Initialize
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i : i + 1], next_score, score)
            history.append(indices)

        # Add end transitions
        score += self.end_transitions

        # Backtrack
        seq_ends = mask.long().sum(dim=1) - 1
        best_tags_list = []

        for idx in range(batch_size):
            best_last_tag = score[idx].argmax().item()
            best_tags = [best_last_tag]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_last_tag].item()
                best_tags.append(best_last_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class ChainCRF(nn.Module):
    """
    A Linear Chain Conditional Random Field output layer.

    This is a PyTorch port of the Keras ChainCRF implementation.
    It carries the loss function and its weights for computing
    the global tag sequence scores.

    During training it computes the CRF loss.
    During inference it applies Viterbi decoding and returns the best sequence.

    Args:
        num_tags: Number of possible tags
    """

    def __init__(self, num_tags: int = None):
        super().__init__()
        self._num_tags = num_tags
        self._built = False

        # Will be initialized in build()
        self.U = None  # Transition matrix
        self.b_start = None  # Start boundary energy
        self.b_end = None  # End boundary energy

    def build(self, num_tags: int, device=None, dtype=None):
        """Initialize layer weights."""
        self._num_tags = num_tags

        # Transition matrix (energy between tag pairs)
        self.U = nn.Parameter(torch.empty(num_tags, num_tags, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.U)

        # Boundary energies
        self.b_start = nn.Parameter(torch.zeros(num_tags, device=device, dtype=dtype))
        self.b_end = nn.Parameter(torch.zeros(num_tags, device=device, dtype=dtype))

        self._built = True

    def forward(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            tags: Gold tag sequence for training [batch_size, seq_len]
            mask: Mask tensor [batch_size, seq_len]

        Returns:
            During training (tags provided): CRF loss
            During inference: Best tag sequence [batch_size, seq_len]
        """
        if not self._built:
            self.build(emissions.size(-1), device=emissions.device, dtype=emissions.dtype)

        if tags is not None:
            # Training: compute loss
            return self.loss(emissions, tags, mask)
        else:
            # Inference: decode
            return self.decode(emissions, mask)

    def loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CRF negative log-likelihood loss.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            tags: Gold tag sequence [batch_size, seq_len]
            mask: Mask tensor [batch_size, seq_len]

        Returns:
            Scalar loss tensor
        """
        # Add boundary energies
        x = self._add_boundary_energy(emissions, mask)

        # Compute path energy for gold sequence
        path_e = self._path_energy(tags, x, mask)

        # Compute partition function (log of sum of all path energies)
        free_e = self._free_energy(x, mask)

        # NLL = -E(y, x) + log(Z)
        nll = -path_e + free_e

        return nll.mean()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Viterbi decoding to find best tag sequence.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            mask: Mask tensor [batch_size, seq_len]

        Returns:
            Best tag sequence [batch_size, seq_len]
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Add boundary energies
        x = self._add_boundary_energy(emissions, mask)

        # Forward pass with max instead of logsumexp
        alpha = x[:, 0, :]  # [batch, num_tags]
        backpointers = []

        for t in range(1, seq_len):
            # [batch, num_tags, 1] + [num_tags, num_tags] -> [batch, num_tags, num_tags]
            broadcast_alpha = alpha.unsqueeze(2)
            next_score = broadcast_alpha + self.U.unsqueeze(0) + x[:, t, :].unsqueeze(1)

            # Max over previous tags
            alpha, bp = next_score.max(dim=1)  # [batch, num_tags]
            backpointers.append(bp)

            # Apply mask if provided
            if mask is not None:
                alpha = torch.where(
                    mask[:, t : t + 1].bool(),
                    alpha,
                    x[:, t, :],  # Reset for masked positions
                )

        # Backtrack
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        best_last = alpha.argmax(dim=1)
        best_paths[:, -1] = best_last

        for t in range(seq_len - 2, -1, -1):
            best_paths[:, t] = backpointers[t].gather(1, best_paths[:, t + 1 : t + 2]).squeeze(1)

        # Apply mask
        if mask is not None:
            best_paths = best_paths * mask.long()

        return best_paths

    def _add_boundary_energy(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add start and end boundary energies to emission scores."""
        if mask is None:
            # Add start boundary to first position
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + self.b_start
            x[:, -1, :] = x[:, -1, :] + self.b_end
        else:
            x = x.clone()
            # Add start boundary to first valid position
            first_mask = self._get_first_mask(mask)
            x = x + first_mask.unsqueeze(-1) * self.b_start

            # Add end boundary to last valid position
            last_mask = self._get_last_mask(mask)
            x = x + last_mask.unsqueeze(-1) * self.b_end

        return x

    def _get_first_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get mask for first valid position in each sequence."""
        # Shift mask right and compare
        shifted = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
        return (mask > shifted).float()

    def _get_last_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get mask for last valid position in each sequence."""
        # Shift mask left and compare
        shifted = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, -1:])], dim=1)
        return (mask > shifted).float()

    def _path_energy(self, tags: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the energy of a tag path."""
        batch_size, seq_len = tags.shape
        num_tags = x.size(-1)

        # Emission energy
        tags_one_hot = torch.nn.functional.one_hot(tags, num_tags).float()
        emission_energy = (x * tags_one_hot).sum(dim=-1)  # [batch, seq_len]

        if mask is not None:
            emission_energy = emission_energy * mask.float()

        emission_energy = emission_energy.sum(dim=1)  # [batch]

        # Transition energy
        tags_from = tags[:, :-1]  # [batch, seq_len-1]
        tags_to = tags[:, 1:]  # [batch, seq_len-1]

        # Flatten transition indices
        U_flat = self.U.view(-1)
        trans_indices = tags_from * num_tags + tags_to
        trans_energy = U_flat[trans_indices]  # [batch, seq_len-1]

        if mask is not None:
            trans_mask = mask[:, :-1] * mask[:, 1:]
            trans_energy = trans_energy * trans_mask.float()

        trans_energy = trans_energy.sum(dim=1)  # [batch]

        return emission_energy + trans_energy

    def _free_energy(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the partition function (log normalizer) using forward algorithm."""
        batch_size, seq_len, num_tags = x.shape

        # Initialize alpha with first emission
        alpha = x[:, 0, :]  # [batch, num_tags]

        for t in range(1, seq_len):
            # [batch, num_tags, 1] + [num_tags, num_tags] + [batch, 1, num_tags]
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emit = x[:, t, :].unsqueeze(1)
            next_alpha = broadcast_alpha + self.U.unsqueeze(0) + broadcast_emit

            # Log-sum-exp over previous tags
            alpha = torch.logsumexp(next_alpha, dim=1)

            # Apply mask
            if mask is not None:
                alpha = torch.where(
                    mask[:, t : t + 1].bool(),
                    alpha,
                    alpha,  # Keep alpha unchanged for masked positions
                )

        # Final log-sum-exp over all tags
        return torch.logsumexp(alpha, dim=1)


def sparse_crf_loss_masked(emissions: torch.Tensor, tags: torch.Tensor, crf: CRF, mask_value: int = 0) -> torch.Tensor:
    """
    CRF loss with special token masking (for BERT models).

    Masks out positions where tags equal mask_value (typically padding/special tokens).

    Args:
        emissions: Emission scores [batch_size, seq_len, num_tags]
        tags: Gold tag sequence [batch_size, seq_len]
        crf: CRF layer
        mask_value: Value in tags to ignore (default: 0)

    Returns:
        Masked CRF loss
    """
    # Create mask: 1 for valid positions, 0 for masked
    mask = (tags != mask_value).float()

    # Zero out emissions for masked positions
    masked_emissions = emissions * mask.unsqueeze(-1)

    return crf(masked_emissions, tags, mask=mask)
