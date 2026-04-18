"""
PyTorch Attention layer for DeLFT text classification and sequence models.

Ported from the Keras implementation in Attention.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    """
    Attention mechanism layer.

    Computes attention weights over a sequence and returns a weighted sum
    of the input features.

    Args:
        step_dim: Length of the sequence (for weight initialization)
        features_dim: Dimension of input features (set automatically if None)
        bias: Whether to use bias in attention computation
    """

    def __init__(
        self, step_dim: int, features_dim: Optional[int] = None, bias: bool = True
    ):
        super().__init__()
        self.step_dim = step_dim
        self.features_dim = features_dim
        self.use_bias = bias
        self.supports_masking = True

        # Will be initialized in first forward pass if features_dim not provided
        self.W = None
        self.b = None
        self._built = False

    def _build(self, features_dim: int):
        """Initialize weights based on input dimension."""
        self.features_dim = features_dim

        self.W = nn.Parameter(torch.empty(features_dim))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))
        self.W = nn.Parameter(self.W.squeeze(0))

        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(self.step_dim))

        self._built = True

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention mechanism.

        Args:
            x: Input tensor [batch_size, seq_len, features_dim]
            mask: Optional mask tensor [batch_size, seq_len]

        Returns:
            Attention-weighted output [batch_size, features_dim]
        """
        if not self._built:
            self._build(x.size(-1))

        batch_size, seq_len, features_dim = x.shape

        # Compute attention scores: e_ij = tanh(x_ij * W + b)
        # Reshape for batch multiplication
        x_reshaped = x.view(-1, features_dim)  # [batch * seq, features]

        eij = torch.matmul(x_reshaped, self.W)  # [batch * seq]
        eij = eij.view(batch_size, seq_len)  # [batch, seq]

        if self.use_bias:
            eij = eij + self.b[:seq_len]

        eij = torch.tanh(eij)

        # Compute attention weights using softmax
        a = torch.exp(eij)

        # Apply mask if provided
        if mask is not None:
            a = a * mask.float()

        # Normalize
        a = a / (a.sum(dim=1, keepdim=True) + 1e-8)

        # Compute weighted sum
        a = a.unsqueeze(-1)  # [batch, seq, 1]
        weighted_input = x * a  # [batch, seq, features]

        return weighted_input.sum(dim=1)  # [batch, features]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass.

        Args:
            query: Query tensor [batch, seq_q, d_model]
            key: Key tensor [batch, seq_k, d_model]
            value: Value tensor [batch, seq_v, d_model]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.W_o(context)

        return output


def dot_product(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for dot product operation used in attention layers.

    Args:
        x: Input tensor
        kernel: Weight tensor

    Returns:
        Dot product result
    """
    return torch.matmul(x, kernel.unsqueeze(-1)).squeeze(-1)
