"""
PyTorch sequence labeling models for DeLFT.

This module contains PyTorch implementations of the sequence labeling architectures,
replacing the TensorFlow/Keras implementations.

Model architectures implemented:
- BidLSTM: BiLSTM + softmax
- BidLSTM_CRF: BiLSTM + CRF
- BidLSTM_ChainCRF: BiLSTM + ChainCRF
- BidLSTM_CNN: BiLSTM + CNN character encoder
- BidLSTM_CNN_CRF: BiLSTM + CNN + CRF
- BidGRU_CRF: BiGRU + CRF
- BidLSTM_CRF_FEATURES: BiLSTM + CRF + discrete features
- BidLSTM_ChainCRF_FEATURES: BiLSTM + ChainCRF + features
- BidLSTM_CRF_CASING: BiLSTM + CRF + casing features
"""

import os
import torch
import torch.nn as nn
import inspect
from typing import Optional, Dict, List

from delft.utilities.crf_pytorch import CRF, ChainCRF
from delft.sequenceLabelling.config import ModelConfig


class CharacterEncoder(nn.Module):
    """
    Character-level encoder using BiLSTM.

    Encodes character sequences for each token using a bidirectional LSTM.

    Args:
        char_vocab_size: Size of character vocabulary
        char_embedding_size: Dimension of character embeddings
        hidden_size: Size of LSTM hidden state
    """

    def __init__(
        self, char_vocab_size: int, char_embedding_size: int, hidden_size: int
    ):
        super().__init__()
        self.char_embeddings = nn.Embedding(
            char_vocab_size, char_embedding_size, padding_idx=0
        )
        self.bilstm = nn.LSTM(
            char_embedding_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_size = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode characters for each token.

        Args:
            x: Character indices [batch, seq_len, max_char_len]

        Returns:
            Character encodings [batch, seq_len, hidden*2]
        """
        batch_size, seq_len, max_char_len = x.shape

        # Reshape for character processing
        x = x.view(batch_size * seq_len, max_char_len)

        # Embed characters
        char_emb = self.char_embeddings(x)  # [batch*seq, max_char, emb_size]

        # Encode with BiLSTM
        _, (hidden, _) = self.bilstm(char_emb)

        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch*seq, hidden*2]

        # Reshape back
        output = hidden.view(batch_size, seq_len, -1)

        return output


class CharacterCNNEncoder(nn.Module):
    """
    Character-level encoder using CNN.

    Encodes character sequences using 1D convolution and max pooling.

    Args:
        char_vocab_size: Size of character vocabulary
        char_embedding_size: Dimension of character embeddings
        num_filters: Number of CNN filters
        kernel_size: Size of convolution kernel
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_size: int,
        num_filters: int = 30,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.char_embeddings = nn.Embedding(
            char_vocab_size, char_embedding_size, padding_idx=0
        )
        self.conv = nn.Conv1d(
            char_embedding_size, num_filters, kernel_size, padding="same"
        )
        self.output_size = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode characters using CNN.

        Args:
            x: Character indices [batch, seq_len, max_char_len]

        Returns:
            Character encodings [batch, seq_len, num_filters]
        """
        batch_size, seq_len, max_char_len = x.shape

        # Reshape for processing
        x = x.view(batch_size * seq_len, max_char_len)

        # Embed characters
        char_emb = self.char_embeddings(x)  # [batch*seq, max_char, emb_size]

        # CNN expects [batch, channels, length]
        char_emb = char_emb.transpose(1, 2)

        # Apply convolution
        conv_out = torch.tanh(self.conv(char_emb))  # [batch*seq, filters, max_char]

        # Global max pooling
        pooled = conv_out.max(dim=2)[0]  # [batch*seq, filters]

        # Reshape back
        output = pooled.view(batch_size, seq_len, -1)

        return output


class BaseSequenceLabeler(nn.Module):
    """
    Base class for DeLFT PyTorch sequence labeling models.

    Args:
        config: Model configuration
        ntags: Number of output tags/labels
    """

    name = "BaseSequenceLabeler"
    use_crf = False
    use_chain_crf = False

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__()
        self.config = config
        self.ntags = ntags

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs: Dictionary of input tensors
            labels: Optional labels for training

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        raise NotImplementedError

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """
        Decode best tag sequence.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            List of tag sequences
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location: str = None):
        """Load model weights."""
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict)


class BidLSTM(BaseSequenceLabeler):
    """
    Bidirectional LSTM with softmax output for sequence labeling.

    Architecture:
    - Word embeddings (provided as input)
    - Character BiLSTM encoder
    - BiLSTM encoder
    - Dense + Softmax
    """

    name = "BidLSTM"

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Input size: word embeddings + character encodings
        input_size = config.word_embedding_size + self.char_encoder.output_size

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Output layer
        self.classifier = nn.Linear(config.num_word_lstm_units * 2, ntags)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]  # [batch, seq, emb_size]
        char_input = inputs["char_input"]  # [batch, seq, max_char]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Concatenate word and character embeddings
        x = torch.cat([word_emb, char_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Classification
        logits = self.classifier(lstm_out)

        outputs = {"logits": logits}

        if labels is not None:
            # Reshape for cross-entropy loss
            loss = self.loss_fn(logits.view(-1, self.ntags), labels.view(-1))
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using argmax."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = outputs["logits"].argmax(dim=-1)
        return predictions.tolist()


class BidLSTM_CRF(BaseSequenceLabeler):
    """
    Bidirectional LSTM with CRF output for sequence labeling.

    Architecture:
    - Word embeddings (provided as input)
    - Character BiLSTM encoder
    - BiLSTM encoder
    - Dense + CRF
    """

    name = "BidLSTM_CRF"
    use_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Input size
        input_size = config.word_embedding_size + self.char_encoder.output_size

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF dense layer
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )
        self.linear = nn.Linear(config.num_word_lstm_units, ntags)

        # CRF layer
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]

        # Get sequence lengths for masking
        lengths = inputs.get("length", None)
        if lengths is not None:
            # Create mask from lengths
            batch_size, seq_len = word_emb.shape[:2]
            mask = torch.arange(seq_len, device=word_emb.device).expand(
                batch_size, seq_len
            )
            mask = mask < lengths.squeeze(-1).unsqueeze(-1)
            mask = mask.float()
        else:
            mask = None

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Concatenate
        x = torch.cat([word_emb, char_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense
        x = torch.tanh(self.dense(lstm_out))
        emissions = self.linear(x)

        outputs = {"logits": emissions}

        if labels is not None:
            # CRF loss
            loss = self.crf(emissions, labels, mask=mask)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)

            # Get mask
            lengths = inputs.get("length", None)
            if lengths is not None:
                batch_size, seq_len = outputs["logits"].shape[:2]
                mask = torch.arange(seq_len, device=outputs["logits"].device).expand(
                    batch_size, seq_len
                )
                mask = mask < lengths.squeeze(-1).unsqueeze(-1)
                mask = mask.float()
            else:
                mask = None

            predictions = self.crf.decode(outputs["logits"], mask=mask)
        return predictions


class BidLSTM_ChainCRF(BaseSequenceLabeler):
    """
    Bidirectional LSTM with ChainCRF output.

    Uses alternative CRF implementation with explicit boundary handling.
    """

    name = "BidLSTM_ChainCRF"
    use_crf = True
    use_chain_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Input size
        input_size = config.word_embedding_size + self.char_encoder.output_size

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF layers
        self.dense1 = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )
        self.dense2 = nn.Linear(config.num_word_lstm_units, ntags)

        # ChainCRF layer
        self.crf = ChainCRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Concatenate
        x = torch.cat([word_emb, char_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense layers
        x = torch.tanh(self.dense1(lstm_out))
        emissions = self.dense2(x)

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BidLSTM_CNN(BaseSequenceLabeler):
    """
    Bidirectional LSTM with CNN character encoder.

    Architecture:
    - Word embeddings + CNN character encoding + casing features
    - BiLSTM encoder
    - Dense + Softmax
    """

    name = "BidLSTM_CNN"

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character CNN encoder
        self.char_encoder = CharacterCNNEncoder(
            config.char_vocab_size, config.char_embedding_size
        )

        # Casing embedding
        self.casing_embedding = nn.Embedding(
            config.case_vocab_size, config.case_embedding_size, padding_idx=0
        )

        # Input size
        input_size = (
            config.word_embedding_size
            + self.char_encoder.output_size
            + config.case_embedding_size
        )

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Output layer
        self.classifier = nn.Linear(config.num_word_lstm_units * 2, ntags)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]
        casing_input = inputs["casing_input"]

        # Encode characters with CNN
        char_encoded = self.char_encoder(char_input)

        # Embed casing
        casing_emb = self.casing_embedding(casing_input)
        casing_emb = self.dropout(casing_emb)

        # Concatenate all inputs
        x = torch.cat([word_emb, char_encoded, casing_emb], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Classification
        logits = self.classifier(lstm_out)

        outputs = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.ntags), labels.view(-1))
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using argmax."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = outputs["logits"].argmax(dim=-1)
        return predictions.tolist()


class BidLSTM_CNN_CRF(BaseSequenceLabeler):
    """
    Bidirectional LSTM-CNN with CRF output.
    """

    name = "BidLSTM_CNN_CRF"
    use_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character CNN encoder
        self.char_encoder = CharacterCNNEncoder(
            config.char_vocab_size, config.char_embedding_size
        )

        # Input size (word emb + char encoding, casing is optional input but not used in emission)
        input_size = config.word_embedding_size + self.char_encoder.output_size

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF dense
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )

        # CRF
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Concatenate
        x = torch.cat([word_emb, char_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense
        emissions = torch.tanh(self.dense(lstm_out))

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BidGRU_CRF(BaseSequenceLabeler):
    """
    Bidirectional GRU with CRF output.

    Uses GRU instead of LSTM.
    """

    name = "BidGRU_CRF"
    use_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Input size
        input_size = config.word_embedding_size + self.char_encoder.output_size

        # Stack of 2 BiGRU layers
        self.bigru1 = nn.GRU(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )
        self.bigru2 = nn.GRU(
            config.num_word_lstm_units * 2,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF dense
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )

        # CRF
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Concatenate
        x = torch.cat([word_emb, char_encoded], dim=-1)
        x = self.dropout(x)

        # BiGRU layers
        gru_out, _ = self.bigru1(x)
        gru_out = self.dropout(gru_out)
        gru_out, _ = self.bigru2(gru_out)

        # Dense
        emissions = torch.tanh(self.dense(gru_out))

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BidLSTM_CRF_FEATURES(BaseSequenceLabeler):
    """
    BiLSTM + CRF with additional discrete features.

    Incorporates layout or other categorical features alongside word/char embeddings.
    """

    name = "BidLSTM_CRF_FEATURES"
    use_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Features embedding
        num_features = len(config.features_indices) if config.features_indices else 1
        features_vocab_size = config.features_vocabulary_size * num_features + 1
        self.features_embedding = nn.Embedding(
            features_vocab_size, config.features_embedding_size, padding_idx=0
        )
        self.features_lstm = nn.LSTM(
            config.features_embedding_size,
            config.features_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # Input size
        input_size = (
            config.word_embedding_size
            + self.char_encoder.output_size
            + config.features_lstm_units * 2
        )

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF dense
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )
        self.dense2 = nn.Linear(config.num_word_lstm_units, ntags)

        # CRF
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]
        features_input = inputs["features_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Process features
        batch_size, seq_len = features_input.shape[:2]
        features_emb = self.features_embedding(features_input)

        # If features have multiple dimensions, process with LSTM
        if len(features_emb.shape) == 4:
            # [batch, seq, num_features, emb] -> [batch*seq, num_features, emb]
            features_emb_flat = features_emb.view(
                -1, features_emb.shape[2], features_emb.shape[3]
            )
            _, (hidden, _) = self.features_lstm(features_emb_flat)
            features_encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
            features_encoded = features_encoded.view(batch_size, seq_len, -1)
        else:
            # Simple embedding
            features_encoded = features_emb.view(batch_size, seq_len, -1)

        features_encoded = self.dropout(features_encoded)

        # Concatenate all inputs
        x = torch.cat([word_emb, char_encoded, features_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense layers
        x = torch.tanh(self.dense(lstm_out))
        emissions = self.dense2(x)

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BidLSTM_ChainCRF_FEATURES(BidLSTM_CRF_FEATURES):
    """BiLSTM + ChainCRF with features."""

    name = "BidLSTM_ChainCRF_FEATURES"
    use_chain_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)
        # Replace CRF with ChainCRF
        self.crf = ChainCRF(ntags)
        # Add extra dense layer before CRF (matching Keras architecture)
        self.dense2 = nn.Linear(config.num_word_lstm_units, ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]
        features_input = inputs["features_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Process features
        batch_size, seq_len = features_input.shape[:2]
        features_emb = self.features_embedding(features_input)

        if len(features_emb.shape) == 4:
            features_emb_flat = features_emb.view(
                -1, features_emb.shape[2], features_emb.shape[3]
            )
            _, (hidden, _) = self.features_lstm(features_emb_flat)
            features_encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
            features_encoded = features_encoded.view(batch_size, seq_len, -1)
        else:
            features_encoded = features_emb.view(batch_size, seq_len, -1)

        features_encoded = self.dropout(features_encoded)

        # Concatenate
        x = torch.cat([word_emb, char_encoded, features_encoded], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense layers
        x = torch.tanh(self.dense(lstm_out))
        emissions = self.dense2(x)

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BidLSTM_CRF_CASING(BaseSequenceLabeler):
    """
    BiLSTM + CRF with casing features.
    """

    name = "BidLSTM_CRF_CASING"
    use_crf = True

    def __init__(self, config: ModelConfig, ntags: int = None):
        super().__init__(config, ntags)

        # Character encoder
        self.char_encoder = CharacterEncoder(
            config.char_vocab_size,
            config.char_embedding_size,
            config.num_char_lstm_units,
        )

        # Casing embedding
        self.casing_embedding = nn.Embedding(
            config.case_vocab_size, config.case_embedding_size, padding_idx=0
        )

        # Input size
        input_size = (
            config.word_embedding_size
            + self.char_encoder.output_size
            + config.case_embedding_size
        )

        # Main BiLSTM
        self.bilstm = nn.LSTM(
            input_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout if config.recurrent_dropout else 0,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Pre-CRF dense
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )

        # CRF
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        word_emb = inputs["word_input"]
        char_input = inputs["char_input"]
        casing_input = inputs["casing_input"]

        # Encode characters
        char_encoded = self.char_encoder(char_input)

        # Embed casing
        casing_emb = self.casing_embedding(casing_input)
        casing_emb = self.dropout(casing_emb)

        # Concatenate
        x = torch.cat([word_emb, char_encoded, casing_emb], dim=-1)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        # Dense
        emissions = torch.tanh(self.dense(lstm_out))

        outputs = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            outputs["loss"] = loss

        return outputs

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


# ============================================================================
# BERT/Transformer-based Models
# ============================================================================


class BERT(BaseSequenceLabeler):
    """
    BERT/Transformer-based sequence labeler with softmax output.

    Uses HuggingFace transformers for the backbone.

    Args:
        config: Model configuration (must include transformer_name)
        ntags: Number of output tags
        load_pretrained_weights: Whether to load pretrained transformer weights
        local_path: Local path to load transformer from
    """

    name = "BERT"

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags)

        from transformers import AutoModel

        transformer_name = config.transformer_name or "bert-base-uncased"

        # Load transformer
        # Note: local_path is for loading a locally-stored transformer (not from HuggingFace)
        # When load_pretrained_weights=False, we load from transformer_name since fine-tuned
        # weights will be loaded separately by the wrapper via load_state_dict()
        if load_pretrained_weights:
            if local_path and os.path.exists(os.path.join(local_path, "config.json")):
                # Check if local_path is a valid transformer directory
                self.transformer = AutoModel.from_pretrained(local_path)
            else:
                self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            # Load pretrained transformer (weights will be replaced by load_state_dict later)
            self.transformer = AutoModel.from_pretrained(transformer_name)

        # Check if transformer accepts token_type_ids
        forward_signature = inspect.signature(self.transformer.forward)
        self.accepts_token_type_ids = "token_type_ids" in forward_signature.parameters

        # Get hidden size from transformer
        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, ntags)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        # Transformer forward
        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        outputs = self.transformer(**transformer_args)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        result = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.ntags), labels.view(-1))
            result["loss"] = loss

        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using argmax."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = outputs["logits"].argmax(dim=-1)
        return predictions.tolist()


class BERT_CRF(BaseSequenceLabeler):
    """
    BERT/Transformer-based sequence labeler with CRF output.
    """

    name = "BERT_CRF"
    use_crf = True

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags)

        from transformers import AutoModel

        transformer_name = config.transformer_name or "bert-base-uncased"

        if load_pretrained_weights:
            if local_path and os.path.exists(os.path.join(local_path, "config.json")):
                self.transformer = AutoModel.from_pretrained(local_path)
            else:
                self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            self.transformer = AutoModel.from_pretrained(transformer_name)

        # Check if transformer accepts token_type_ids
        forward_signature = inspect.signature(self.transformer.forward)
        self.accepts_token_type_ids = "token_type_ids" in forward_signature.parameters

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, ntags)
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        outputs = self.transformer(**transformer_args)
        x = self.dropout(outputs.last_hidden_state)
        emissions = self.linear(x)

        # Create mask for CRF (ignore padding and special tokens)
        if attention_mask is not None:
            mask = attention_mask.float()
        else:
            mask = None

        result = {"logits": emissions}

        if labels is not None:
            # Mask out special token labels (typically 0)
            loss = self.crf(emissions, labels, mask=mask)
            result["loss"] = loss

        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            attention_mask = inputs.get("attention_mask", None)
            mask = attention_mask.float() if attention_mask is not None else None
            predictions = self.crf.decode(outputs["logits"], mask=mask)
        return predictions


class BERT_ChainCRF(BaseSequenceLabeler):
    """
    BERT/Transformer-based sequence labeler with ChainCRF output.
    """

    name = "BERT_ChainCRF"
    use_crf = True
    use_chain_crf = True

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags)

        from transformers import AutoModel

        transformer_name = config.transformer_name or "bert-base-uncased"

        if load_pretrained_weights:
            if local_path and os.path.exists(os.path.join(local_path, "config.json")):
                self.transformer = AutoModel.from_pretrained(local_path)
            else:
                self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            self.transformer = AutoModel.from_pretrained(transformer_name)

        # Check if transformer accepts token_type_ids
        forward_signature = inspect.signature(self.transformer.forward)
        self.accepts_token_type_ids = "token_type_ids" in forward_signature.parameters

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, ntags)
        self.crf = ChainCRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        outputs = self.transformer(**transformer_args)
        x = self.dropout(outputs.last_hidden_state)
        emissions = self.dense(x)

        result = {"logits": emissions}

        if labels is not None:
            loss = self.crf(emissions, labels)
            result["loss"] = loss

        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


class BERT_FEATURES(BaseSequenceLabeler):
    """
    BERT + softmax with additional discrete features.
    """

    name = "BERT_FEATURES"

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags)

        from transformers import AutoModel

        transformer_name = config.transformer_name or "bert-base-uncased"

        if load_pretrained_weights:
            if local_path and os.path.exists(os.path.join(local_path, "config.json")):
                self.transformer = AutoModel.from_pretrained(local_path)
            else:
                self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            self.transformer = AutoModel.from_pretrained(transformer_name)

        # Check if transformer accepts token_type_ids
        forward_signature = inspect.signature(self.transformer.forward)
        self.accepts_token_type_ids = "token_type_ids" in forward_signature.parameters

        hidden_size = self.transformer.config.hidden_size

        # Features embedding
        num_features = len(config.features_indices) if config.features_indices else 1
        features_vocab_size = config.features_vocabulary_size * num_features + 1
        self.features_embedding = nn.Embedding(
            features_vocab_size, config.features_embedding_size, padding_idx=0
        )
        self.features_lstm = nn.LSTM(
            config.features_embedding_size,
            config.features_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # Combined processing
        combined_size = hidden_size + config.features_lstm_units * 2

        self.bilstm = nn.LSTM(
            combined_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_word_lstm_units * 2, ntags)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        features_input = inputs["features_input"]

        # Transformer
        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        transformer_out = self.transformer(**transformer_args)
        text_emb = self.dropout(transformer_out.last_hidden_state)

        # Features
        batch_size, seq_len = features_input.shape[:2]
        features_emb = self.features_embedding(features_input)

        if len(features_emb.shape) == 4:
            features_emb_flat = features_emb.view(
                -1, features_emb.shape[2], features_emb.shape[3]
            )
            _, (hidden, _) = self.features_lstm(features_emb_flat)
            features_encoded = torch.cat([hidden[0], hidden[1]], dim=-1)
            features_encoded = features_encoded.view(batch_size, seq_len, -1)
        else:
            features_encoded = features_emb.view(batch_size, seq_len, -1)

        features_encoded = self.dropout(features_encoded)

        # Combine
        x = torch.cat([text_emb, features_encoded], dim=-1)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        logits = self.classifier(lstm_out)

        result = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.ntags), labels.view(-1))
            result["loss"] = loss

        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using argmax."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = outputs["logits"].argmax(dim=-1)
        return predictions.tolist()


class BERT_CRF_FEATURES(BaseSequenceLabeler):
    """
    BERT + CRF with additional discrete features.
    """

    name = "BERT_CRF_FEATURES"
    use_crf = True

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags)

        from transformers import AutoModel

        transformer_name = config.transformer_name or "bert-base-uncased"

        if load_pretrained_weights:
            if local_path and os.path.exists(os.path.join(local_path, "config.json")):
                self.transformer = AutoModel.from_pretrained(local_path)
            else:
                self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            self.transformer = AutoModel.from_pretrained(transformer_name)

        # Check if transformer accepts token_type_ids
        forward_signature = inspect.signature(self.transformer.forward)
        self.accepts_token_type_ids = "token_type_ids" in forward_signature.parameters

        hidden_size = self.transformer.config.hidden_size

        # Features
        num_features = len(config.features_indices) if config.features_indices else 1
        features_vocab_size = config.features_vocabulary_size * num_features + 1
        self.features_embedding = nn.Embedding(
            features_vocab_size, config.features_embedding_size, padding_idx=0
        )
        self.features_lstm = nn.LSTM(
            config.features_embedding_size,
            config.features_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        combined_size = hidden_size + config.features_lstm_units * 2

        self.bilstm = nn.LSTM(
            combined_size,
            config.num_word_lstm_units,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(
            config.num_word_lstm_units * 2, config.num_word_lstm_units
        )
        self.crf = CRF(ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        features_input = inputs["features_input"]

        # Transformer
        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        transformer_out = self.transformer(**transformer_args)
        text_emb = self.dropout(transformer_out.last_hidden_state)

        # Features
        batch_size, seq_len = features_input.shape[:2]
        features_emb = self.features_embedding(features_input)
        if len(features_emb.shape) == 4:
            features_emb_flat = features_emb.view(
                -1, features_emb.shape[2], features_emb.shape[3]
            )
            _, (hidden, _) = self.features_lstm(features_emb_flat)
            features_encoded = torch.cat([hidden[0], hidden[1]], dim=-1).view(
                batch_size, seq_len, -1
            )
        else:
            features_encoded = features_emb.view(batch_size, seq_len, -1)
        features_encoded = self.dropout(features_encoded)

        # Combine
        x = torch.cat([text_emb, features_encoded], dim=-1)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)
        emissions = torch.tanh(self.dense(lstm_out))

        result = {"logits": emissions}

        if labels is not None:
            mask = attention_mask.float() if attention_mask is not None else None
            loss = self.crf(emissions, labels, mask=mask)
            result["loss"] = loss

        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> List[List[int]]:
        """Decode using Viterbi."""
        with torch.no_grad():
            outputs = self.forward(inputs)
            attention_mask = inputs.get("attention_mask", None)
            mask = attention_mask.float() if attention_mask is not None else None
            predictions = self.crf.decode(outputs["logits"], mask=mask)
        return predictions


class BERT_ChainCRF_FEATURES(BERT_CRF_FEATURES):
    """BERT + ChainCRF with features."""

    name = "BERT_ChainCRF_FEATURES"
    use_chain_crf = True

    def __init__(
        self,
        config: ModelConfig,
        ntags: int = None,
        load_pretrained_weights: bool = True,
        local_path: str = None,
    ):
        super().__init__(config, ntags, load_pretrained_weights, local_path)
        self.crf = ChainCRF(ntags)
        self.dense2 = nn.Linear(config.num_word_lstm_units, ntags)

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        features_input = inputs["features_input"]

        transformer_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids:
            transformer_args["token_type_ids"] = token_type_ids

        transformer_out = self.transformer(**transformer_args)
        text_emb = self.dropout(transformer_out.last_hidden_state)

        batch_size, seq_len = features_input.shape[:2]
        features_emb = self.features_embedding(features_input)
        if len(features_emb.shape) == 4:
            features_emb_flat = features_emb.view(
                -1, features_emb.shape[2], features_emb.shape[3]
            )
            _, (hidden, _) = self.features_lstm(features_emb_flat)
            features_encoded = torch.cat([hidden[0], hidden[1]], dim=-1).view(
                batch_size, seq_len, -1
            )
        else:
            features_encoded = features_emb.view(batch_size, seq_len, -1)
        features_encoded = self.dropout(features_encoded)

        x = torch.cat([text_emb, features_encoded], dim=-1)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)
        x = torch.tanh(self.dense(lstm_out))
        emissions = self.dense2(x)

        result = {"logits": emissions}
        if labels is not None:
            loss = self.crf(emissions, labels)
            result["loss"] = loss
        return result

    def decode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = self.crf.decode(outputs["logits"])
        return predictions


# Model registry
MODEL_REGISTRY = {
    "BidLSTM": BidLSTM,
    "BidLSTM_CRF": BidLSTM_CRF,
    "BidLSTM_ChainCRF": BidLSTM_ChainCRF,
    "BidLSTM_CNN": BidLSTM_CNN,
    "BidLSTM_CNN_CRF": BidLSTM_CNN_CRF,
    "BidGRU_CRF": BidGRU_CRF,
    "BidLSTM_CRF_FEATURES": BidLSTM_CRF_FEATURES,
    "BidLSTM_ChainCRF_FEATURES": BidLSTM_ChainCRF_FEATURES,
    "BidLSTM_CRF_CASING": BidLSTM_CRF_CASING,
    # BERT models
    "BERT": BERT,
    "BERT_CRF": BERT_CRF,
    "BERT_ChainCRF": BERT_ChainCRF,
    "BERT_FEATURES": BERT_FEATURES,
    "BERT_CRF_FEATURES": BERT_CRF_FEATURES,
    "BERT_ChainCRF_FEATURES": BERT_ChainCRF_FEATURES,
}


def get_model(
    config: ModelConfig,
    ntags: int,
    load_pretrained_weights: bool = True,
    local_path: str = None,
) -> BaseSequenceLabeler:
    """
    Get a model instance by architecture name.

    Args:
        config: Model configuration with architecture name
        ntags: Number of output tags
        load_pretrained_weights: Whether to load pretrained transformer weights
        local_path: Local path for transformer models

    Returns:
        Model instance
    """
    architecture = config.architecture

    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[architecture]

    # BERT models take additional arguments
    if architecture.startswith("BERT"):
        return model_class(config, ntags, load_pretrained_weights, local_path)
    else:
        return model_class(config, ntags)
