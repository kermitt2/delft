"""
PyTorch text classification models for DeLFT.

This module contains PyTorch implementations of the text classification architectures.

Model architectures implemented:
- lstm: LSTM classifier
- bidLstm_simple: Bidirectional LSTM
- cnn, cnn2, cnn3: CNN variants with GRU
- gru, gru_simple: GRU variants
- gru_lstm: Mixed GRU + LSTM
- lstm_cnn: LSTM + CNN
- dpcnn: Deep Pyramid CNN
- bert: Transformer-based
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class BaseTextClassifier(nn.Module):
    """
    Base class for DeLFT PyTorch text classification models.

    Supports two input modes:
    1. Index mode: inputs are word indices [batch, seq_len], embedding layer converts to vectors
    2. Vector mode: inputs are pre-computed embedding vectors [batch, seq_len, embed_size]

    Args:
        model_config: Model configuration
        training_config: Training configuration
        vocab_size: Size of word vocabulary (enables index mode)
    """

    name = "BaseTextClassifier"

    def __init__(self, model_config, training_config, vocab_size: int = None):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.nb_classes = len(model_config.list_classes)

        # Default parameters
        self.maxlen = getattr(model_config, "maxlen", 300)
        self.embed_size = getattr(model_config, "embed_size", 300)
        self.dropout_rate = getattr(model_config, "dropout_rate", 0.3)
        self.recurrent_units = getattr(model_config, "recurrent_units", 64)
        self.dense_size = getattr(model_config, "dense_size", 32)

        # Embedding layer for index-based inputs
        self.vocab_size = vocab_size
        self.use_embedding_layer = vocab_size is not None
        if self.use_embedding_layer:
            self.embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
        else:
            self.embedding = None

    def set_embedding_weights(self, weights: np.ndarray):
        """
        Set pretrained embedding weights.

        Args:
            weights: numpy array of shape [vocab_size, embed_size]
        """
        if self.embedding is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.from_numpy(weights))

    def embed_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Convert inputs to embeddings if needed.

        Args:
            inputs: Either word indices [batch, seq_len] or vectors [batch, seq_len, embed_size]

        Returns:
            Embedded tensor [batch, seq_len, embed_size]
        """
        if self.use_embedding_layer and inputs.dim() == 2:
            # Index input - lookup embeddings
            return self.embedding(inputs)
        else:
            # Already embedded
            return inputs

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs: Input embeddings [batch, seq_len, embed_size] or indices [batch, seq_len]
            labels: Optional labels for training [batch, num_classes]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        raise NotImplementedError

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get predictions.

        Args:
            inputs: Input embeddings or indices

        Returns:
            Prediction probabilities
        """
        with torch.no_grad():
            outputs = self.forward(inputs)
            return torch.sigmoid(outputs["logits"])

    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location: str = None):
        """Load model weights."""
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict)


class lstm(BaseTextClassifier):
    """
    LSTM classifier for text classification.
    """

    name = "lstm"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(self.recurrent_units * 2, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Global pooling
        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        logits = self.dense2(x)

        outputs = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss

        return outputs


class bidLstm_simple(BaseTextClassifier):
    """
    Bidirectional LSTM classifier.
    """

    name = "bidLstm_simple"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        recurrent_units = getattr(model_config, "recurrent_units", 300)

        self.bilstm = nn.LSTM(
            self.embed_size,
            recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(recurrent_units * 4, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.bilstm(x)
        x = self.dropout(x)

        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class cnn(BaseTextClassifier):
    """
    CNN + GRU classifier.
    """

    name = "cnn"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1 = nn.Conv1d(
            self.embed_size, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv2 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv3 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.pool = nn.MaxPool1d(2)
        self.gru = nn.GRU(self.recurrent_units, self.recurrent_units, batch_first=True)
        self.dense1 = nn.Linear(self.recurrent_units, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Embed inputs if needed, then transpose for conv
        x = self.embed_inputs(inputs)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Back to [batch, seq, features]
        x = x.transpose(1, 2)
        _, hidden = self.gru(x)
        x = hidden.squeeze(0)

        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class cnn2(BaseTextClassifier):
    """
    CNN variant with GRU.
    """

    name = "cnn2"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1 = nn.Conv1d(
            self.embed_size, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv2 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv3 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.gru = nn.GRU(
            self.recurrent_units,
            self.recurrent_units,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.dense1 = nn.Linear(self.recurrent_units, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.transpose(1, 2)
        _, hidden = self.gru(x)
        x = hidden.squeeze(0)

        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class cnn3(BaseTextClassifier):
    """
    GRU + CNN variant with pooling.
    """

    name = "cnn3"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.gru = nn.GRU(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.conv1 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv2 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv3 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.pool = nn.MaxPool1d(2)
        self.dense1 = nn.Linear(self.recurrent_units * 2, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.gru(x)

        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.transpose(1, 2)
        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class lstm_cnn(BaseTextClassifier):
    """
    LSTM + CNN classifier.
    """

    name = "lstm_cnn"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1 = nn.Conv1d(
            self.recurrent_units, self.recurrent_units, kernel_size=2, padding="same"
        )
        self.conv2 = nn.Conv1d(
            self.recurrent_units, 300, kernel_size=5, padding="valid"
        )
        self.dense1 = nn.Linear(300 * 2, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))

        x = x.transpose(1, 2)
        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class gru(BaseTextClassifier):
    """
    Two-layer Bidirectional GRU classifier.
    """

    name = "gru"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.bigru1 = nn.GRU(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.bigru2 = nn.GRU(
            self.recurrent_units * 2,
            self.recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(self.recurrent_units * 4, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.bigru1(x)
        x = self.dropout(x)
        x, _ = self.bigru2(x)

        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class gru_simple(BaseTextClassifier):
    """
    Single-layer Bidirectional GRU classifier.
    """

    name = "gru_simple"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.bigru = nn.GRU(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.dense1 = nn.Linear(self.recurrent_units * 4, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.bigru(x)

        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class gru_lstm(BaseTextClassifier):
    """
    Mixed Bidirectional GRU + LSTM classifier.
    """

    name = "gru_lstm"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        self.bigru = nn.GRU(
            self.embed_size,
            self.recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.bilstm = nn.LSTM(
            self.recurrent_units * 2,
            self.recurrent_units,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(self.recurrent_units * 4, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x, _ = self.bigru(x)
        x = self.dropout(x)
        x, _ = self.bilstm(x)

        x_max = x.max(dim=1)[0]
        x_avg = x.mean(dim=1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.relu(self.dense1(x))
        logits = self.dense2(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class dpcnn(BaseTextClassifier):
    """
    Deep Pyramid CNN classifier.
    """

    name = "dpcnn"

    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)

        # Initial projection
        self.initial_conv = nn.Conv1d(
            self.embed_size, self.recurrent_units, kernel_size=1
        )

        # Block 1
        self.conv1a = nn.Conv1d(
            self.recurrent_units,
            self.recurrent_units,
            kernel_size=2,
            stride=3,
            padding=0,
        )
        self.conv1b = nn.Conv1d(
            self.recurrent_units,
            self.recurrent_units,
            kernel_size=2,
            stride=3,
            padding=0,
        )

        # Block 2
        self.conv2a = nn.Conv1d(
            self.recurrent_units,
            self.recurrent_units,
            kernel_size=2,
            stride=3,
            padding=0,
        )
        self.conv2b = nn.Conv1d(
            self.recurrent_units,
            self.recurrent_units,
            kernel_size=2,
            stride=3,
            padding=0,
        )

        self.pool = nn.MaxPool1d(3, stride=2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Linear(self.recurrent_units, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_inputs(inputs)
        x = x.transpose(1, 2)
        x = self.initial_conv(x)

        # Block 1
        shortcut = x
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))

        # Adaptive residual (handle size mismatch)
        if x.size(2) < shortcut.size(2):
            shortcut = F.adaptive_max_pool1d(shortcut, x.size(2))
        x = x + shortcut
        x = self.pool(x)

        # Block 2
        shortcut = x
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))

        if x.size(2) < shortcut.size(2):
            shortcut = F.adaptive_max_pool1d(shortcut, x.size(2))
        elif x.size(2) > shortcut.size(2):
            x = F.adaptive_max_pool1d(x, shortcut.size(2))
        x = x + shortcut

        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        logits = self.dense(x)

        outputs = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss
        return outputs


class bert(BaseTextClassifier):
    """
    BERT/Transformer-based text classifier.
    """

    name = "bert"

    def __init__(
        self,
        model_config,
        training_config,
        load_pretrained_weights=True,
        local_path=None,
    ):
        super().__init__(model_config, training_config)

        from transformers import AutoModel, AutoConfig

        transformer_name = model_config.transformer_name or "bert-base-uncased"

        if local_path:
            self.transformer = AutoModel.from_pretrained(local_path)
        elif load_pretrained_weights:
            self.transformer = AutoModel.from_pretrained(transformer_name)
        else:
            config = AutoConfig.from_pretrained(transformer_name)
            self.transformer = AutoModel.from_config(config)

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, self.nb_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        result = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            result["loss"] = loss
        return result


# Model registry
MODEL_REGISTRY = {
    "lstm": lstm,
    "bidLstm_simple": bidLstm_simple,
    "cnn": cnn,
    "cnn2": cnn2,
    "cnn3": cnn3,
    "lstm_cnn": lstm_cnn,
    "gru": gru,
    "gru_simple": gru_simple,
    "gru_lstm": gru_lstm,
    "dpcnn": dpcnn,
    "bert": bert,
}

# List of available architectures for external use
architectures = list(MODEL_REGISTRY.keys())


def getModel(
    model_config, training_config, load_pretrained_weights=True, local_path=None
):
    """
    Get a model instance by architecture name.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        load_pretrained_weights: Whether to load pretrained transformer weights
        local_path: Local path for transformer models

    Returns:
        Model instance
    """
    architecture = model_config.architecture

    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[architecture]

    if architecture == "bert":
        return model_class(
            model_config, training_config, load_pretrained_weights, local_path
        )
    else:
        return model_class(model_config, training_config)
