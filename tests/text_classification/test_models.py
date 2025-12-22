"""
Tests for text classification models in delft/textClassification/models.py
"""

import logging
import numpy as np
import pytest
import torch

from delft.textClassification.models import (
    BaseTextClassifier,
    lstm,
    gru,
    cnn,
    dpcnn,
    getModel,
    MODEL_REGISTRY,
)

LOGGER = logging.getLogger(__name__)


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(
        self,
        list_classes=None,
        maxlen=50,
        embed_size=100,
        dropout_rate=0.1,
        recurrent_units=32,
        dense_size=16,
        transformer_name=None,
    ):
        self.list_classes = list_classes or ["class1", "class2", "class3"]
        self.maxlen = maxlen
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.recurrent_units = recurrent_units
        self.dense_size = dense_size
        self.transformer_name = transformer_name
        self.architecture = "lstm"


class MockTrainingConfig:
    """Mock training configuration for testing."""

    def __init__(self):
        pass


class TestBaseTextClassifier:
    """Tests for BaseTextClassifier base class."""

    def test_embedding_layer_created_when_vocab_size_provided(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        # Create a concrete subclass for testing
        class ConcreteClassifier(BaseTextClassifier):
            def forward(self, inputs, labels=None):
                x = self.embed_inputs(inputs)
                return {"logits": x.mean(dim=(1, 2)).unsqueeze(1).expand(-1, self.nb_classes)}

        model = ConcreteClassifier(config, training_config, vocab_size=1000)

        assert model.embedding is not None
        assert model.embedding.num_embeddings == 1000
        assert model.embedding.embedding_dim == config.embed_size

    def test_no_embedding_layer_when_vocab_size_not_provided(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        class ConcreteClassifier(BaseTextClassifier):
            def forward(self, inputs, labels=None):
                return {"logits": inputs.mean(dim=(1, 2)).unsqueeze(1).expand(-1, self.nb_classes)}

        model = ConcreteClassifier(config, training_config)

        assert model.embedding is None
        assert model.use_embedding_layer is False

    def test_set_embedding_weights(self):
        config = MockModelConfig(embed_size=50)
        training_config = MockTrainingConfig()

        class ConcreteClassifier(BaseTextClassifier):
            def forward(self, inputs, labels=None):
                return {"logits": torch.zeros(inputs.size(0), self.nb_classes)}

        model = ConcreteClassifier(config, training_config, vocab_size=100)

        # Set custom weights
        weights = np.random.randn(100, 50).astype(np.float32)
        model.set_embedding_weights(weights)

        assert torch.allclose(
            model.embedding.weight.data, torch.from_numpy(weights), atol=1e-6
        )

    def test_embed_inputs_with_indices(self):
        config = MockModelConfig(embed_size=50)
        training_config = MockTrainingConfig()

        class ConcreteClassifier(BaseTextClassifier):
            def forward(self, inputs, labels=None):
                x = self.embed_inputs(inputs)
                return {"logits": x.mean(dim=(1, 2)).unsqueeze(1).expand(-1, self.nb_classes)}

        model = ConcreteClassifier(config, training_config, vocab_size=100)

        # Test with index input [batch, seq_len]
        indices = torch.randint(0, 100, (4, 20))
        embedded = model.embed_inputs(indices)

        assert embedded.shape == (4, 20, 50)

    def test_embed_inputs_passthrough_for_vectors(self):
        config = MockModelConfig(embed_size=50)
        training_config = MockTrainingConfig()

        class ConcreteClassifier(BaseTextClassifier):
            def forward(self, inputs, labels=None):
                x = self.embed_inputs(inputs)
                return {"logits": x.mean(dim=(1, 2)).unsqueeze(1).expand(-1, self.nb_classes)}

        model = ConcreteClassifier(config, training_config)  # No vocab_size

        # Test with vector input [batch, seq_len, embed_size]
        vectors = torch.randn(4, 20, 50)
        result = model.embed_inputs(vectors)

        assert torch.equal(result, vectors)


class TestLstmModel:
    """Tests for lstm model."""

    def test_forward_with_vector_input(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        model = lstm(config, training_config)
        model.eval()

        # Vector input [batch, seq, embed]
        x = torch.randn(4, 50, 100)
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (4, 3)

    def test_forward_with_labels_returns_loss(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        model = lstm(config, training_config)
        model.train()

        x = torch.randn(4, 50, 100)
        labels = torch.zeros(4, 3)
        labels[:, 0] = 1  # One-hot

        output = model(x, labels=labels)

        assert "logits" in output
        assert "loss" in output
        assert output["loss"].item() > 0

    def test_forward_with_index_input(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        # Note: The lstm subclass would need to pass vocab_size through __init__
        # For now, we test that the base functionality works through a custom class
        class LstmWithEmbedding(lstm):
            def __init__(self, model_config, training_config, vocab_size=None):
                # Initialize base first without vocab_size
                BaseTextClassifier.__init__(self, model_config, training_config, vocab_size)
                # Then add lstm-specific layers
                self.lstm_layer = torch.nn.LSTM(
                    self.embed_size,
                    self.recurrent_units,
                    batch_first=True,
                    dropout=self.dropout_rate,
                )
                self.dropout = torch.nn.Dropout(self.dropout_rate)
                self.dense1 = torch.nn.Linear(self.recurrent_units * 2, self.dense_size)
                self.dense2 = torch.nn.Linear(self.dense_size, self.nb_classes)
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

            def forward(self, inputs, labels=None):
                x = self.embed_inputs(inputs)
                x, _ = self.lstm_layer(x)
                x = self.dropout(x)
                x_max = x.max(dim=1)[0]
                x_avg = x.mean(dim=1)
                x = torch.cat([x_max, x_avg], dim=1)
                x = torch.nn.functional.relu(self.dense1(x))
                x = self.dropout(x)
                logits = self.dense2(x)
                outputs = {"logits": logits}
                if labels is not None:
                    loss = self.loss_fn(logits, labels.float())
                    outputs["loss"] = loss
                return outputs

        model = LstmWithEmbedding(config, training_config, vocab_size=1000)
        model.eval()

        # Index input [batch, seq]
        x = torch.randint(0, 1000, (4, 50))
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (4, 3)


class TestGruModel:
    """Tests for gru model."""

    def test_forward(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        model = gru(config, training_config)
        model.eval()

        x = torch.randn(4, 50, 100)
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (4, 3)


class TestCnnModel:
    """Tests for cnn model."""

    def test_forward(self):
        config = MockModelConfig()
        training_config = MockTrainingConfig()

        model = cnn(config, training_config)
        model.eval()

        x = torch.randn(4, 50, 100)
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (4, 3)


class TestDpcnnModel:
    """Tests for dpcnn model."""

    def test_forward(self):
        # DPCNN requires longer sequences due to multiple pooling layers
        config = MockModelConfig(maxlen=100)  # Longer sequence
        training_config = MockTrainingConfig()

        model = dpcnn(config, training_config)
        model.eval()

        # Use longer sequence to avoid pooling issues
        x = torch.randn(4, 100, 100)
        output = model(x)

        assert "logits" in output
        assert output["logits"].shape == (4, 3)


class TestModelRegistry:
    """Tests for model registry and getModel function."""

    def test_registry_contains_expected_models(self):
        expected_models = [
            "lstm",
            "gru",
            "cnn",
            "dpcnn",
            "bert",
            "bidLstm_simple",
            "gru_simple",
            "gru_lstm",
        ]
        for model_name in expected_models:
            assert model_name in MODEL_REGISTRY

    def test_get_model_returns_correct_type(self):
        config = MockModelConfig()
        config.architecture = "lstm"
        training_config = MockTrainingConfig()

        model = getModel(config, training_config)

        assert isinstance(model, lstm)
