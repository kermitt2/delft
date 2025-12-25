"""
PyTorch Trainer for DeLFT sequence labeling models.

Provides training loop, evaluation, and callbacks for PyTorch models.
"""

import os
import json
import logging
from typing import List, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.evaluation import classification_report
from delft.sequenceLabelling.preprocess import Preprocessor


# Default file names
DEFAULT_WEIGHT_FILE_NAME = "model_weights.pt"
CONFIG_FILE_NAME = "config.json"
PROCESSOR_FILE_NAME = "preprocessor.json"

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' depending on monitored metric
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Save model weights when validation metric improves.

    Args:
        filepath: Path to save model weights
        monitor: Metric to monitor
        mode: 'min' or 'max'
    """

    def __init__(self, filepath: str, monitor: str = "f1", mode: str = "max"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def __call__(self, model: nn.Module, score: float) -> bool:
        """Save model if score improved. Returns True if saved."""
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            return True

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self._save(model)
            return True

        return False

    def _save(self, model: nn.Module):
        """Save model weights."""
        torch.save(model.state_dict(), self.filepath)
        logger.info(f"Model saved to {self.filepath}")


class Trainer:
    """
    Trainer for PyTorch sequence labeling models.

    Args:
        model: PyTorch model
        config: Model configuration
        training_config: Training configuration
        preprocessor: Data preprocessor
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_path: Path to save checkpoints
        enable_wandb: Whether to log to Weights & Biases
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        training_config: TrainingConfig,
        preprocessor: Preprocessor = None,
        device: str = None,
        checkpoint_path: str = "",
        save_path: str = "",
        enable_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.training_config = training_config
        self.preprocessor = preprocessor

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.enable_wandb = enable_wandb

        # Initialize wandb if enabled
        if self.enable_wandb:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not available, disabling logging")
                self.enable_wandb = False

    def compile_model(self, train_size: int):
        """
        Set up optimizer and learning rate scheduler.

        Args:
            train_size: Number of training samples (for learning rate scheduling)
        """
        # Choose optimizer
        if self.config.transformer_name:
            # Use AdamW for transformer models
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=0.01,
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(), lr=self.training_config.learning_rate
            )

        # Learning rate scheduler
        num_training_steps = (
            train_size // self.training_config.batch_size
        ) * self.training_config.max_epoch

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

    def train(
        self, train_loader, valid_loader=None, callbacks: List[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader (optional)
            callbacks: List of callback functions

        Returns:
            Training history dictionary
        """
        # Compile model
        train_size = (
            len(train_loader.dataset)
            if hasattr(train_loader, "dataset")
            else len(train_loader) * self.training_config.batch_size
        )
        self.compile_model(train_size)

        # Set up callbacks
        early_stopping = EarlyStopping(patience=self.training_config.patience)
        
        # Use model-specific checkpoint filename to avoid conflicts between architectures
        checkpoint_filename = f"{self.config.model_name}_{DEFAULT_WEIGHT_FILE_NAME}"
        checkpoint_filepath = (
            os.path.join(self.checkpoint_path, checkpoint_filename)
            if self.checkpoint_path
            else checkpoint_filename
        )
        checkpoint = ModelCheckpoint(checkpoint_filepath)

        history = {"loss": [], "val_loss": [], "f1": [], "precision": [], "recall": []}

        best_f1 = 0.0

        for epoch in range(self.training_config.max_epoch):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            train_iter = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.training_config.max_epoch}"
            )

            for batch in train_iter:
                inputs, labels = batch

                # Move to device
                inputs = self._to_device(inputs)
                if labels is not None:
                    labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs["loss"]

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                train_iter.set_postfix({"loss": train_loss / num_batches})

            avg_train_loss = train_loss / num_batches
            history["loss"].append(avg_train_loss)

            # Validation phase
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                history["val_loss"].append(val_metrics.get("loss", 0))
                history["f1"].append(val_metrics["f1"])
                history["precision"].append(val_metrics["precision"])
                history["recall"].append(val_metrics["recall"])

                logger.info(
                    f"Epoch {epoch + 1}: loss={avg_train_loss:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, "
                    f"val_precision={val_metrics['precision']:.4f}, "
                    f"val_recall={val_metrics['recall']:.4f}"
                )

                # Update learning rate
                self.scheduler.step(val_metrics["f1"])

                # Model checkpoint
                if checkpoint(self.model, val_metrics["f1"]):
                    best_f1 = val_metrics["f1"]

                # Early stopping
                if early_stopping(val_metrics["f1"]):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Log to wandb
                if self.enable_wandb:
                    self.wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_f1": val_metrics["f1"],
                            "val_precision": val_metrics["precision"],
                            "val_recall": val_metrics["recall"],
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )
            else:
                logger.info(f"Epoch {epoch + 1}: loss={avg_train_loss:.4f}")

        # Load best model from model-specific checkpoint and cleanup
        best_model_path = checkpoint_filepath
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            # Remove checkpoint file - the wrapper will save the final model
            try:
                os.remove(best_model_path)
                logger.info(f"Removed temporary checkpoint: {best_model_path}")
            except OSError:
                pass  # Ignore if file can't be removed

        return history

    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Dictionary with metrics (f1, precision, recall, loss)
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch

                inputs = self._to_device(inputs)
                if labels is not None:
                    labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs, labels=labels)

                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1

                # Get predictions
                if hasattr(self.model, "decode"):
                    predictions = self.model.decode(inputs)
                else:
                    predictions = outputs["logits"].argmax(dim=-1).tolist()

                # Collect predictions and labels
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.tolist()

                if labels is not None:
                    labels_list = labels.tolist()

                    for pred, label in zip(predictions, labels_list):
                        # Filter padding (label == 0)
                        valid_pred = []
                        valid_label = []
                        for p, l in zip(pred, label):
                            if l != 0:  # Skip padding
                                valid_pred.append(p)
                                valid_label.append(l)
                        all_predictions.append(valid_pred)
                        all_labels.append(valid_label)

        # Convert indices back to labels
        if self.preprocessor:
            idx_to_label = {
                idx: label for label, idx in self.preprocessor.vocab_tag.items()
            }

            pred_labels = []
            true_labels = []

            for pred, label in zip(all_predictions, all_labels):
                pred_labels.append([idx_to_label.get(p, "O") for p in pred])
                true_labels.append([idx_to_label.get(l, "O") for l in label])

            # Calculate metrics
            report, evaluation = classification_report(
                true_labels, pred_labels, digits=4
            )

            # Use evaluation dictionary directly
            metrics = {
                "f1": evaluation["micro"]["f1"],
                "precision": evaluation["micro"]["precision"],
                "recall": evaluation["micro"]["recall"],
            }
        else:
            # Simple accuracy-based metrics if no preprocessor
            correct = sum(
                1
                for p, l in zip(all_predictions, all_labels)
                if len(p) == len(l) and all(pi == li for pi, li in zip(p, l))
            )
            total = len(all_predictions)
            metrics = {
                "f1": correct / total if total > 0 else 0,
                "precision": correct / total if total > 0 else 0,
                "recall": correct / total if total > 0 else 0,
            }

        if num_batches > 0:
            metrics["loss"] = total_loss / num_batches

        return metrics

    def _parse_report(self, report: str) -> Dict[str, float]:
        """Parse classification report to extract aggregate metrics."""
        # Default values
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        for line in report.split("\n"):
            if "micro avg" in line or "weighted avg" in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        metrics["precision"] = float(parts[-4])
                        metrics["recall"] = float(parts[-3])
                        metrics["f1"] = float(parts[-2])
                    except (ValueError, IndexError):
                        pass
                break

        return metrics

    def _to_device(self, inputs) -> Dict[str, torch.Tensor]:
        """Move inputs to device."""
        if isinstance(inputs, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            return [
                v.to(self.device) if isinstance(v, torch.Tensor) else v for v in inputs
            ]
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        return inputs

    def save_config(self, dir_path: str):
        """Save model and training configuration."""
        os.makedirs(dir_path, exist_ok=True)

        config_dict = {
            "model_config": self.config.__dict__,
            "training_config": self.training_config.__dict__
            if self.training_config
            else {},
        }

        config_path = os.path.join(dir_path, CONFIG_FILE_NAME)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)


class Scorer:
    """
    Callback for computing and logging metrics during training.

    Args:
        valid_loader: Validation data loader
        preprocessor: Data preprocessor
        evaluation: Whether this is final evaluation (more detailed)
    """

    def __init__(
        self, valid_loader, preprocessor: Preprocessor = None, evaluation: bool = False
    ):
        self.valid_loader = valid_loader
        self.preprocessor = preprocessor
        self.evaluation = evaluation

        self.f1 = -1.0
        self.precision = -1.0
        self.recall = -1.0
        self.report = None

    def on_epoch_end(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        """Compute metrics at end of epoch."""
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.valid_loader:
                inputs, labels = batch

                # Move to device
                if isinstance(inputs, dict):
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }

                # Get predictions
                if hasattr(model, "decode"):
                    predictions = model.decode(inputs)
                else:
                    outputs = model(inputs)
                    predictions = outputs["logits"].argmax(dim=-1).tolist()

                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.tolist()

                if labels is not None:
                    labels_list = labels.tolist()

                    for pred, label in zip(predictions, labels_list):
                        valid_pred = []
                        valid_label = []
                        for p, l in zip(pred, label):
                            if l != 0:
                                valid_pred.append(p)
                                valid_label.append(l)
                        all_predictions.append(valid_pred)
                        all_labels.append(valid_label)

        # Convert to labels and compute metrics
        if self.preprocessor:
            idx_to_label = {
                idx: label for label, idx in self.preprocessor.vocab_tag.items()
            }

            pred_labels = [
                [idx_to_label.get(p, "O") for p in pred] for pred in all_predictions
            ]
            true_labels = [
                [idx_to_label.get(l, "O") for l in label] for label in all_labels
            ]

            self.report, evaluation = classification_report(
                true_labels, pred_labels, digits=4
            )

            # Parse metrics
            if "micro" in evaluation:
                self.precision = evaluation["micro"]["precision"]
                self.recall = evaluation["micro"]["recall"]
                self.f1 = evaluation["micro"]["f1"]

        if self.evaluation:
            print(self.report)

        return {"f1": self.f1, "precision": self.precision, "recall": self.recall}


def to_wandb_table(report_as_map):
    """Convert evaluation report to wandb Table format.
    
    Args:
        report_as_map: Dictionary containing evaluation metrics from classification_report
        
    Returns:
        Tuple of (columns, data) for creating wandb.Table
    """
    columns = ["", "precision", "recall", "f1-score", "support"]
    data = []
    
    # Add each label's metrics
    if 'labels' in report_as_map:
        for label, metrics in report_as_map['labels'].items():
            row = [
                label,
                round(metrics['precision'], 4),
                round(metrics['recall'], 4),
                round(metrics['f1'], 4),
                int(metrics['support'])
            ]
            data.append(row)
    
    # Add micro average
    if 'micro' in report_as_map:
        micro = report_as_map['micro']
        micro_row = [
            "all (micro avg.)",
            round(micro['precision'], 4),
            round(micro['recall'], 4),
            round(micro['f1'], 4),
            int(micro['support'])
        ]
        data.append(micro_row)

    # Add macro average
    if 'macro' in report_as_map:
        macro = report_as_map['macro']
        macro_row = [
            "all (macro avg.)",
            round(macro['precision'], 4),
            round(macro['recall'], 4),
            round(macro['f1'], 4),
            int(macro['support'])
        ]
        data.append(macro_row)

    return columns, data
