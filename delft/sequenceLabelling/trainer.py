"""
PyTorch Trainer for DeLFT sequence labeling models.

Provides training loop, evaluation, and callbacks for PyTorch models.
"""

import json
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.evaluation import classification_report
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.sequenceLabelling.windows import join_scored_windows
from delft.utilities.Utilities import pick_device

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
        # Handle DDP wrapped models - get underlying model
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, self.filepath)
        logger.info(f"Model saved to {self.filepath}")


def unique_checkpoint_path(directory: str, model_name: str, suffix: str) -> str:
    """
    A file of its own for the best weights of one training, in ``directory``.

    Named after the model alone, the file was shared by the trainings of a same model
    running at the same time from a same directory, a sweep over hyper-parameters for
    instance: each wrote over the weights of the others, and at its end loaded back
    whatever was there, from another training as likely as not, then deleted it.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle, path = tempfile.mkstemp(prefix=f"{model_name}_", suffix=f"_{suffix}", dir=directory or ".")
    os.close(handle)
    return path


def remove_file(path: Optional[str]):
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


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
        distributed: Whether to use DistributedDataParallel
        local_rank: Local GPU rank for distributed training
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
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.config = config
        self.training_config = training_config
        self.preprocessor = preprocessor
        self.distributed = distributed
        self.local_rank = local_rank

        # Set device
        if device is None:
            self.device = pick_device()
        else:
            self.device = torch.device(device)

        # Move model to device first
        model.to(self.device)

        # Wrap model with DDP if distributed training
        if self.distributed:
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            self._unwrapped_model = model  # Keep reference for saving
        else:
            self.model = model
            self._unwrapped_model = model

        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.enable_wandb = enable_wandb

        # Only enable wandb on main process for distributed training
        if self.distributed:
            from delft.utilities.distributed import is_main_process

            if not is_main_process():
                self.enable_wandb = False

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
            self.optimizer = Adam(self.model.parameters(), lr=self.training_config.learning_rate)

        # Learning rate scheduler
        # num_training_steps = (train_size // self.training_config.batch_size) * self.training_config.max_epoch

        # lr_decay is the factor the learning rate is multiplied by when the validation F1
        # has not improved for two epochs
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=self.training_config.lr_decay, patience=2)

    def train(self, train_loader, valid_loader=None, callbacks: List[Callable] = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader (optional)
            callbacks: functions called at the end of every epoch as ``callback(epoch, logs)``,
                with the number of the epoch, from 1, and its metrics: "loss", and with a
                validation set "val_loss", "f1", "precision", "recall" and "learning_rate"

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

        early_stopping = EarlyStopping(patience=self.training_config.patience)

        # files are written by the main process only
        is_main = True
        if self.distributed:
            from delft.utilities.distributed import is_main_process

            is_main = is_main_process()

        checkpoint_filepath = None
        if is_main and valid_loader is not None:
            checkpoint_filepath = unique_checkpoint_path(
                self.checkpoint_path, self.config.model_name, DEFAULT_WEIGHT_FILE_NAME
            )
        checkpoint = ModelCheckpoint(checkpoint_filepath)

        try:
            return self._run_epochs(train_loader, valid_loader, callbacks or [], early_stopping, checkpoint, is_main)
        finally:
            # the best weights are a temporary file: the wrapper saves the model
            remove_file(checkpoint_filepath)

    def _run_epochs(self, train_loader, valid_loader, callbacks, early_stopping, checkpoint, is_main):
        checkpoint_filepath = checkpoint.filepath
        history = {"loss": [], "val_loss": [], "f1": [], "precision": [], "recall": []}

        best_f1 = 0.0

        for epoch in range(self.training_config.max_epoch):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.training_config.max_epoch}")

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
                if self.training_config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.clip_gradients)

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

                print(
                    f"Epoch {epoch + 1}: loss={avg_train_loss:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, "
                    f"val_precision={val_metrics['precision']:.4f}, "
                    f"val_recall={val_metrics['recall']:.4f}"
                )

                # Update learning rate
                self.scheduler.step(val_metrics["f1"])

                # Model checkpoint - only on main process
                if is_main and checkpoint(self._unwrapped_model, val_metrics["f1"]):
                    best_f1 = val_metrics["f1"]

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
                            "best_f1": best_f1,
                        }
                    )
            else:
                print(f"Epoch {epoch + 1}: loss={avg_train_loss:.4f}")

            logs = {"loss": avg_train_loss}
            if valid_loader is not None:
                logs.update(
                    val_loss=val_metrics.get("loss", 0),
                    f1=val_metrics["f1"],
                    precision=val_metrics["precision"],
                    recall=val_metrics["recall"],
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                )
            if is_main:
                self._keep_epoch_checkpoint(epoch + 1)
            for callback in callbacks:
                callback(epoch + 1, logs)

            # Early stopping
            if valid_loader is not None and self.training_config.early_stop and early_stopping(val_metrics["f1"]):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load the best weights back (only on main process); train() removes the file.
        # It is empty when no epoch was validated.
        if is_main and checkpoint.best_score is not None and os.path.getsize(checkpoint_filepath) > 0:
            self._unwrapped_model.load_state_dict(torch.load(checkpoint_filepath, map_location=self.device))

        # Synchronize all processes after loading
        if self.distributed:
            from delft.utilities.distributed import barrier

            barrier()

        return history

    def _keep_epoch_checkpoint(self, epoch: int):
        """
        With ``max_checkpoints_to_keep`` above 0, save the weights of every epoch in the
        checkpoint directory as ``<model name>-epoch<N>.pt`` and keep those of the last
        epochs only. They stay there after the training.
        """
        nb_to_keep = self.training_config.max_checkpoints_to_keep or 0
        if nb_to_keep <= 0:
            return
        directory = self.checkpoint_path or "."
        os.makedirs(directory, exist_ok=True)

        def path(n):
            return os.path.join(directory, f"{self.config.model_name}-epoch{n}.pt")

        torch.save(self._unwrapped_model.state_dict(), path(epoch))
        remove_file(path(epoch - nb_to_keep) if epoch > nb_to_keep else None)

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

        all_predictions, all_labels = join_scored_windows(data_loader, all_predictions, all_labels)

        # Convert indices back to labels
        if self.preprocessor:
            idx_to_label = {idx: label for label, idx in self.preprocessor.vocab_tag.items()}

            pred_labels = []
            true_labels = []

            for pred, label in zip(all_predictions, all_labels):
                pred_labels.append([idx_to_label.get(p, "O") for p in pred])
                true_labels.append([idx_to_label.get(l, "O") for l in label])

            # Calculate metrics
            report, evaluation = classification_report(true_labels, pred_labels, digits=4)

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
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in inputs]
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        return inputs

    def save_config(self, dir_path: str):
        """Save model and training configuration."""
        os.makedirs(dir_path, exist_ok=True)

        config_dict = {
            "model_config": self.config.__dict__,
            "training_config": self.training_config.__dict__ if self.training_config else {},
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

    def __init__(self, valid_loader, preprocessor: Preprocessor = None, evaluation: bool = False):
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
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Get predictions
                if hasattr(model, "decode"):
                    predictions = model.decode(inputs)
                else:
                    outputs = model(inputs)
                    predictions = outputs["logits"].argmax(dim=-1).tolist()

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

        all_predictions, all_labels = join_scored_windows(self.valid_loader, all_predictions, all_labels)

        # Convert to labels and compute metrics
        if self.preprocessor:
            idx_to_label = {idx: label for label, idx in self.preprocessor.vocab_tag.items()}

            pred_labels = [[idx_to_label.get(p, "O") for p in pred] for pred in all_predictions]
            true_labels = [[idx_to_label.get(l, "O") for l in label] for label in all_labels]

            self.report, evaluation = classification_report(true_labels, pred_labels, digits=4)

            # Parse metrics
            if "micro" in evaluation:
                self.precision = evaluation["micro"]["precision"]
                self.recall = evaluation["micro"]["recall"]
                self.f1 = evaluation["micro"]["f1"]

        if self.evaluation:
            print(self.report)

        return {"f1": self.f1, "precision": self.precision, "recall": self.recall}
