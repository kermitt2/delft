import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score

from delft.sequenceLabelling.trainer import EarlyStopping, ModelCheckpoint


def compute_roc_auc(y_true, y_pred):
    """
    Mean ROC-AUC over the classes of a (possibly multi-label) problem.

    Each class is scored on its own rather than through roc_auc_score's own
    averaging: a class holding a single label value in this split makes
    roc_auc_score raise, and averaging them all in one call turns that into a
    0.0 for the whole evaluation instead of for that one class. Such a class
    falls back to r2_score, clamped at 0, as the Keras implementation did.

    Args:
        y_true: (n_samples, n_classes) array of true labels
        y_pred: (n_samples, n_classes) array of predicted probabilities

    Returns:
        The mean score as a float, 0.0 when there are no classes.
    """
    num_classes = y_true.shape[1]
    if num_classes == 0:
        return 0.0

    total_roc_auc = 0.0
    for j in range(num_classes):
        if len(np.unique(y_true[:, j])) == 1:
            class_roc_auc = max(0.0, r2_score(y_true[:, j], y_pred[:, j]))
        else:
            try:
                class_roc_auc = roc_auc_score(y_true[:, j], y_pred[:, j])
            except ValueError:
                class_roc_auc = 0.0
        total_roc_auc += class_roc_auc

    return total_roc_auc / num_classes


class Trainer(object):
    def __init__(self, model, model_config, training_config, device="cpu", checkpoint_path=""):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Optimizer
        learning_rate = training_config.learning_rate
        if model_config.transformer_name is not None:
            # BERT models usually use AdamW
            from torch.optim import AdamW

            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.BCEWithLogitsLoss()

        # Callbacks - use mode="min" for loss-based early stopping
        self.early_stopping = EarlyStopping(patience=training_config.patience, min_delta=0, mode="min")

        # Model checkpoint with model-specific filename
        checkpoint_filename = f"{model_config.model_name}_best_model.pth"
        checkpoint_filepath = (
            os.path.join(checkpoint_path, checkpoint_filename) if checkpoint_path else checkpoint_filename
        )
        self.model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor="loss", mode="min")

    def train(self, train_loader, valid_loader=None):
        for epoch in range(self.training_config.max_epoch):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                    labels = labels.to(self.device)
                else:
                    inputs = batch
                    labels = None

                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * (labels.size(0) if labels is not None else 1)

            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.training_config.max_epoch}, Train Loss: {avg_train_loss:.4f}")

            # Validation
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

                # Check metrics for early stopping
                # Default to ROC-AUC if enabled, else Loss
                if self.training_config.use_roc_auc:
                    # score = val_metrics["roc_auc"]
                    pass

                # Save model checkpoint if improved
                self.model_checkpoint(self.model, val_metrics["loss"])

                # Check early stopping (uses loss for stopping decision)
                if self.early_stopping(val_metrics["loss"]):
                    print("Early stopping")
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, labels = batch
                    labels = labels.to(self.device)
                else:
                    inputs = batch
                    labels = None

                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)

                outputs = self.model(inputs, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

                val_loss += loss.item() * labels.size(0)

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(dataloader.dataset)

        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)

        return {"loss": avg_val_loss, "roc_auc": compute_roc_auc(y_true, y_pred)}
