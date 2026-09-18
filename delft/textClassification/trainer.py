import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score

from delft.sequenceLabelling.trainer import EarlyStopping, ModelCheckpoint, remove_file, unique_checkpoint_path


def restore_best_weights(model, checkpoint_filepath, device="cpu"):
    """
    Load the best-epoch checkpoint back into the model and drop the file.

    ModelCheckpoint writes the weights of the best epoch as training goes, but
    the model left in memory at the end of training is the last epoch's. The
    checkpoint is a temporary artefact - the wrapper saves the real model - so
    it is removed once restored.

    Args:
        model: the model to restore in place, DDP-wrapped or not
        checkpoint_filepath: path written by ModelCheckpoint
        device: map_location for torch.load

    Returns:
        True when weights were restored, False when there is no checkpoint
        (no validation set, or no epoch ever improved).
    """
    if not os.path.exists(checkpoint_filepath):
        return False

    # ModelCheckpoint saves module.state_dict() for DDP-wrapped models
    target = model.module if hasattr(model, "module") else model
    # the checkpoint is a plain state_dict written by ModelCheckpoint
    target.load_state_dict(torch.load(checkpoint_filepath, map_location=device, weights_only=True))

    try:
        os.remove(checkpoint_filepath)
    except OSError:
        pass  # ignore if the file can't be removed

    return True


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
        self._apply_class_weights()

        # Metric driving both checkpointing and early stopping: ROC-AUC when
        # use_roc_auc is set (higher is better), validation loss otherwise.
        self.monitor = "roc_auc" if training_config.use_roc_auc else "loss"
        monitor_mode = "max" if self.monitor == "roc_auc" else "min"

        self.early_stopping = EarlyStopping(patience=training_config.patience, min_delta=0, mode=monitor_mode)

        # The file of the best weights is created by train(), one for each training
        self.model_checkpoint = ModelCheckpoint(None, monitor=self.monitor, mode=monitor_mode)

    def _apply_class_weights(self):
        """
        ``class_weights`` gives a weight to classes by their index, as in {0: 1.5, 1: 1.0}:
        the loss of a class is multiplied by its weight, 1 when it has none.
        """
        class_weights = self.training_config.class_weights
        if not class_weights or not hasattr(self.model, "loss_fn"):
            return
        nb_classes = len(self.model_config.list_classes)
        weights = torch.ones(nb_classes, dtype=torch.float32)
        for index, weight in class_weights.items():
            if not 0 <= int(index) < nb_classes:
                raise ValueError(f"class_weights: no class {index}, the model has {nb_classes} classes")
            weights[int(index)] = float(weight)
        self.model.loss_fn = nn.BCEWithLogitsLoss(weight=weights.to(self.device))

    def train(self, train_loader, valid_loader=None):
        # a file of its own for the best weights of this training, removed at its end
        if valid_loader is not None:
            self.model_checkpoint.filepath = unique_checkpoint_path(
                self.checkpoint_path, self.model_config.model_name, "best_model.pth"
            )
        try:
            self._run_epochs(train_loader, valid_loader)
        finally:
            remove_file(self.model_checkpoint.filepath)

    def _run_epochs(self, train_loader, valid_loader=None):
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
                if self.training_config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.clip_gradients)
                self.optimizer.step()

                train_loss += loss.item() * (labels.size(0) if labels is not None else 1)

            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.training_config.max_epoch}, Train Loss: {avg_train_loss:.4f}")

            # Validation
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

                score = val_metrics[self.monitor]

                # Save model checkpoint if improved
                self.model_checkpoint(self.model, score)

                # Check early stopping
                if self.early_stopping(score):
                    print("Early stopping")
                    break

        # Training ends on the last epoch, which - with early stopping - is
        # `patience` epochs past the best one. Put the best weights back before
        # the wrapper saves the model, as the sequence labelling trainer does.
        if self.model_checkpoint.best_score is not None and restore_best_weights(
            self.model, self.model_checkpoint.filepath, self.device
        ):
            print(f"Restored best weights from {self.model_checkpoint.filepath}")

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
