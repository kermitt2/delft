import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from delft.sequenceLabelling.trainer import EarlyStopping, ModelCheckpoint


class Trainer(object):
    def __init__(
        self, model, model_config, training_config, device="cpu", checkpoint_path=""
    ):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Optimizer
        learning_rate = training_config.learning_rate
        if model_config.transformer_name is not None:
            # BERT models usually use AdamW
            from transformers import AdamW

            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.BCEWithLogitsLoss()

        # Callbacks - use mode="min" for loss-based early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.patience, min_delta=0, mode="min"
        )
        
        # Model checkpoint with model-specific filename
        checkpoint_filename = f"{model_config.model_name}_best_model.pth"
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename) if checkpoint_path else checkpoint_filename
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

                train_loss += loss.item() * (
                    labels.size(0) if labels is not None else 1
                )

            avg_train_loss = train_loss / len(train_loader.dataset)
            print(
                f"Epoch {epoch + 1}/{self.training_config.max_epoch}, Train Loss: {avg_train_loss:.4f}"
            )

            # Validation
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                print(
                    f"Val Loss: {val_metrics['loss']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}"
                )

                # Check metrics for early stopping
                # Default to ROC-AUC if enabled, else Loss
                if self.training_config.use_roc_auc:
                    score = val_metrics["roc_auc"]
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

        # Calculate ROC-AUC
        # Handle single class vs multi-class
        if y_true.shape[1] > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_pred, average="macro")
            except ValueError:
                roc_auc = 0.0
        else:
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                roc_auc = 0.0

        return {"loss": avg_val_loss, "roc_auc": roc_auc}
