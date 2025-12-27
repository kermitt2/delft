import os
import time
import numpy as np

import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score

from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.utilities.misc import print_parameters, to_wandb_table
from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import getModel
from delft.textClassification.data_loader import create_dataloader
from delft.textClassification.trainer import Trainer
from delft.textClassification.preprocess import TextPreprocessor

from delft import DELFT_PROJECT_DIR

# File names for saving/loading
PREPROCESSOR_FILE = "preprocessor.json"


class Classifier(object):
    config_file = "config.json"
    weight_file = "model_weights.pth"

    def __init__(
        self,
        model_name=None,
        architecture="gru",
        embeddings_name=None,
        list_classes=[],
        char_emb_size=25,
        dropout=0.5,
        recurrent_dropout=0.25,
        use_char_feature=False,
        batch_size=256,
        optimizer="adam",
        learning_rate=0.001,
        lr_decay=0.9,
        clip_gradients=5.0,
        max_epoch=50,
        patience=5,
        log_dir=None,
        maxlen=300,
        fold_number=1,
        use_roc_auc=True,
        early_stop=True,
        class_weights=None,
        multiprocessing=True,
        transformer_name: str = None,
        device=None,
        report_to_wandb=False,
    ):
        self.model_config = ModelConfig(
            model_name=model_name,
            architecture=architecture,
            embeddings_name=embeddings_name,
            list_classes=list_classes,
            char_emb_size=char_emb_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            use_char_feature=use_char_feature,
            maxlen=maxlen,
            fold_number=fold_number,
            batch_size=batch_size,
            transformer_name=transformer_name,
        )

        self.training_config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_decay=lr_decay,
            clip_gradients=clip_gradients,
            max_epoch=max_epoch,
            patience=patience,
            use_roc_auc=use_roc_auc,
            early_stop=early_stop,
            class_weights=class_weights,
            multiprocessing=multiprocessing,
        )

        self.model = None
        self.models = None
        self.embeddings = None
        self.preprocessor = None
        self.report_to_wandb = report_to_wandb
        self.wandb = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.registry = load_resource_registry(
            os.path.join(DELFT_PROJECT_DIR, "resources-registry.json")
        )

        if embeddings_name is not None:
            self.embeddings = Embeddings(
                embeddings_name, resource_registry=self.registry
            )
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.model_config.word_embedding_size = 0

        if report_to_wandb:
            self._init_wandb(model_name)

    def _init_wandb(self, model_name, run_id=None):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            from dotenv import load_dotenv

            load_dotenv(override=True)
            if os.getenv("WANDB_API_KEY") is None:
                print("Warning: WANDB_API_KEY not set, wandb disabled")
                self.report_to_wandb = False
                return

            if run_id:
                wandb.init(id=run_id, resume="must")
                print(f"Resumed wandb run: {run_id}")
            else:
                wandb.init(
                    name=model_name,
                    config={
                        "model_name": self.model_config.model_name,
                        "architecture": self.model_config.architecture,
                        "transformer_name": self.model_config.transformer_name,
                        "embeddings_name": self.model_config.embeddings_name,
                        "batch_size": self.training_config.batch_size,
                        "learning_rate": self.training_config.learning_rate,
                        "max_epoch": self.training_config.max_epoch,
                    },
                )
            self.wandb = wandb
        except ImportError:
            print("Warning: wandb not available")
            self.report_to_wandb = False

    def train(
        self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None
    ):
        if self.model_config.fold_number == 1:
            self.train_single(x_train, y_train, vocab_init, incremental, callbacks)
        else:
            self.train_nfold(x_train, y_train, vocab_init, incremental, callbacks)

    def train_single(
        self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None
    ):
        # Create Data Loaders
        # Note: We need to handle validation split here if not n-fold
        # For simplicity, let's take last 10% as validation if early_stop is True and no folds

        x_valid = None
        y_valid = None

        if self.training_config.early_stop:
            split_idx = int(len(x_train) * 0.9)
            x_valid = x_train[split_idx:]
            y_valid = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]

        # Init model
        self.model = getModel(self.model_config, self.training_config)
        self.model.to(self.device)

        print(f"Model: {self.model_config.architecture}")

        # Helper to get tokenizer if needed
        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            # Logic to fetch tokenizer from model or transformer helper
            # In models_pytorch.py we use AutoModel.
            # We need generic way to get tokenizer.
            from transformers import AutoTokenizer

            transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.transformer_name
            )

        train_loader = create_dataloader(
            x_train,
            y_train,
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

        valid_loader = None
        if x_valid is not None:
            valid_loader = create_dataloader(
                x_valid,
                y_valid,
                self.model_config,
                embeddings=self.embeddings,
                transformer_tokenizer=transformer_tokenizer,
                batch_size=self.training_config.batch_size,
                shuffle=False,
            )

        # Ensure model output directory exists for checkpoints
        model_dir = self._get_model_dir()
        os.makedirs(model_dir, exist_ok=True)

        trainer = Trainer(
            self.model,
            self.model_config,
            self.training_config,
            device=str(self.device),
            checkpoint_path=model_dir,
        )

        trainer.train(train_loader, valid_loader)

    def train_nfold(
        self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None
    ):
        pass  # To implement if needed, following logic in wrapper.py

    def eval(self, x_test, y_test):
        """Evaluate model on test data.

        Args:
            x_test: Test texts
            y_test: Test labels (numpy array with shape [n_samples, n_classes])
        """
        print_parameters(self.model_config, self.training_config)

        if self.model is None:
            raise OSError("Model not loaded")

        self.model.eval()
        self.model.to(self.device)

        # Get transformer tokenizer if needed
        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            from transformers import AutoTokenizer

            transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.transformer_name
            )

        # Create dataloader
        test_loader = create_dataloader(
            x_test,
            y_test,
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            batch_size=self.model_config.batch_size,
            shuffle=False,
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
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

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs["logits"])
                all_preds.append(probs.cpu().numpy())
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())

        y_pred_probs = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0) if all_labels else None

        if y_true is None:
            print("No labels provided for evaluation")
            return

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_probs > 0.5).astype(int)

        # Calculate per-class metrics
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true, y_pred_binary, average=None
        )

        # Print results
        print("\n-----------------------------------------------")
        print(f"Evaluation on {len(x_test)} instances:")
        print(
            f"{'':>14}  {'precision':>12}  {'recall':>12}  {'f-score':>12}  {'support':>12}"
        )

        evaluation = {"labels": {}, "micro": {}, "macro": {}}
        total_support = 0

        for i, class_name in enumerate(self.model_config.list_classes):
            class_name_short = class_name[:14]
            print(
                f"{class_name_short:>14}  {precision[i]:>12.4f}  {recall[i]:>12.4f}  {fscore[i]:>12.4f}  {int(support[i]):>12}"
            )
            evaluation["labels"][class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(fscore[i]),
                "support": int(support[i]),
            }
            total_support += int(support[i])

        # Calculate macro and micro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(fscore)

        # Flatten for micro average calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_binary.flatten()
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")
        micro_precision, micro_recall, _, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average="micro"
        )

        print(
            f"{'macro avg':>14}  {macro_precision:>12.4f}  {macro_recall:>12.4f}  {macro_f1:>12.4f}  {total_support:>12}"
        )
        print(
            f"{'micro avg':>14}  {micro_precision:>12.4f}  {micro_recall:>12.4f}  {micro_f1:>12.4f}  {total_support:>12}"
        )
        print("-----------------------------------------------")

        evaluation["macro"] = {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
            "support": total_support,
        }
        evaluation["micro"] = {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1": float(micro_f1),
            "support": total_support,
        }

        # Log to wandb if enabled
        if self.report_to_wandb and hasattr(self, "wandb") and self.wandb is not None:
            metrics = {
                "eval_f1": micro_f1,
                "eval_precision": micro_precision,
                "eval_recall": micro_recall,
            }
            self.wandb.log(metrics)
            # Log evaluation table
            columns, data = to_wandb_table(evaluation)
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({"Evaluation scores": table})
            print(f"Logged evaluation metrics to wandb: f1={micro_f1:.4f}")

        return evaluation

    def predict(
        self, texts, output_format="json", use_main_thread_only=False, batch_size=None
    ):
        if batch_size is not None:
            self.model_config.batch_size = batch_size

        if self.model is None:
            raise OSError("Model not loaded")

        self.model.eval()
        self.model.to(self.device)

        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            from transformers import AutoTokenizer

            transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.transformer_name
            )

        # Preprocess texts if they are raw strings
        if len(texts) > 0 and isinstance(texts[0], str):
            # Clean text?
            # data_loader expects raw text usually and preprocesses inside Dataset if we set up logic right.
            # In TextClassificationDataset we call to_vector_single which cleans text.
            pass

        data_loader = create_dataloader(
            texts,
            None,
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            batch_size=self.model_config.batch_size,
            shuffle=False,
        )

        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch  # no labels
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs["logits"])
                all_preds.append(probs.cpu().numpy())

        result = np.concatenate(all_preds, axis=0)

        if output_format == "json":
            res = {
                "software": "DeLFT",
                "date": time.ctime(),
                "model": self.model_config.model_name,
                "classifications": [],
            }

            for i in range(len(texts)):
                classification = {
                    "text": texts[i],
                    # ... format as expected
                }
                # Simplify for now
                best_class_idx = np.argmax(result[i])
                classification["class"] = self.model_config.list_classes[best_class_idx]
                classification["score"] = float(result[i][best_class_idx])
                res["classifications"].append(classification)
            return res
        else:
            return result

    def save(self, dir_path="data/models/textClassification/"):
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, self.config_file))

        # Save preprocessor if present
        if self.preprocessor is not None:
            self.preprocessor.save(os.path.join(directory, PREPROCESSOR_FILE))
            print("Preprocessor saved")

        # Save PyTorch model
        torch.save(self.model.state_dict(), os.path.join(directory, self.weight_file))
        print(f"Model saved to {directory}")

    def load(self, dir_path="data/models/textClassification/"):
        model_path = os.path.join(dir_path, self.model_config.model_name)

        # Load config
        self.model_config = ModelConfig.load(os.path.join(model_path, self.config_file))

        # Load preprocessor if present
        preprocessor_path = os.path.join(model_path, PREPROCESSOR_FILE)
        if os.path.exists(preprocessor_path):
            self.preprocessor = TextPreprocessor.load(preprocessor_path)
            print("Preprocessor loaded")

        # Load embeddings if needed
        if self.model_config.embeddings_name is not None:
            self.embeddings = Embeddings(
                self.model_config.embeddings_name,
                resource_registry=self.registry,
            )

        # Init model
        self.model = getModel(self.model_config, self.training_config)

        # Load weights
        weight_path = os.path.join(model_path, self.weight_file)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {weight_path}")

    def _get_model_dir(self):
        return os.path.join(
            "data/models/textClassification/", self.model_config.model_name
        )
