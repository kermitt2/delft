"""
PyTorch-based Sequence Labeling Wrapper for DeLFT.

This module replaces the TensorFlow-based wrapper with PyTorch implementations.
"""

import os
import time
from itertools import islice

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from delft import DELFT_PROJECT_DIR
from delft.utilities.misc import print_parameters

from delft.sequenceLabelling.trainer import Trainer, Scorer
from delft.sequenceLabelling.trainer import (
    DEFAULT_WEIGHT_FILE_NAME,
    CONFIG_FILE_NAME,
    PROCESSOR_FILE_NAME,
    to_wandb_table,
)

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import prepare_preprocessor, Preprocessor
from delft.sequenceLabelling.data_loader import create_dataloader

from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.utilities.numpy import concatenate_or_none

from delft.sequenceLabelling.evaluation import classification_report

import transformers

transformers.logging.set_verbosity(transformers.logging.ERROR)


class Sequence(object):
    """
    PyTorch-based sequence labeling wrapper.

    Provides high-level API for training, evaluation, and tagging with
    sequence labeling models.
    """

    def __init__(
        self,
        model_name=None,
        architecture=None,
        embeddings_name=None,
        char_emb_size=25,
        max_char_length=30,
        char_lstm_units=25,
        word_lstm_units=100,
        max_sequence_length=300,
        dropout=0.5,
        recurrent_dropout=0.25,
        batch_size=20,
        optimizer="adam",
        learning_rate=None,
        lr_decay=0.9,
        clip_gradients=5.0,
        max_epoch=50,
        early_stop=True,
        patience=5,
        max_checkpoints_to_keep=0,
        log_dir=None,
        fold_number=1,
        multiprocessing=True,
        features_indices=None,
        transformer_name: str = None,
        report_to_wandb=False,
        device=None,
        nb_workers: int = None,
    ):
        if model_name is None:
            model_name = architecture
            if embeddings_name is not None:
                model_name += "_" + embeddings_name
            if transformer_name is not None:
                model_name += "_" + transformer_name

        self.model = None
        self.models = None
        self.p: Preprocessor = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name
        self.report_to_wandb = report_to_wandb
        
        # Set number of workers: default to cpu_count - 1, minimum 1
        if nb_workers is None:
            self.nb_workers = max(1, os.cpu_count() - 1)
        else:
            self.nb_workers = nb_workers

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        word_emb_size = 0
        self.embeddings = None
        self.model_local_path = None

        self.registry = load_resource_registry(
            os.path.join(DELFT_PROJECT_DIR, "resources-registry.json")
        )

        if self.embeddings_name is not None:
            self.embeddings = Embeddings(
                self.embeddings_name, resource_registry=self.registry
            )
            word_emb_size = self.embeddings.embed_size
        else:
            self.embeddings = None
            word_emb_size = 0

        if learning_rate is None:
            if transformer_name is None:
                learning_rate = 0.001
            else:
                learning_rate = 2e-5

        self.model_config = ModelConfig(
            model_name=model_name,
            architecture=architecture,
            embeddings_name=embeddings_name,
            word_embedding_size=word_emb_size,
            char_emb_size=char_emb_size,
            char_lstm_units=char_lstm_units,
            max_char_length=max_char_length,
            word_lstm_units=word_lstm_units,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            fold_number=fold_number,
            batch_size=batch_size,
            features_indices=features_indices,
            transformer_name=transformer_name,
        )

        self.training_config = TrainingConfig(
            learning_rate,
            batch_size,
            optimizer,
            lr_decay,
            clip_gradients,
            max_epoch,
            early_stop,
            patience,
            max_checkpoints_to_keep,
            multiprocessing,
        )

        if report_to_wandb:
            self._init_wandb(model_name)

    def _init_wandb(self, model_name, run_id=None):
        """Initialize Weights & Biases logging.
        
        Args:
            model_name: Name for the wandb run
            run_id: Optional run ID to resume an existing run
        """
        try:
            import wandb
            from dotenv import load_dotenv

            load_dotenv(override=True)
            if os.getenv("WANDB_API_KEY") is None:
                print("Warning: WANDB_API_KEY not set, wandb disabled")
                self.report_to_wandb = False
                return
            
            # Resume existing run or start new one
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
                        "embedding_size": self.model_config.word_embedding_size,
                        "batch_size": self.training_config.batch_size,
                        "learning_rate": self.training_config.learning_rate,
                        "max_epoch": self.training_config.max_epoch,
                    },
                )
            self.wandb = wandb
            wandb.define_metric("f1", summary="max")
            wandb.define_metric("eval_f1", summary="max")
        except ImportError:
            print("Warning: wandb not available")
            self.report_to_wandb = False

    def init_wandb_for_eval(self, run_id=None):
        """Initialize wandb for evaluation logging.
        
        Call this after model.load() to enable logging eval results to wandb.
        
        Args:
            run_id: Optional wandb run ID to resume an existing run.
                   If None, starts a new run.
        """
        self.report_to_wandb = True
        self._init_wandb(self.model_config.model_name, run_id=run_id)

    def train(
        self,
        x_train,
        y_train,
        f_train=None,
        x_valid=None,
        y_valid=None,
        f_valid=None,
        incremental=False,
        callbacks=None,
        multi_gpu=False,
    ):
        """Train the model."""
        # Multi-GPU support with PyTorch DataParallel
        if multi_gpu and torch.cuda.device_count() > 1:
            print(
                f"Running with multi-gpu. Number of devices: {torch.cuda.device_count()}"
            )

        self._train(
            x_train, y_train, f_train, x_valid, y_valid, f_valid, incremental, callbacks
        )

    def _train(
        self,
        x_train,
        y_train,
        f_train=None,
        x_valid=None,
        y_valid=None,
        f_valid=None,
        incremental=False,
        callbacks=None,
    ):
        """Internal training implementation."""
        # Concatenate all data for vocabulary building
        if x_valid is not None:
            x_all = np.concatenate((x_train, x_valid), axis=0)
        else:
            x_all = x_train

        if y_valid is not None:
            y_all = np.concatenate((y_train, y_valid), axis=0)
        else:
            y_all = y_train

        features_all = concatenate_or_none((f_train, f_valid), axis=0)

        if incremental:
            if self.model is None and self.models is None:
                print("Error: you must load a model first for incremental training")
                return
            print(
                "Incremental training from loaded model", self.model_config.model_name
            )
            self.p.extend(x_all, y_all)
        else:
            # Initialize preprocessor
            self.p = prepare_preprocessor(
                x_all, y_all, features=features_all, model_config=self.model_config
            )
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.case_vocab_size = len(self.p.vocab_case)

            # Create model
            self.model = get_model(
                self.model_config, len(self.p.vocab_tag), load_pretrained_weights=True
            )
            self.model.to(self.device)

        print_parameters(self.model_config, self.training_config)
        print(f"\nModel: {self.model_config.architecture}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create data loaders
        train_loader = create_dataloader(
            x_train,
            y_train,
            preprocessor=self.p,
            embeddings=self.embeddings,
            batch_size=self.training_config.batch_size,
            features=f_train,
            shuffle=True,
            model_config=self.model_config,
        )

        valid_loader = None
        if x_valid is not None:
            valid_loader = create_dataloader(
                x_valid,
                y_valid,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.training_config.batch_size,
                features=f_valid,
                shuffle=False,
                model_config=self.model_config,
            )

        # Use model output directory for checkpoints to keep files organized
        model_output_dir = os.path.join(
            "data/models/sequenceLabelling/", self.model_config.model_name
        )
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Create trainer
        trainer = Trainer(
            self.model,
            self.model_config,
            self.training_config,
            preprocessor=self.p,
            device=str(self.device),
            checkpoint_path=self.log_dir or model_output_dir,
            enable_wandb=self.report_to_wandb,
        )

        # Train
        trainer.train(train_loader, valid_loader, callbacks=callbacks)

    def train_nfold(
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        f_train=None,
        f_valid=None,
        incremental=False,
        callbacks=None,
        multi_gpu=False,
    ):
        """Train with n-fold cross validation."""
        x_all = (
            np.concatenate((x_train, x_valid), axis=0)
            if x_valid is not None
            else x_train
        )
        y_all = (
            np.concatenate((y_train, y_valid), axis=0)
            if y_valid is not None
            else y_train
        )
        features_all = concatenate_or_none((f_train, f_valid), axis=0)

        # Use model output directory for checkpoints
        model_output_dir = os.path.join(
            "data/models/sequenceLabelling/", self.model_config.model_name
        )
        os.makedirs(model_output_dir, exist_ok=True)

        if not incremental:
            self.p = prepare_preprocessor(
                x_all, y_all, features=features_all, model_config=self.model_config
            )
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.case_vocab_size = len(self.p.vocab_case)
            self.models = []

        fold_count = self.model_config.fold_number
        fold_size = len(x_train) // fold_count

        for fold_id in range(fold_count):
            print(
                f"\n------------------------ fold {fold_id} --------------------------------------"
            )

            # Split data for this fold
            fold_start = fold_size * fold_id
            fold_end = (
                fold_start + fold_size if fold_id < fold_count - 1 else len(x_train)
            )

            fold_x_train = np.concatenate([x_train[:fold_start], x_train[fold_end:]])
            fold_y_train = np.concatenate([y_train[:fold_start], y_train[fold_end:]])
            fold_x_valid = x_train[fold_start:fold_end]
            fold_y_valid = y_train[fold_start:fold_end]

            # Create model for this fold
            fold_model = get_model(
                self.model_config, len(self.p.vocab_tag), load_pretrained_weights=True
            )
            fold_model.to(self.device)

            if fold_id == 0:
                print_parameters(self.model_config, self.training_config)

            # Create data loaders
            train_loader = create_dataloader(
                fold_x_train,
                fold_y_train,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                model_config=self.model_config,
            )
            valid_loader = create_dataloader(
                fold_x_valid,
                fold_y_valid,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                model_config=self.model_config,
            )

            trainer = Trainer(
                fold_model,
                self.model_config,
                self.training_config,
                preprocessor=self.p,
                device=str(self.device),
                checkpoint_path=model_output_dir,
            )
            trainer.train(train_loader, valid_loader)

            self.models.append(fold_model)

    def eval(self, x_test, y_test, features=None):
        """Evaluate the model."""
        if self.model_config.fold_number > 1:
            self.eval_nfold(x_test, y_test, features=features)
        else:
            self.eval_single(x_test, y_test, features=features)

    def eval_single(self, x_test, y_test, features=None):
        """Evaluate single model."""
        if self.model is None:
            raise OSError("Could not find a model.")

        print_parameters(self.model_config, self.training_config)

        # Create test data loader
        test_loader = create_dataloader(
            x_test,
            y_test,
            preprocessor=self.p,
            embeddings=self.embeddings,
            batch_size=self.model_config.batch_size,
            features=features,
            shuffle=False,
            model_config=self.model_config,
        )

        # Evaluate
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                if hasattr(self.model, "decode"):
                    predictions = self.model.decode(inputs)
                else:
                    outputs = self.model(inputs)
                    predictions = outputs["logits"].argmax(dim=-1).tolist()

                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.tolist()

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

        # Convert to labels
        idx_to_label = {idx: label for label, idx in self.p.vocab_tag.items()}
        pred_labels = [
            [idx_to_label.get(p, "O") for p in pred] for pred in all_predictions
        ]
        true_labels = [
            [idx_to_label.get(l, "O") for l in label] for label in all_labels
        ]

        report, evaluation = classification_report(true_labels, pred_labels, digits=4)
        print(report)
        
        # Extract metrics for return and wandb logging
        metrics = {}
        if "micro" in evaluation:
            metrics = {
                "eval_f1": evaluation["micro"]["f1"],
                "eval_precision": evaluation["micro"]["precision"],
                "eval_recall": evaluation["micro"]["recall"],
            }
        
        # Log to wandb if enabled
        if self.report_to_wandb and hasattr(self, 'wandb'):
            # Log metrics
            self.wandb.log(metrics)
            # Log evaluation table
            columns, data = to_wandb_table(evaluation)
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({"Evaluation scores": table})
            print(f"Logged evaluation metrics to wandb: f1={metrics.get('eval_f1', 0):.4f}")
        
        return metrics

    def eval_nfold(self, x_test, y_test, features=None):
        """Evaluate n-fold models."""
        if self.models is None:
            raise OSError("No fold models found.")

        reports = []
        total_f1 = 0
        best_f1 = 0
        best_index = 0

        for i, model in enumerate(self.models):
            print(
                f"\n------------------------ fold {i} --------------------------------------"
            )

            test_loader = create_dataloader(
                x_test,
                y_test,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.model_config.batch_size,
                features=features,
                shuffle=False,
                model_config=self.model_config,
            )

            scorer = Scorer(test_loader, self.p, evaluation=True)
            metrics = scorer.on_epoch_end(model, self.device)

            f1 = metrics["f1"]
            total_f1 += f1
            if f1 > best_f1:
                best_f1 = f1
                best_index = i
            reports.append(scorer.report)

        print(
            "\n----------------------------------------------------------------------"
        )
        print(f"\nBest model: fold {best_index} with F1={best_f1:.4f}")
        print(f"Average F1: {total_f1 / len(self.models):.4f}")

        # Set best model as main model
        self.model = self.models[best_index]

    def tag(
        self, texts, output_format, features=None, batch_size=None, multi_gpu=False
    ):
        """Tag texts with the model."""
        if batch_size is not None:
            self.model_config.batch_size = batch_size

        if self.model is None:
            raise OSError("Could not find a model.")

        self.model.eval()
        start_time = time.time()

        # Preprocess texts
        from delft.sequenceLabelling.tagger import Tagger

        tagger = Tagger(
            self.model,
            self.model_config,
            self.embeddings,
            preprocessor=self.p,
            device=self.device,
        )

        annotations = tagger.tag(texts, output_format, features=features)

        runtime = round(time.time() - start_time, 3)
        if output_format == "json":
            annotations["runtime"] = runtime

        return annotations

    def save(
        self,
        dir_path="data/models/sequenceLabelling/",
        weight_file=DEFAULT_WEIGHT_FILE_NAME,
    ):
        """Save model to disk."""
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, CONFIG_FILE_NAME))
        print("Model config saved")

        self.p.save(os.path.join(directory, PROCESSOR_FILE_NAME))
        print("Preprocessor saved")

        if self.model is None and self.model_config.fold_number > 1:
            print("Error: model not saved. Run eval first to select best fold model.")
        else:
            # Save PyTorch model
            weight_path = os.path.join(directory, weight_file)
            torch.save(self.model.state_dict(), weight_path)
            print(f"Model weights saved to {weight_path}")

    def load(
        self,
        dir_path="data/models/sequenceLabelling/",
        weight_file=DEFAULT_WEIGHT_FILE_NAME,
    ):
        """Load model from disk."""
        model_path = os.path.join(dir_path, self.model_config.model_name)
        self.model_config = ModelConfig.load(os.path.join(model_path, CONFIG_FILE_NAME))

        if self.model_config.embeddings_name is not None:
            self.embeddings = Embeddings(
                self.model_config.embeddings_name,
                resource_registry=self.registry,
                use_cache=False,
            )
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.embeddings = None
            self.model_config.word_embedding_size = 0

        self.p = Preprocessor.load(os.path.join(model_path, PROCESSOR_FILE_NAME))

        self.model = get_model(
            self.model_config,
            len(self.p.vocab_tag),
            load_pretrained_weights=False,
            local_path=model_path,
        )

        weight_path = os.path.join(model_path, weight_file)
        print(f"Loading weights from {weight_path}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)

        print(f"Model loaded: {self.model_config.architecture}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


def next_n_lines(file_opened, N):
    """Read next N lines from file."""
    return [x.strip() for x in islice(file_opened, N)]
