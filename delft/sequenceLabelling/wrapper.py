"""
PyTorch-based Sequence Labeling Wrapper for DeLFT.

This module replaces the TensorFlow-based wrapper with PyTorch implementations.
"""

import os
import time
import warnings
from itertools import islice

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import transformers

from delft import DELFT_PROJECT_DIR
from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.evaluation import classification_report
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import Preprocessor, prepare_preprocessor
from delft.sequenceLabelling.text_features import text_from_features, tokens_per_position
from delft.sequenceLabelling.trainer import (
    CONFIG_FILE_NAME,
    DEFAULT_WEIGHT_FILE_NAME,
    PROCESSOR_FILE_NAME,
    Scorer,
    Trainer,
)
from delft.sequenceLabelling.windows import join_scored_windows
from delft.utilities.cuda_setup import configure_cudnn_for_device, validate_device_arch_compatibility
from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.utilities.hub_models import fetch_model, is_remote, resolve_model
from delft.utilities.misc import print_parameters, to_wandb_table
from delft.utilities.numpy import concatenate_or_none
from delft.utilities.Utilities import pick_device
from delft.utilities.weights import (
    SAFETENSORS_WEIGHT_FILE_NAME,
    find_weight_file,
    load_weights,
    remove_other_weights,
    save_weights,
)

transformers.logging.set_verbosity(transformers.logging.ERROR)


def summarize_fold_scores(scores):
    """
    The mean and the population standard deviation, over the folds, of the scores of
    an n-fold evaluation, given as one dict of ``precision``, ``recall`` and ``f1``
    per fold, and the index of the fold with the best f1 (the first one on a tie).
    """
    if not scores:
        raise ValueError("No fold scores to summarize")
    keys = ("precision", "recall", "f1")
    values = {key: np.array([score[key] for score in scores], dtype=float) for key in keys}
    return {
        "folds": [dict(score) for score in scores],
        "mean": {key: float(values[key].mean()) for key in keys},
        "std": {key: float(values[key].std()) for key in keys},
        "best_fold": int(np.argmax(values["f1"])),
    }


class Sequence(object):
    """
    PyTorch-based sequence labeling wrapper.

    Provides high-level API for training, evaluation, and tagging with
    sequence labeling models.

    Word embeddings are optional. Pass ``embeddings_name=None`` together
    with ``transformer_name=None`` to train using only character (and
    optional features) inputs — this avoids loading multi-GB embedding
    files at the cost of a small accuracy drop. See issue #216.
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
        lr_decay=0.5,
        clip_gradients=1.0,
        max_epoch=50,
        early_stop=True,
        patience=5,
        max_checkpoints_to_keep=0,
        log_dir=None,
        fold_number=1,
        multiprocessing=True,
        features_indices=None,
        features_vocabulary_size: int = None,
        transformer_name: str = None,
        report_to_wandb=False,
        wandb_project: str = None,
        device=None,
        nb_workers: int = None,
        short_model_name: str = None,
        window_stride: int = None,
        text_features_indices=None,
        continuous_features_indices=None,
        whole_text_tokenization=False,
    ):
        self.short_model_name = short_model_name
        if model_name is None:
            model_name = architecture
            if embeddings_name is not None:
                model_name += "_" + embeddings_name
            if transformer_name is not None:
                model_name += "_" + transformer_name

        self.model = None
        self.models = None
        self.p: Preprocessor = None
        self._tagger = None  # built lazily and reused across tag() calls
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name
        self.report_to_wandb = report_to_wandb
        self.wandb_project = wandb_project

        # Set number of DataLoader worker processes. The per-epoch respawn
        # cost on macOS makes large worker counts counterproductive for small
        # datasets, and ``create_dataloader`` further auto-caps based on
        # dataset size. Default to a modest 4 (or fewer on small machines).
        # ``nb_workers=0`` means in-process loading, with no worker process
        # spawned at all — the value to use when DeLFT runs embedded in a
        # host process such as GROBID (see ``tag()``).
        self.nb_workers_explicit = nb_workers is not None
        if nb_workers is None:
            self.nb_workers = max(1, min(4, os.cpu_count() - 1))
        else:
            self.nb_workers = max(0, nb_workers)

        # Set device
        self.device = pick_device(device)
        validate_device_arch_compatibility(self.device)
        configure_cudnn_for_device(self.device)

        word_emb_size = 0
        self.embeddings = None
        self.model_local_path = None

        self.registry = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))

        if self.embeddings_name is not None:
            self.embeddings = Embeddings(self.embeddings_name, resource_registry=self.registry)
            # one vector per column the text of a token is taken from
            word_emb_size = self.embeddings.embed_size * tokens_per_position(text_features_indices)
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
            features_vocabulary_size=features_vocabulary_size or ModelConfig.DEFAULT_FEATURES_VOCABULARY_SIZE,
            transformer_name=transformer_name,
            window_stride=window_stride,
            text_features_indices=text_features_indices,
            continuous_features_indices=continuous_features_indices,
            whole_text_tokenization=whole_text_tokenization,
        )
        self.window_stride = window_stride

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

            if not hasattr(wandb, "init"):
                # ``wandb`` resolved to an empty PEP-420 namespace package
                # from a local ``wandb/`` directory (wandb's own run-log
                # folder) because the real wandb package is *not installed*
                # in this environment. With wandb installed, the regular
                # package in site-packages wins over the namespace candidate;
                # without it, the cwd directory wins by elimination.
                print(
                    f"Warning: 'wandb' module imported but has no 'init' "
                    f"attribute — wandb is not installed in this environment "
                    f"and a local 'wandb/' directory in {os.getcwd()!r} is "
                    f"being picked up as an empty namespace package (PEP 420). "
                    f"Run 'pip install wandb' or remove the local 'wandb/' tree. "
                    f"Disabling wandb."
                )
                self.report_to_wandb = False
                return

            load_dotenv(override=True)
            if os.getenv("WANDB_API_KEY") is None:
                print("Warning: WANDB_API_KEY not set, wandb disabled")
                self.report_to_wandb = False
                return

            # wandb's default run-log dir is ``./wandb/``. With wandb installed
            # that's harmless (the regular package always wins). But if this
            # repo is later run in an environment where wandb isn't installed,
            # that directory becomes a PEP-420 namespace package and ``import
            # wandb`` silently resolves to an empty module — masking the real
            # "wandb not installed" diagnosis. wandb sucks for picking that
            # name; we override to ``./.wandb/`` (dot-prefix → impossible as a
            # Python module name → can never be imported as a namespace pkg).
            # Respect any user-provided WANDB_DIR from env/.env.
            if not os.getenv("WANDB_DIR"):
                os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), ".wandb")
                print(
                    f"wandb's default './wandb/' run dir shadows the wandb "
                    f"package import — using {os.environ['WANDB_DIR']!r} "
                    f"instead. Set WANDB_DIR to override."
                )

            # CLI/API arg wins; otherwise fall back to WANDB_PROJECT env var.
            # Passing project=None lets wandb apply its own default ("uncategorized").
            project = self.wandb_project or os.getenv("WANDB_PROJECT")

            # Resume existing run or start new one
            if run_id:
                wandb.init(id=run_id, resume="must", project=project)
                print(f"Resumed wandb run: {run_id}")
            else:
                wandb.init(
                    name=model_name,
                    project=project,
                    config={
                        "model_name": self.model_config.model_name,
                        "short_model_name": self.short_model_name,
                        "architecture": self.model_config.architecture,
                        "transformer_name": self.model_config.transformer_name,
                        "embeddings_name": self.model_config.embeddings_name,
                        "embedding_size": self.model_config.word_embedding_size,
                        "batch_size": self.training_config.batch_size,
                        "learning_rate": self.training_config.learning_rate,
                        "max_epoch": self.training_config.max_epoch,
                        "patience": self.training_config.patience,
                        "early_stop": self.training_config.early_stop,
                        "max_sequence_length": self.model_config.max_sequence_length,
                    },
                )
            self.wandb = wandb
            wandb.define_metric("f1", summary="max")
            wandb.define_metric("eval_f1", summary="max")
        except ImportError:
            print("Warning: wandb not available")
            self.report_to_wandb = False

    def init_wandb_for_eval(self, run_id=None, wandb_project=None):
        """Initialize wandb for evaluation logging.

        Call this after model.load() to enable logging eval results to wandb.

        Args:
            run_id: Optional wandb run ID to resume an existing run.
                   If None, starts a new run.
            wandb_project: Optional wandb project name. Overrides any value set
                   at construction time. If neither is provided, falls back to
                   the WANDB_PROJECT env var.
        """
        self.report_to_wandb = True
        if wandb_project is not None:
            self.wandb_project = wandb_project
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
        distributed = False
        local_rank = 0

        # Multi-GPU support with PyTorch DistributedDataParallel
        if multi_gpu:
            from delft.utilities.distributed import (
                get_world_size,
                is_main_process,
                setup_distributed,
            )

            local_rank = setup_distributed()

            if get_world_size() > 1:
                distributed = True
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(self.device)

                if is_main_process():
                    print(f"Running distributed training with {get_world_size()} GPUs")
            else:
                if torch.cuda.device_count() > 1:
                    print(f"Warning: {torch.cuda.device_count()} GPUs available but running single-process.")
                    print("For multi-GPU training, launch with: torchrun --nproc_per_node=N")

        self._train(
            x_train,
            y_train,
            f_train,
            x_valid,
            y_valid,
            f_valid,
            incremental,
            callbacks,
            distributed,
            local_rank,
        )

        # Cleanup distributed training
        if distributed:
            from delft.utilities.distributed import cleanup_distributed

            cleanup_distributed()

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
        distributed=False,
        local_rank=0,
    ):
        """Internal training implementation."""
        # Import distributed utilities if needed
        if distributed:
            from delft.utilities.distributed import is_main_process

        # Concatenate all data for vocabulary building
        x_all = concatenate_or_none((x_train, x_valid))
        y_all = concatenate_or_none((y_train, y_valid))
        features_all = concatenate_or_none((f_train, f_valid))
        # the characters are those of the text the model reads, which may come from the features
        x_all = text_from_features(x_all, features_all, self.model_config.text_features_indices)

        if incremental:
            if self.model is None and self.models is None:
                print("Error: you must load a model first for incremental training")
                return
            print("Incremental training from loaded model", self.model_config.model_name)
            self.p.extend(x_all, y_all)
        else:
            # Initialize preprocessor
            self.p = prepare_preprocessor(x_all, y_all, features=features_all, model_config=self.model_config)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.case_vocab_size = len(self.p.vocab_case)

            # Create model
            self.model = get_model(self.model_config, len(self.p.vocab_tag), load_pretrained_weights=True)
            self.model.to(self.device)

        # Only print on main process for distributed training
        if not distributed or is_main_process():
            print_parameters(self.model_config, self.training_config)
            print(f"\nModel: {self.model_config.architecture}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create data loaders with distributed sampler if needed
        train_loader = create_dataloader(
            x_train,
            y_train,
            preprocessor=self.p,
            embeddings=self.embeddings,
            batch_size=self.training_config.batch_size,
            features=f_train,
            shuffle=True,
            model_config=self.model_config,
            num_workers=self.nb_workers,
            distributed=distributed,
            role="train",
            window_stride=self.model_config.window_stride,
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
                num_workers=self.nb_workers,
                distributed=distributed,
                role="valid",
                window_stride=self._scoring_window_stride(),
            )

        # Use model output directory for checkpoints to keep files organized
        model_output_dir = os.path.join("data/models/sequenceLabelling/", self.model_config.model_name)
        if not distributed or is_main_process():
            os.makedirs(model_output_dir, exist_ok=True)

        # Create trainer with distributed support
        trainer = Trainer(
            self.model,
            self.model_config,
            self.training_config,
            preprocessor=self.p,
            device=str(self.device),
            checkpoint_path=self.log_dir or model_output_dir,
            enable_wandb=self.report_to_wandb,
            distributed=distributed,
            local_rank=local_rank,
        )

        # Train
        trainer.train(train_loader, valid_loader, callbacks=callbacks)

        # Get the unwrapped model back from trainer for saving
        if distributed:
            self.model = trainer._unwrapped_model

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
        x_all = concatenate_or_none((x_train, x_valid))
        y_all = concatenate_or_none((y_train, y_valid))
        features_all = concatenate_or_none((f_train, f_valid))
        x_all = text_from_features(x_all, features_all, self.model_config.text_features_indices)

        # Use model output directory for checkpoints
        model_output_dir = os.path.join("data/models/sequenceLabelling/", self.model_config.model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        if not incremental:
            self.p = prepare_preprocessor(x_all, y_all, features=features_all, model_config=self.model_config)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.case_vocab_size = len(self.p.vocab_case)
            self.models = []

        fold_count = self.model_config.fold_number
        fold_size = len(x_train) // fold_count

        for fold_id in range(fold_count):
            print(f"\n------------------------ fold {fold_id} --------------------------------------")

            # Split data for this fold
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size if fold_id < fold_count - 1 else len(x_train)

            fold_x_train = concatenate_or_none([x_train[:fold_start], x_train[fold_end:]])
            fold_y_train = concatenate_or_none([y_train[:fold_start], y_train[fold_end:]])
            fold_x_valid = x_train[fold_start:fold_end]
            fold_y_valid = y_train[fold_start:fold_end]
            fold_f_train = fold_f_valid = None
            if f_train is not None:
                fold_f_train = concatenate_or_none([f_train[:fold_start], f_train[fold_end:]])
                fold_f_valid = f_train[fold_start:fold_end]

            # Create model for this fold
            fold_model = get_model(self.model_config, len(self.p.vocab_tag), load_pretrained_weights=True)
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
                features=fold_f_train,
                shuffle=True,
                model_config=self.model_config,
                num_workers=self.nb_workers,
                role=f"fold{fold_id}-train",
                window_stride=self.model_config.window_stride,
            )
            valid_loader = create_dataloader(
                fold_x_valid,
                fold_y_valid,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.training_config.batch_size,
                features=fold_f_valid,
                shuffle=False,
                model_config=self.model_config,
                num_workers=self.nb_workers,
                role=f"fold{fold_id}-valid",
                window_stride=self._scoring_window_stride(),
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

    def _scoring_window_stride(self):
        """
        The validation and evaluation sets are cut into windows when the training set is,
        but side by side: each token is then scored once, and the windows of a sequence
        are put back together before scoring.
        """
        return self.model_config.max_sequence_length if self.model_config.window_stride else None

    def eval(self, x_test, y_test, features=None):
        """Evaluate the model."""
        if self.model_config.fold_number > 1:
            return self.eval_nfold(x_test, y_test, features=features)
        return self.eval_single(x_test, y_test, features=features)

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
            num_workers=self.nb_workers,
            role="eval",
            window_stride=self._scoring_window_stride(),
        )

        # Evaluate
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                if hasattr(self.model, "decode"):
                    predictions = self.model.decode(inputs)
                else:
                    outputs = self.model(inputs)
                    predictions = outputs["logits"].argmax(dim=-1).tolist()

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

        all_predictions, all_labels = join_scored_windows(test_loader, all_predictions, all_labels)

        # Convert to labels
        idx_to_label = {idx: label for label, idx in self.p.vocab_tag.items()}
        pred_labels = [[idx_to_label.get(p, "O") for p in pred] for pred in all_predictions]
        true_labels = [[idx_to_label.get(l, "O") for l in label] for label in all_labels]

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
        if self.report_to_wandb and hasattr(self, "wandb"):
            # Log metrics
            self.wandb.log(metrics)
            # Log evaluation table
            columns, data = to_wandb_table(evaluation)
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({"Evaluation scores": table})
            print(f"Logged evaluation metrics to wandb: f1={metrics.get('eval_f1', 0):.4f}")

        return metrics

    def eval_nfold(self, x_test, y_test, features=None):
        """
        Evaluate the models of every fold on the test set, and report the mean and the
        standard deviation of their scores.

        Returns a dict with the micro precision, recall and f1 of every fold under
        ``"folds"``, and their ``"mean"`` and ``"std"`` (population standard deviation)
        over the folds, plus ``"best_fold"``. The model of the best fold becomes the
        model of the wrapper.
        """
        if self.models is None:
            raise OSError("No fold models found.")

        reports = []
        scores = []

        for i, model in enumerate(self.models):
            print(f"\n------------------------ fold {i} --------------------------------------")

            test_loader = create_dataloader(
                x_test,
                y_test,
                preprocessor=self.p,
                embeddings=self.embeddings,
                batch_size=self.model_config.batch_size,
                features=features,
                shuffle=False,
                model_config=self.model_config,
                num_workers=self.nb_workers,
                role=f"fold{i}-eval",
                window_stride=self._scoring_window_stride(),
            )

            scorer = Scorer(test_loader, self.p, evaluation=True)
            metrics = scorer.on_epoch_end(model, self.device)
            scores.append({key: float(metrics[key]) for key in ("precision", "recall", "f1")})
            reports.append(scorer.report)

        summary = summarize_fold_scores(scores)
        best_index = summary["best_fold"]

        print("\n----------------------------------------------------------------------")
        print(f"\nBest model: fold {best_index} with F1={scores[best_index]['f1']:.4f}")
        print(f"\n{'fold':>6}  {'precision':>12}  {'recall':>12}  {'f-score':>12}")
        for i, score in enumerate(scores):
            print(f"{i:>6}  {score['precision']:>12.4f}  {score['recall']:>12.4f}  {score['f1']:>12.4f}")
        mean, std = summary["mean"], summary["std"]
        print(f"{'mean':>6}  {mean['precision']:>12.4f}  {mean['recall']:>12.4f}  {mean['f1']:>12.4f}")
        print(f"{'std':>6}  {std['precision']:>12.4f}  {std['recall']:>12.4f}  {std['f1']:>12.4f}")
        print(f"\nAverage F1: {mean['f1']:.4f} (std {std['f1']:.4f}) over {len(scores)} folds")

        if self.report_to_wandb and hasattr(self, "wandb"):
            self.wandb.log(
                {
                    "eval_f1_mean": mean["f1"],
                    "eval_f1_std": std["f1"],
                    "eval_precision_mean": mean["precision"],
                    "eval_precision_std": std["precision"],
                    "eval_recall_mean": mean["recall"],
                    "eval_recall_std": std["recall"],
                    "eval_best_fold": best_index,
                }
            )
            columns = ["fold", "precision", "recall", "f1"]
            data = [[i, s["precision"], s["recall"], s["f1"]] for i, s in enumerate(scores)]
            data.append(["mean", mean["precision"], mean["recall"], mean["f1"]])
            data.append(["std", std["precision"], std["recall"], std["f1"]])
            self.wandb.log({"Fold scores": self.wandb.Table(columns=columns, data=data)})

        # Set best model as main model
        self.model = self.models[best_index]

        return summary

    def tag(
        self, texts, output_format, features=None, batch_size=None, multi_gpu=False, nb_workers=None, window_stride=None
    ):
        """Tag texts with the model.

        ``window_stride``: a text longer than ``max_sequence_length`` is labelled whole,
        in windows of that length, one every ``window_stride``, rather than truncated.
        It defaults to the stride the model was trained with, and a model trained
        without windows truncates (see ``Tagger.tag``).

        ``nb_workers`` is the number of DataLoader worker processes to use for
        this call. It falls back to the value given to the constructor, and to
        0 (in-process, no worker process spawned) when neither was set — a
        DataLoader is built per call, so worker processes are respawned on
        every tag() and rarely pay for themselves at inference time. Callers
        embedding DeLFT in a host process (e.g. GROBID through JEP) should
        keep it at 0.
        """
        if batch_size is not None:
            self.model_config.batch_size = batch_size

        if nb_workers is None:
            nb_workers = self.nb_workers if self.nb_workers_explicit else 0

        if self.model is None:
            raise OSError("Could not find a model.")

        self.model.eval()
        start_time = time.time()

        annotations = self._get_tagger(nb_workers).tag(
            texts, output_format, features=features, window_stride=window_stride
        )

        runtime = round(time.time() - start_time, 3)
        if output_format == "json":
            annotations["runtime"] = runtime

        return annotations

    def _get_tagger(self, nb_workers):
        """Return the Tagger for the current model, building it at most once.

        GROBID calls tag() many times per document and a Tagger is cheap but
        not free, so keep it around. It is rebuilt whenever anything it closes
        over is replaced — a load(), a fold selection, or a different worker
        count.
        """
        from delft.sequenceLabelling.tagger import Tagger

        cached = getattr(self, "_tagger", None)
        if (
            cached is not None
            and cached.model is self.model
            and cached.preprocessor is self.p
            and cached.embeddings is self.embeddings
            and cached.nb_workers == nb_workers
        ):
            return cached

        self._tagger = Tagger(
            self.model,
            self.model_config,
            self.embeddings,
            preprocessor=self.p,
            device=self.device,
            nb_workers=nb_workers,
        )
        return self._tagger

    def save(
        self,
        dir_path="data/models/sequenceLabelling/",
        weight_file=SAFETENSORS_WEIGHT_FILE_NAME,
    ):
        """Save model to disk.

        The weights are saved as safetensors, or as a pickled state dict when
        ``weight_file`` does not end with ``.safetensors``, as in
        ``DEFAULT_WEIGHT_FILE_NAME``, the format written up to DeLFT 1.1.0 (see
        ``delft.utilities.weights``). Weights the directory holds in the other format,
        from a previous training, are removed.
        """
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
            save_weights(self.model, weight_path)
            remove_other_weights(directory, weight_file)
            print(f"Model weights saved to {weight_path}")

    def load(
        self,
        dir_path="data/models/sequenceLabelling/",
        weight_file=SAFETENSORS_WEIGHT_FILE_NAME,
        cache_dir=None,
        token=None,
    ):
        """Load model from disk, from the Hugging Face Hub or over HTTP.

        ``dir_path`` is either

        - a models directory, the model being the folder of it with the name of the
          model. When it is not there, it is first downloaded there from the Hub, if the
          resources registry gives it a place on the Hub. A model trained or copied there
          is never touched;
        - the directory of the model itself, whatever its name;
        - a repository or a bucket of the Hub, ``hf://lfoppiano/grobid-model-header`` or
          ``hf://lfoppiano/grobid-model-header@v1.1.0``, the model being the folder of it
          with the name of the model, or that folder itself,
          ``hf://owner/repository/model-name``, whatever the name of the model;
        - the URL of an archive of the model directory, ``https://.../model-name.zip``;
          or of the model directory as a folder of files, or of the folder holding it
          under the name of the model, or else ``{name of the model}.zip``.

        See ``delft.utilities.hub_models`` for the last two, whose model is downloaded
        to ``cache_dir`` when it is not there yet. ``cache_dir`` defaults to the directory
        the ``DELFT_MODELS_DIR`` environment variable names, else to
        ``~/.cache/delft/models``. ``token`` gives access to what is private.

        When ``weight_file`` is not in the model directory, the weights it holds in the
        other format (pickled state dict or safetensors) are loaded instead.
        """
        model_name = self.model_config.model_name
        if is_remote(dir_path):
            model_path = resolve_model(dir_path, cache_dir=cache_dir, token=token, model_name=model_name)
        elif os.path.isfile(os.path.join(dir_path, CONFIG_FILE_NAME)):
            model_path = dir_path
        else:
            model_path = os.path.join(dir_path, model_name)
            fetch_model(model_name, dir_path, self.registry, token=token)
        self._load_from_directory(model_path, weight_file)

    def _load_from_directory(self, model_path, weight_file=SAFETENSORS_WEIGHT_FILE_NAME):
        self.model_config = ModelConfig.load(os.path.join(model_path, CONFIG_FILE_NAME))
        if self.window_stride is not None:
            self.model_config.window_stride = self.window_stride

        if self.model_config.embeddings_name is not None:
            self.embeddings = Embeddings(
                self.model_config.embeddings_name,
                resource_registry=self.registry,
                use_cache=False,
            )
            self.model_config.word_embedding_size = self.embeddings.embed_size * tokens_per_position(
                self.model_config.text_features_indices
            )
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

        weight_path = find_weight_file(model_path, weight_file)
        print(f"Loading weights from {weight_path}")
        load_weights(self.model, weight_path, device=self.device)
        self.model.to(self.device)

        print(f"Model loaded: {self.model_config.architecture}")
        # what the model was trained with, which its configuration tells, not the command line
        if self.model_config.transformer_name is not None:
            print(f"Transformer: {self.model_config.transformer_name}")
        elif self.model_config.embeddings_name is not None:
            print(f"Word embeddings: {self.model_config.embeddings_name}")
        else:
            print("Word embeddings: none (char-only)")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


def next_n_lines(file_opened, N):
    """Read next N lines from file."""
    return [x.strip() for x in islice(file_opened, N)]
