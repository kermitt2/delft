import os
import re
import time
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from delft import DELFT_PROJECT_DIR, default_nb_workers
from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.data_loader import create_dataloader
from delft.textClassification.models import DEFAULT_WORD_EMBEDDING_SIZE, getModel
from delft.textClassification.preprocess import TextPreprocessor
from delft.textClassification.reader import TextsOnDisk
from delft.textClassification.trainer import Trainer
from delft.utilities.cuda_setup import configure_cudnn_for_device, validate_device_arch_compatibility
from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.utilities.hub_models import fetch_model, is_remote, resolve_model
from delft.utilities.misc import print_parameters, to_wandb_table
from delft.utilities.numpy import shuffle_triple_with_view
from delft.utilities.transformer_tokenizers import get_tokenizer
from delft.utilities.Utilities import pick_device
from delft.utilities.weights import (
    SAFETENSORS_WEIGHT_FILE_NAME,
    WEIGHT_FILE_NAMES,
    find_weight_file,
    load_weights,
    save_weights,
)


def split_train_validation(x_train, y_train, split_ratio=0.9):
    """
    Carve a validation set off the end of the training data.

    The data is shuffled first, keeping x and y aligned: the split is positional,
    so class-ordered input would leave whole classes out of validation, and a
    class holding a single label value there has no defined ROC-AUC (see
    delft.textClassification.trainer.compute_roc_auc).

    Returns:
        (x_train, y_train, x_valid, y_valid)
    """
    if not isinstance(x_train, TextsOnDisk):  # np.asarray would read every text into memory
        x_train = np.asarray(x_train)
    x_train, y_train, _ = shuffle_triple_with_view(x_train, np.asarray(y_train))
    split_idx = int(len(x_train) * split_ratio)
    return (
        x_train[:split_idx],
        y_train[:split_idx],
        x_train[split_idx:],
        y_train[split_idx:],
    )


# File names for saving/loading
PREPROCESSOR_FILE = "preprocessor.json"

# the weights of the model of a fold, of a classifier trained over several folds
FOLD_WEIGHT_FILE_PATTERN = re.compile(r"(?P<stem>.+)_fold(?P<fold>\d+)(?P<extension>\.[A-Za-z]+)")


def fold_weight_file(weight_file, fold_id):
    """The name of the weights of the model of fold ``fold_id``: model.safetensors gives model_fold0.safetensors."""
    stem, extension = os.path.splitext(weight_file)
    return f"{stem}_fold{fold_id}{extension}"


def remove_weights_except(directory, kept):
    """
    Remove from a model directory the weights of a model or of its folds that are not
    among the files ``kept``, just written: those of an earlier training in another
    format, with another number of folds or without folds, which would otherwise stay
    next to the new ones.
    """
    for name in os.listdir(directory):
        match = FOLD_WEIGHT_FILE_PATTERN.fullmatch(name)
        of_a_fold = match is not None and match.group("stem") + match.group("extension") in WEIGHT_FILE_NAMES
        if (name in WEIGHT_FILE_NAMES or of_a_fold) and name not in kept:
            os.remove(os.path.join(directory, name))


class Classifier(object):
    config_file = "config.json"
    # name of pickled weights, the format written up to DeLFT 1.1.0 and still loaded
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
        wandb_project: str = None,
        nb_workers: int = None,
        short_model_name: str = None,
    ):
        self.short_model_name = short_model_name
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
        self.wandb_project = wandb_project
        self.wandb = None

        self.device = pick_device(device)
        validate_device_arch_compatibility(self.device)
        configure_cudnn_for_device(self.device)

        self.registry = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))

        if embeddings_name is not None:
            self.embeddings = Embeddings(embeddings_name, resource_registry=self.registry)
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.model_config.word_embedding_size = 0

        # Set number of DataLoader worker processes. Per-call
        # ``create_dataloader`` further auto-caps based on dataset size;
        # default to a modest 4 to avoid the per-epoch spawn cost dominating
        # tiny classification runs. ``nb_workers=0`` means in-process loading,
        # with no worker process spawned at all — the value to use when DeLFT
        # runs embedded in a host process such as GROBID (see ``predict()``).
        self.nb_workers_explicit = nb_workers is not None
        if nb_workers is None:
            # the cores this process may run on, not those of the node: a SLURM task
            # allocated one or two of them was spawning four workers on them. On a
            # single core the data is loaded in the process itself: a worker would
            # only share that core with it.
            self.nb_workers = default_nb_workers()
        else:
            self.nb_workers = max(0, nb_workers)

        if report_to_wandb:
            self._init_wandb(model_name)

    def _init_wandb(self, model_name, run_id=None):
        """Initialize Weights & Biases logging."""
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
            project = self.wandb_project or os.getenv("WANDB_PROJECT")

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
                        "batch_size": self.training_config.batch_size,
                        "learning_rate": self.training_config.learning_rate,
                        "max_epoch": self.training_config.max_epoch,
                        "patience": self.training_config.patience,
                        "early_stop": self.training_config.early_stop,
                        "maxlen": self.model_config.maxlen,
                    },
                )
            self.wandb = wandb
        except ImportError:
            print("Warning: wandb not available")
            self.report_to_wandb = False

    def train(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        """
        Train on the texts ``x_train`` with the labels ``y_train``: one model, or one per
        fold when the classifier has several (``fold_number``, see ``train_nfold``). With
        ``incremental``, the training goes on from what was loaded (see ``load``), with
        its classes, embeddings and architecture, rather than from new models.
        """
        if self.model_config.fold_number == 1:
            self.train_single(x_train, y_train, vocab_init, incremental, callbacks)
        else:
            self.train_nfold(x_train, y_train, vocab_init, incremental, callbacks)

    def train_single(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        if incremental and self.model is None:
            # go on from the loaded model: the argument was accepted and ignored, and a
            # new model was trained in its place
            raise ValueError("Incremental training starts from a model: load one first")
        self._prepare_training(x_train, y_train, incremental)

        # the last tenth of the texts, shuffled, is the validation set of early stopping
        x_valid = None
        y_valid = None
        if self.training_config.early_stop:
            x_train, y_train, x_valid, y_valid = split_train_validation(x_train, y_train)

        if incremental:
            print("Incremental training from loaded model", self.model_config.model_name)
        else:
            self.model = getModel(self.model_config, self.training_config)
        self.models = None

        print(f"Model: {self.model_config.architecture}")
        self._train_model(self.model, x_train, y_train, x_valid, y_valid)

    def train_nfold(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        """
        Train one model per fold, as DeLFT did with Keras: the texts are cut into
        ``fold_number`` folds, and the model of a fold is trained on the texts of the
        other folds, the fold itself being its validation set when early stopping is on.
        The models then classify together, see ``predict``.

        The texts are shuffled first, as for the validation set of a single model: cut as
        they come, texts ordered by class would leave whole classes out of a fold. With
        ``incremental``, the training of the loaded fold models goes on.
        """
        fold_count = self.model_config.fold_number
        if fold_count < 2:
            raise ValueError(f"Training over folds needs at least 2 of them, fold_number is {fold_count}")
        if len(x_train) < fold_count:
            raise ValueError(f"{len(x_train)} texts cannot be cut into {fold_count} folds")
        if incremental and (not self.models or len(self.models) != fold_count):
            raise ValueError(f"Incremental training over {fold_count} folds starts from their models: load them first")
        self._prepare_training(x_train, y_train, incremental)

        if incremental:
            print("Incremental n-fold training from loaded models", self.model_config.model_name)
        else:
            self.models = []
        self.model = None

        if not isinstance(x_train, TextsOnDisk):  # np.asarray would read every text into memory
            x_train = np.asarray(x_train)
        x_train, y_train, _ = shuffle_triple_with_view(x_train, np.asarray(y_train))
        indices = np.arange(len(x_train))
        fold_size = len(x_train) // fold_count

        print(f"Model: {self.model_config.architecture}, {fold_count} folds")
        for fold_id in range(fold_count):
            fold_start = fold_size * fold_id
            # the last fold takes the texts the division left over
            fold_end = len(x_train) if fold_id == fold_count - 1 else fold_start + fold_size
            train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])

            print(f"\n------------------------ fold {fold_id} --------------------------------------")
            model = self.models[fold_id] if incremental else getModel(self.model_config, self.training_config)
            x_valid = y_valid = None
            if self.training_config.early_stop:
                x_valid, y_valid = x_train[fold_start:fold_end], y_train[fold_start:fold_end]
            self._train_model(
                model, x_train[train_indices], y_train[train_indices], x_valid, y_valid, role=f"fold{fold_id}-"
            )
            if not incremental:
                self.models.append(model)
            if self._parks_fold_models():
                model.to("cpu")

    def _prepare_training(self, x_train, y_train, incremental):
        """
        What a training needs before its models are built: the vocabulary of a model that
        learns its word embeddings, or, when going on from loaded models, data that has
        their classes.
        """
        if incremental:
            nb_classes = np.asarray(y_train).shape[-1]
            if nb_classes != len(self.model_config.list_classes):
                raise ValueError(
                    f"The loaded model {self.model_config.model_name} has {len(self.model_config.list_classes)} "
                    f"classes, and the training data {nb_classes}"
                )
        elif self._learns_word_embeddings():
            # No pre-trained word embeddings and no transformer: the model learns the
            # embeddings of the words of its training texts, as a sequence labelling
            # model given no embeddings learns from the characters alone.
            print("No word embeddings: they are learned from the training texts")
            self.preprocessor = TextPreprocessor(maxlen=self.model_config.maxlen)
            self.preprocessor.fit(x_train)
            self.model_config.vocab_size = len(self.preprocessor.vocab_word)
            self.model_config.word_embedding_size = DEFAULT_WORD_EMBEDDING_SIZE
        else:
            self.preprocessor = None
            self.model_config.vocab_size = None

    def _train_model(self, model, x_train, y_train, x_valid=None, y_valid=None, role=""):
        """Train ``model`` on the training texts, with the validation texts when there are some."""
        model.to(self.device)

        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            transformer_tokenizer = get_tokenizer(self.model_config.transformer_name)

        train_loader = create_dataloader(
            x_train,
            y_train,
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            preprocessor=self.preprocessor,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.nb_workers,
            role=f"{role}train",
        )

        valid_loader = None
        if x_valid is not None:
            valid_loader = create_dataloader(
                x_valid,
                y_valid,
                self.model_config,
                embeddings=self.embeddings,
                transformer_tokenizer=transformer_tokenizer,
                preprocessor=self.preprocessor,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.nb_workers,
                role=f"{role}valid",
            )

        # Ensure model output directory exists for checkpoints
        model_dir = self._get_model_dir()
        os.makedirs(model_dir, exist_ok=True)

        trainer = Trainer(
            model,
            self.model_config,
            self.training_config,
            device=str(self.device),
            checkpoint_path=model_dir,
        )
        trainer.train(train_loader, valid_loader)

    def _learns_word_embeddings(self):
        """
        Whether the model is given neither pre-trained word embeddings nor a transformer:
        it then learns the embeddings of the words of its training texts.
        """
        return (
            self.embeddings is None
            and self.model_config.transformer_name is None
            and self.model_config.architecture != "bert"
        )

    def _trained_models(self):
        """The model of the classifier, or the models of its folds."""
        if self.model_config.fold_number > 1:
            if not self.models:
                raise OSError("Could not find nfolds models.")
            return self.models
        if self.model is None:
            raise OSError("Model not loaded")
        return [self.model]

    def _parks_fold_models(self):
        """
        Whether the fold models are kept out of the GPU, each being moved there for the
        time it runs: several transformers may not fit in it together.
        """
        return (
            self.model_config.fold_number > 1
            and self.model_config.transformer_name is not None
            and self.device.type != "cpu"
        )

    @contextmanager
    def _on_device(self, model):
        if not self._parks_fold_models():
            # moved to the device once, when trained or loaded
            yield model
            return
        model.to(self.device)
        try:
            yield model
        finally:
            model.to("cpu")

    def _predict_probabilities(self, loader, with_labels=False):
        """
        The probability of every class for every text of ``loader`` and, ``with_labels``,
        the labels the loader gives with the texts.

        With several folds, the probability of a class is the geometric mean of the
        probabilities the fold models give it, as in DeLFT with Keras.
        """
        fold_probabilities = []
        y_true = None
        for model in self._trained_models():
            model.eval()
            predictions, labels = [], []
            with self._on_device(model), torch.no_grad():
                for batch in loader:
                    if with_labels:
                        inputs, batch_labels = batch
                        labels.append(batch_labels.cpu().numpy())
                    else:
                        inputs = batch
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(self.device)
                    predictions.append(torch.sigmoid(model(inputs)["logits"]).cpu().numpy())
            fold_probabilities.append(np.concatenate(predictions, axis=0))
            if with_labels and y_true is None:
                y_true = np.concatenate(labels, axis=0)

        if len(fold_probabilities) == 1:
            return fold_probabilities[0], y_true
        product = np.prod(np.stack(fold_probabilities).astype(np.float64), axis=0)
        return (product ** (1.0 / len(fold_probabilities))).astype(np.float32), y_true

    def eval(self, x_test, y_test):
        """Evaluate model on test data.

        Args:
            x_test: Test texts
            y_test: Test labels (numpy array with shape [n_samples, n_classes])
        """
        print_parameters(self.model_config, self.training_config)

        self._trained_models()  # an error when there is none

        # Get transformer tokenizer if needed
        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            # loaded once for the process, not for every call
            transformer_tokenizer = get_tokenizer(self.model_config.transformer_name)

        # Create dataloader
        test_loader = create_dataloader(
            x_test,
            y_test,
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            preprocessor=self.preprocessor,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.nb_workers,
            role="eval",
        )

        # of the model, or of the models of the folds together
        y_pred_probs, y_true = self._predict_probabilities(test_loader, with_labels=y_test is not None)

        if y_true is None:
            print("No labels provided for evaluation")
            return

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_probs > 0.5).astype(int)

        # Calculate per-class metrics
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred_binary, average=None)

        # Print results
        print("\n-----------------------------------------------")
        print(f"Evaluation on {len(x_test)} instances:")
        print(f"{'':>14}  {'precision':>12}  {'recall':>12}  {'f-score':>12}  {'support':>12}")

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
        micro_precision, micro_recall, _, _ = precision_recall_fscore_support(y_true_flat, y_pred_flat, average="micro")

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

    def predict(self, texts, output_format="json", use_main_thread_only=False, batch_size=None, nb_workers=None):
        """Classify texts with the model.

        ``nb_workers`` is the number of DataLoader worker processes to use for
        this call. It falls back to the value given to the constructor, and to
        0 (in-process, no worker process spawned) when neither was set — a
        DataLoader is built per call, so worker processes are respawned on
        every predict() and rarely pay for themselves at inference time.
        ``use_main_thread_only=True`` forces 0 whatever the other settings;
        callers embedding DeLFT in a host process (e.g. GROBID through JEP)
        should use either.
        """
        if batch_size is not None:
            self.model_config.batch_size = batch_size

        if use_main_thread_only:
            nb_workers = 0
        elif nb_workers is None:
            nb_workers = self.nb_workers if self.nb_workers_explicit else 0

        # The models are moved to self.device once, at load()/train() time; no
        # need to walk their parameters again on every predict() call.
        self._trained_models()  # an error when there is none

        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
            # loaded once for the process, not for every call
            transformer_tokenizer = get_tokenizer(self.model_config.transformer_name)

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
            preprocessor=self.preprocessor,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=nb_workers,
            role="predict",
        )

        # of the model, or of the models of the folds together
        result, _ = self._predict_probabilities(data_loader)

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

    def save(self, dir_path="data/models/textClassification/", weight_file=None):
        """Save model to disk.

        The weights are saved as safetensors, or as a pickled state dict when
        ``weight_file`` does not end with ``.safetensors`` (see
        ``delft.utilities.weights``). A classifier trained over several folds saves the
        weights of every fold model, ``model_fold0.safetensors`` and so on. Weights the
        directory holds from a previous training, in the other format or for another
        number of folds, are removed.
        """
        weight_file = weight_file or SAFETENSORS_WEIGHT_FILE_NAME
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, self.config_file))

        # Save preprocessor if present, and leave none of an earlier training otherwise
        if self.preprocessor is not None:
            self.preprocessor.save(os.path.join(directory, PREPROCESSOR_FILE))
            print("Preprocessor saved")
        elif os.path.isfile(os.path.join(directory, PREPROCESSOR_FILE)):
            os.remove(os.path.join(directory, PREPROCESSOR_FILE))

        # Save PyTorch model: its weights, or those of the model of every fold
        models = self._trained_models()
        if self.model_config.fold_number > 1:
            weight_files = [fold_weight_file(weight_file, fold_id) for fold_id in range(len(models))]
        else:
            weight_files = [weight_file]
        for model, name in zip(models, weight_files):
            save_weights(model, os.path.join(directory, name))
        remove_weights_except(directory, weight_files)
        print(f"Model saved to {directory}" + (f" ({len(models)} folds)" if len(models) > 1 else ""))

    def load(self, dir_path="data/models/textClassification/", cache_dir=None, token=None):
        """Load model from disk, from the Hugging Face Hub or over HTTP, its weights being
        in either format ``save`` writes.

        ``dir_path`` is either

        - a models directory, the model being the folder of it with the name of the
          model. When it is not there, it is first downloaded there from the Hub, if the
          resources registry gives it a place on the Hub. A model trained or copied there
          is never touched;
        - the directory of the model itself, whatever its name;
        - a repository or a bucket of the Hub, ``hf://owner/repository``, the model being
          the folder of it with the name of the model, or that folder itself,
          ``hf://owner/repository/model-name``, whatever the name of the model;
        - the URL of an archive of the model directory, ``https://.../model-name.zip``;
          or of the model directory as a folder of files, or of the folder holding it
          under the name of the model, or else ``{name of the model}.zip``.

        See ``delft.utilities.hub_models`` for the last two, whose model is downloaded
        to ``cache_dir`` when it is not there yet.
        """
        model_name = self.model_config.model_name
        if is_remote(dir_path):
            model_path = resolve_model(dir_path, cache_dir=cache_dir, token=token, model_name=model_name)
        elif os.path.isfile(os.path.join(dir_path, self.config_file)):
            model_path = dir_path
        else:
            model_path = os.path.join(dir_path, model_name)
            fetch_model(model_name, dir_path, self.registry, token=token)
        self._load_from_directory(model_path)

    def _load_from_directory(self, model_path):
        # Load config
        self.model_config = ModelConfig.load(os.path.join(model_path, self.config_file))

        # The vocabulary of a model that learned its word embeddings. A model that reads
        # pre-trained ones or a transformer has none, whatever file an earlier training
        # left in its directory.
        self.preprocessor = None
        preprocessor_path = os.path.join(model_path, PREPROCESSOR_FILE)
        if getattr(self.model_config, "vocab_size", None) and os.path.exists(preprocessor_path):
            self.preprocessor = TextPreprocessor.load(preprocessor_path)
            print("Preprocessor loaded")

        # Load embeddings if needed: those of the model, not those the wrapper was created with
        if self.model_config.embeddings_name is not None:
            self.embeddings = Embeddings(
                self.model_config.embeddings_name,
                resource_registry=self.registry,
            )
        else:
            self.embeddings = None

        if self.model_config.fold_number > 1:
            # the model of every fold: they classify together
            self.model = None
            self.models = []
            for fold_id in range(self.model_config.fold_number):
                weight_path = os.path.join(model_path, fold_weight_file(SAFETENSORS_WEIGHT_FILE_NAME, fold_id))
                if not os.path.isfile(weight_path):
                    weight_path = os.path.join(model_path, fold_weight_file(self.weight_file, fold_id))
                model = getModel(self.model_config, self.training_config)
                load_weights(model, weight_path, device=self.device)
                model.to("cpu" if self._parks_fold_models() else self.device)
                self.models.append(model)
            print(f"Models of {len(self.models)} folds loaded from {model_path}")
        else:
            # Init model
            self.model = getModel(self.model_config, self.training_config)
            self.models = None

            # Load weights
            weight_path = find_weight_file(model_path, SAFETENSORS_WEIGHT_FILE_NAME)
            load_weights(self.model, weight_path, device=self.device)
            self.model.to(self.device)
            print(f"Model loaded from {weight_path}")
        # what the model was trained with, which its configuration tells, not the command line
        if self.model_config.transformer_name is not None:
            print(f"Transformer: {self.model_config.transformer_name}")
        elif self.model_config.embeddings_name is not None:
            print(f"Word embeddings: {self.model_config.embeddings_name}")
        else:
            print("Word embeddings: none (learned from the training texts)")

    def _get_model_dir(self):
        return os.path.join("data/models/textClassification/", self.model_config.model_name)
