"""
Data loading utilities for text classification.

Supports both legacy embedding-vector approach and new preprocessor-based index approach.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from delft.textClassification.preprocess import to_indices_single, to_vector_single
from delft.utilities.dataloader_utils import (
    effective_num_workers as _effective_num_workers,
)
from delft.utilities.dataloader_utils import (
    safe_multiprocessing_context as _safe_multiprocessing_context,
)


def _worker_init_fn(worker_id):
    # Each fork-mode worker inherits the parent's LMDB env handle, which is unsafe.
    # Re-open per worker so each gets an independent reader-locktable slot.
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    embeddings = getattr(info.dataset, "embeddings", None)
    if embeddings is not None and hasattr(embeddings, "reopen_lmdb"):
        embeddings.reopen_lmdb()


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification.

    Supports two modes:
    1. Preprocessor mode: texts are converted to word indices (new approach)
    2. Legacy mode: texts are converted to embedding vectors directly
    """

    def __init__(
        self,
        x,
        y,
        model_config,
        embeddings=None,
        transformer_tokenizer=None,
        preprocessor=None,
    ):
        """
        Initialize the dataset.

        Args:
            x: input texts
            y: labels (one-hot encoded)
            model_config: model configuration
            embeddings: Embeddings instance (legacy mode)
            transformer_tokenizer: transformer tokenizer (BERT mode)
            preprocessor: TextPreprocessor instance (new mode)
        """
        self.x = x
        self.y = y
        self.model_config = model_config
        self.embeddings = embeddings
        self.transformer_tokenizer = transformer_tokenizer
        self.preprocessor = preprocessor
        self.maxlen = model_config.maxlen
        self.list_classes = model_config.list_classes
        self.bert_data = model_config.transformer_name is not None

        # Determine mode
        self.use_preprocessor = preprocessor is not None and not self.bert_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text = self.x[idx]

        # Prepare Input
        if self.bert_data:
            # BERT mode: use transformer tokenizer
            inputs = self.transformer_tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.maxlen,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Squeeze to remove batch dimension added by return_tensors='pt'
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        elif self.use_preprocessor:
            # New preprocessor mode: return word indices
            indices = to_indices_single(text, self.preprocessor.vocab_word, self.maxlen)
            inputs = torch.tensor(indices, dtype=torch.long)
        else:
            # Legacy mode: return embedding vectors directly
            vector = to_vector_single(text, self.embeddings, self.maxlen)
            inputs = torch.tensor(vector, dtype=torch.float32)

        # Prepare Label
        if self.y is not None:
            label = self.y[idx]
            # Labels are one-hot encoded for multi-label classification
            target = torch.tensor(label, dtype=torch.float32)
            return inputs, target
        else:
            return inputs


def create_dataloader(
    x,
    y,
    model_config,
    embeddings=None,
    transformer_tokenizer=None,
    preprocessor=None,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    role="loader",
):
    """
    Create a DataLoader for text classification.

    Args:
        x: input texts
        y: labels
        model_config: model configuration
        embeddings: Embeddings instance (legacy mode)
        transformer_tokenizer: transformer tokenizer (BERT mode)
        preprocessor: TextPreprocessor instance (new mode)
        batch_size: batch size
        shuffle: whether to shuffle data
        num_workers: number of worker processes for data loading
        pin_memory: whether to pin host memory (only effective on CUDA)
        role: short label used in worker-count log lines

    Returns:
        DataLoader instance
    """
    dataset = TextClassificationDataset(
        x,
        y,
        model_config,
        embeddings=embeddings,
        transformer_tokenizer=transformer_tokenizer,
        preprocessor=preprocessor,
    )

    effective_workers = _effective_num_workers(num_workers, len(dataset), batch_size, role=role)
    # pin_memory only helps CUDA host->device transfers; on MPS/CPU it's overhead.
    effective_pin_memory = pin_memory and torch.cuda.is_available()
    mp_context = _safe_multiprocessing_context() if effective_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        persistent_workers=effective_workers > 0,
        pin_memory=effective_pin_memory,
        worker_init_fn=_worker_init_fn if effective_workers > 0 else None,
        multiprocessing_context=mp_context,
    )

    return loader
