"""
PyTorch Dataset and DataLoader for DeLFT sequence labeling models.

Replaces the Keras data generators with PyTorch equivalents.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List, Tuple

from delft.utilities.Utilities import truncate_batch_values, len_until_first_pad
from delft.utilities.numpy import shuffle_triple_with_view
from delft.sequenceLabelling.preprocess import (
    to_vector_single,
    to_casing_single,
    Preprocessor,
    BERTPreprocessor,
)
from delft.utilities.Tokenizer import tokenizeAndFilterSimple


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.

    Pads sequences to the maximum length in the batch.
    """
    # Separate inputs and labels
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch if item[1] is not None]

    if not labels:
        labels = None

    # Process inputs
    if isinstance(inputs[0], dict):
        # Dictionary of tensors
        collated_inputs = {}
        for key in inputs[0].keys():
            # Check if we need to pad (for variable length sequences)
            # length key is just a scalar, so we stack
            if key == "length":
                values = [inp[key] for inp in inputs]
                collated_inputs[key] = (
                    torch.stack(values).squeeze(-1)
                    if values[0].dim() > 0
                    else torch.stack(values)
                )
            elif key == "char_input":
                # char_input is (seq_len, char_seq_len), we need to pad the first dim
                # Actually char_input is usually (seq_len, max_char_len)
                values = [inp[key] for inp in inputs]
                max_len = max(v.shape[0] for v in values)
                char_dim = values[0].shape[1]
                padded_values = []
                for v in values:
                    pad_len = max_len - v.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros((pad_len, char_dim), dtype=v.dtype)
                        padded_values.append(torch.cat([v, padding]))
                    else:
                        padded_values.append(v)
                collated_inputs[key] = torch.stack(padded_values)
            else:
                # word_input, casing_input, features_input - all (seq_len, ...)
                # we need to pad dim 0
                values = [inp[key] for inp in inputs]
                if isinstance(values[0], torch.Tensor):
                    # Pad tokens
                    collated_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True
                    )
                else:
                    collated_inputs[key] = values

        if labels is not None:
            # Pad labels
            collated_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        else:
            collated_labels = None

        return collated_inputs, collated_labels
    else:
        # Tuple/list behavior - falling back to simple stack but warning if not padded
        # This branch might be unused if we always return dict in dataset
        num_inputs = len(inputs[0])
        collated_inputs = []
        for i in range(num_inputs):
            values = [inp[i] for inp in inputs]
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() > 0:
                    collated_inputs.append(
                        torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
                    )
                else:
                    collated_inputs.append(torch.stack(values))
            else:
                collated_inputs.append(values)

        if labels is not None:
            collated_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        else:
            collated_labels = None

        return collated_inputs, collated_labels


class SequenceLabelingDataset(Dataset):
    """
    PyTorch Dataset for sequence labeling with word embeddings.

    Replaces the Keras DataGenerator class.

    Args:
        x: Input sequences (list of token lists or raw texts)
        y: Labels (list of tag lists, None for inference)
        preprocessor: DeLFT Preprocessor instance
        embeddings: Embeddings instance
        char_embed_size: Character embedding size
        max_sequence_length: Maximum sequence length (truncates longer sequences)
        tokenize: Whether to tokenize input texts
        features: Optional additional features
        use_chain_crf: Whether using ChainCRF (affects label encoding)
    """

    def __init__(
        self,
        x: List,
        y: Optional[List] = None,
        preprocessor: Preprocessor = None,
        embeddings=None,
        char_embed_size: int = 25,
        max_sequence_length: Optional[int] = None,
        tokenize: bool = False,
        features: Optional[List] = None,
        use_chain_crf: bool = False,
    ):
        self.x = x
        self.y = y
        self.features = features
        self.preprocessor = preprocessor
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.max_sequence_length = max_sequence_length
        self.tokenize = tokenize
        self.use_chain_crf = use_chain_crf

        if preprocessor:
            self.labels = preprocessor.vocab_tag

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[dict, Optional[torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (inputs_dict, labels_tensor)
        """
        # Get raw data
        x_item = self.x[idx]
        y_item = self.y[idx] if self.y is not None else None
        f_item = self.features[idx] if self.features is not None else None

        # Tokenize if needed
        if self.tokenize:
            x_tokens = tokenizeAndFilterSimple(x_item)
        else:
            x_tokens = x_item

        # Truncate if needed
        seq_len = len(x_tokens)
        if self.max_sequence_length and seq_len > self.max_sequence_length:
            x_tokens = x_tokens[: self.max_sequence_length]
            if y_item is not None:
                y_item = y_item[: self.max_sequence_length]
            if f_item is not None:
                f_item = f_item[: self.max_sequence_length]
            seq_len = self.max_sequence_length

        # Prevent length 1 sequences (causes issues with CRF)
        extend = seq_len == 1
        if extend:
            seq_len = 2

        # Get word embeddings
        word_emb = to_vector_single(x_tokens, self.embeddings, seq_len)

        # Get character indices
        if self.preprocessor.return_chars:
            char_indices = self.preprocessor.transform_chars([x_tokens], extend=extend)[
                0
            ]
        else:
            char_indices = np.zeros(
                (seq_len, self.preprocessor.max_char_length), dtype=np.int32
            )

        # Get casing features
        if self.preprocessor.return_casing:
            casing = to_casing_single(x_tokens, seq_len)
        else:
            casing = np.zeros(seq_len, dtype=np.int32)

        # Get additional features
        if self.preprocessor.return_features and f_item is not None:
            features = self.preprocessor.transform_features([f_item], extend=extend)[0]
        else:
            features = np.zeros((seq_len, 1), dtype=np.int32)

        # Process labels
        if y_item is not None:
            if self.use_chain_crf:
                _, labels = self.preprocessor.transform(
                    [x_tokens], [y_item], extend=extend, label_indices=False
                )
            else:
                _, labels = self.preprocessor.transform(
                    [x_tokens], [y_item], extend=extend, label_indices=True
                )
            labels = np.array(labels[0], dtype=np.int64)
        else:
            labels = None

        # Convert to tensors
        inputs = {
            "word_input": torch.from_numpy(word_emb).float(),
            "char_input": torch.from_numpy(np.array(char_indices)).long(),
            "length": torch.tensor([seq_len], dtype=torch.long),
        }

        if self.preprocessor.return_casing:
            inputs["casing_input"] = torch.from_numpy(casing).long()

        if self.preprocessor.return_features:
            inputs["features_input"] = torch.from_numpy(features).long()

        if labels is not None:
            labels = torch.from_numpy(labels).long()

        return inputs, labels


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for sequence labeling with transformer embeddings.

    Replaces the Keras DataGeneratorTransformers class.

    Args:
        x: Input sequences (list of token lists or raw texts)
        y: Labels (list of tag lists, None for inference)
        preprocessor: DeLFT Preprocessor instance
        bert_preprocessor: BERTPreprocessor instance
        max_sequence_length: Maximum sequence length
        tokenize: Whether to tokenize input texts
        features: Optional additional features
        use_chain_crf: Whether using ChainCRF
        output_input_offsets: Whether to output token offsets for alignment
    """

    def __init__(
        self,
        x: List,
        y: Optional[List] = None,
        preprocessor: Preprocessor = None,
        bert_preprocessor: BERTPreprocessor = None,
        max_sequence_length: Optional[int] = None,
        tokenize: bool = False,
        features: Optional[List] = None,
        use_chain_crf: bool = False,
        output_input_offsets: bool = False,
    ):
        self.x = x
        self.y = y
        self.features = features
        self.preprocessor = preprocessor
        self.bert_preprocessor = bert_preprocessor
        self.max_sequence_length = max_sequence_length
        self.tokenize = tokenize
        self.use_chain_crf = use_chain_crf
        self.output_input_offsets = output_input_offsets

        if bert_preprocessor and bert_preprocessor.empty_features_vector is None:
            bert_preprocessor.empty_features_vector = (
                preprocessor.empty_features_vector()
            )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[dict, Optional[torch.Tensor]]:
        """Get a single sample."""
        x_item = self.x[idx]
        y_item = [self.y[idx]] if self.y is not None else None
        f_item = [self.features[idx]] if self.features is not None else None

        # Tokenize if needed
        if self.tokenize:
            x_tokens = [tokenizeAndFilterSimple(x_item)]
        else:
            x_tokens = [x_item]

        # Truncate if needed
        seq_len = len(x_tokens[0])
        if self.max_sequence_length and seq_len > self.max_sequence_length:
            x_tokens = [x_tokens[0][: self.max_sequence_length]]
            if y_item is not None:
                y_item = [y_item[0][: self.max_sequence_length]]
            if f_item is not None:
                f_item = [f_item[0][: self.max_sequence_length]]

        # Get character indices
        batch_c = self.preprocessor.transform(x_tokens)[0]

        # Process features
        if self.preprocessor.return_features and f_item is not None:
            sub_f = self.preprocessor.transform_features(f_item)
        else:
            sub_f = None

        # Tokenize and align for transformer
        (
            input_ids,
            token_type_ids,
            attention_mask,
            input_chars,
            input_features,
            input_labels,
            input_offsets,
        ) = self.bert_preprocessor.tokenize_and_align_features_and_labels(
            x_tokens, batch_c, sub_f, y_item, maxlen=self.max_sequence_length
        )

        # Get actual length
        actual_len = len_until_first_pad(input_ids[0], 0)

        # Process labels
        if y_item is not None:
            _, labels = self.preprocessor.transform(
                x_tokens, input_labels, label_indices=True
            )
            labels = np.array(labels[0][:actual_len], dtype=np.int64)
        else:
            labels = None

        # Convert to tensors
        inputs = {
            "input_ids": torch.tensor(input_ids[0][:actual_len], dtype=torch.long),
            "token_type_ids": torch.tensor(
                token_type_ids[0][:actual_len], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_mask[0][:actual_len], dtype=torch.long
            ),
        }

        if self.preprocessor.return_chars:
            inputs["char_input"] = torch.tensor(
                input_chars[0][:actual_len], dtype=torch.long
            )

        if self.preprocessor.return_features:
            inputs["features_input"] = torch.tensor(
                input_features[0][:actual_len], dtype=torch.long
            )

        if self.output_input_offsets:
            inputs["input_offsets"] = input_offsets[0][:actual_len]

        if labels is not None:
            labels = torch.from_numpy(labels).long()

        return inputs, labels


def create_dataloader(
    x,
    y=None,
    batch_size: int = 24,
    preprocessor: Preprocessor = None,
    embeddings=None,
    features=None,
    shuffle: bool = True,
    model_config=None,
    num_workers: int = 0,
    pin_memory: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for a DeLFT dataset.
    Factory method that creates the appropriate Dataset based on configuration.
    
    Args:
        distributed: If True, use DistributedSampler for multi-GPU training
    """
    if model_config and model_config.transformer_name:
        from transformers import AutoTokenizer
        # Initialize BERT/Transformer preprocessor
        tokenizer = AutoTokenizer.from_pretrained(model_config.transformer_name)
        bert_preprocessor = BERTPreprocessor(tokenizer)

        dataset = TransformerDataset(
            x,
            y,
            preprocessor=preprocessor,
            bert_preprocessor=bert_preprocessor,
            max_sequence_length=model_config.max_sequence_length,
            tokenize=False,  # Input x is usually already tokenized in DeLFT list-of-lists format
            features=features,
            use_chain_crf=model_config.use_crf if model_config else False,
        )

        # Use DistributedSampler for multi-GPU training
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    # Default to SequenceLabelingDataset for now which covers RNNs
    # TODO: Add TransformerDataset logic if needed by checking model_config

    dataset = SequenceLabelingDataset(
        x,
        y,
        preprocessor=preprocessor,
        embeddings=embeddings,
        features=features,
        max_sequence_length=model_config.max_sequence_length if model_config else None,
        use_chain_crf=model_config.use_crf if model_config else False,
    )

    # Use DistributedSampler for multi-GPU training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


class BatchedDataLoader:
    """
    A DataLoader-like wrapper that processes data in batches similar to Keras generators.

    This provides a more direct port of the Keras generator behavior for compatibility.

    Args:
        x: Input sequences
        y: Labels (optional)
        batch_size: Batch size
        preprocessor: DeLFT Preprocessor
        embeddings: Embeddings instance
        max_sequence_length: Maximum sequence length
        shuffle: Whether to shuffle data each epoch
        features: Optional features
        use_chain_crf: Whether using ChainCRF
    """

    def __init__(
        self,
        x: List,
        y: Optional[List] = None,
        batch_size: int = 24,
        preprocessor: Preprocessor = None,
        embeddings=None,
        max_sequence_length: Optional[int] = None,
        shuffle: bool = True,
        features: Optional[List] = None,
        use_chain_crf: bool = False,
        tokenize: bool = False,
    ):
        self.original_x = self.x = x
        self.original_y = self.y = y
        self.original_features = self.features = features
        self.preprocessor = preprocessor
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        self.use_chain_crf = use_chain_crf
        self.tokenize = tokenize

        self._current_batch = 0

    def __len__(self) -> int:
        if self.original_x is None:
            return 0
        return int(np.ceil(len(self.original_x) / self.batch_size))

    def __iter__(self):
        self._current_batch = 0
        if self.shuffle and self.y is not None:
            self.x, self.y, self.features = shuffle_triple_with_view(
                self.original_x, self.original_y, self.original_features
            )
        return self

    def __next__(self) -> Tuple[dict, Optional[torch.Tensor]]:
        if self._current_batch >= len(self):
            raise StopIteration

        batch = self._get_batch(self._current_batch)
        self._current_batch += 1
        return batch

    def _get_batch(self, index: int) -> Tuple[dict, Optional[torch.Tensor]]:
        """Generate one batch of data."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.x))

        # Get batch data
        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx] if self.y is not None else None
        batch_f = (
            self.features[start_idx:end_idx] if self.features is not None else None
        )

        # Tokenize if needed
        if self.tokenize:
            batch_x = [tokenizeAndFilterSimple(text) for text in batch_x]

        # Get max length
        max_len = max(len(tokens) for tokens in batch_x)
        if self.max_sequence_length and max_len > self.max_sequence_length:
            max_len = self.max_sequence_length
            batch_x = truncate_batch_values(batch_x, max_len)
            if batch_y is not None:
                batch_y = truncate_batch_values(batch_y, max_len)
            if batch_f is not None:
                batch_f = truncate_batch_values(batch_f, max_len)

        # Prevent length 1 sequences
        extend = max_len == 1
        if extend:
            max_len = 2

        batch_size = len(batch_x)

        # Word embeddings
        word_emb = np.zeros(
            (batch_size, max_len, self.embeddings.embed_size), dtype="float32"
        )
        for i, tokens in enumerate(batch_x):
            word_emb[i] = to_vector_single(tokens, self.embeddings, max_len)

        # Character indices
        batches = self.preprocessor.transform(batch_x, extend=extend)
        char_indices = np.array(batches[0], dtype=np.int32)
        lengths = batches[1]

        # Labels
        if batch_y is not None:
            if self.use_chain_crf:
                _, labels = self.preprocessor.transform(
                    batch_x, batch_y, extend=extend, label_indices=False
                )
            else:
                _, labels = self.preprocessor.transform(
                    batch_x, batch_y, extend=extend, label_indices=True
                )
            labels = np.array(truncate_batch_values(labels, max_len), dtype=np.int64)
        else:
            labels = None

        # Convert to tensors
        inputs = {
            "word_input": torch.from_numpy(word_emb).float(),
            "char_input": torch.from_numpy(char_indices).long(),
            "length": torch.from_numpy(np.array(lengths)).long(),
        }

        if self.preprocessor.return_casing:
            casing = np.zeros((batch_size, max_len), dtype=np.int32)
            for i, tokens in enumerate(batch_x):
                casing[i] = to_casing_single(tokens, max_len)
            inputs["casing_input"] = torch.from_numpy(casing).long()

        if self.preprocessor.return_features and batch_f is not None:
            features = self.preprocessor.transform_features(batch_f, extend=extend)
            inputs["features_input"] = torch.from_numpy(np.array(features)).long()

        if labels is not None:
            labels = torch.from_numpy(labels).long()

        return inputs, labels
