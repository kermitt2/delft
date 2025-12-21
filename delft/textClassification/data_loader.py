import torch
from torch.utils.data import Dataset, DataLoader
from delft.textClassification.preprocess import to_vector_single


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification.
    """

    def __init__(self, x, y, model_config, embeddings=None, transformer_tokenizer=None):
        self.x = x
        self.y = y
        self.model_config = model_config
        self.embeddings = embeddings
        self.transformer_tokenizer = transformer_tokenizer
        self.maxlen = model_config.maxlen
        self.list_classes = model_config.list_classes
        self.bert_data = model_config.transformer_name is not None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text = self.x[idx]

        # Prepare Input
        if self.bert_data:
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
        else:
            # Word embeddings
            # to_vector_single returns numpy array [maxlen, embed_size]
            vector = to_vector_single(text, self.embeddings, self.maxlen)
            inputs = torch.tensor(vector, dtype=torch.float32)

        # Prepare Label
        if self.y is not None:
            label = self.y[idx]
            # Assumes label is already one-hot or proper format.
            # In Keras generator: batch_y[i] = self.y[(index*self.batch_size)+i]
            # If y is categorical/one-hot, we might want to keep it float for BCEWithLogitsLoss
            # or long for CrossEntropyLoss (if single class index).
            # Looking at Keras models: loss='binary_crossentropy', activation='sigmoid'.
            # So labels can be multi-label one-hot.
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
    batch_size=32,
    shuffle=True,
):
    dataset = TextClassificationDataset(
        x,
        y,
        model_config,
        embeddings=embeddings,
        transformer_tokenizer=transformer_tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Simple for now
    )

    return loader
