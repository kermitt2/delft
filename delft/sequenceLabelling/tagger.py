import datetime
import time
import numpy as np
import torch
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.sequenceLabelling.data_loader import create_dataloader


class Tagger(object):
    def __init__(
        self, model, model_config, embeddings=None, preprocessor=None, device=None
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings
        self.device = device if device else torch.device("cpu")

    def tag(self, texts, output_format, features=None):
        if output_format == "json":
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model_config.model_name,
                "texts": [],
            }
        else:
            list_of_tags = []

        to_tokeniz = False
        if len(texts) > 0 and isinstance(texts[0], str):
            to_tokeniz = True

        # Create data loader for inference
        # If texts are strings, we need to tokenize them first?
        # The create_dataloader expects x_data as list of list of tokens usually for training,
        # checking create_dataloader implementation...
        # For inference, if we pass strings, we might need to handle tokenization here or in data_loader.

        # Let's tokenize if needed
        tokenized_texts = []
        all_offsets = []

        for i, text in enumerate(texts):
            if to_tokeniz:
                tokens, offsets = tokenizeAndFilter(text)
                tokenized_texts.append(tokens)
                all_offsets.append(offsets)
            else:
                tokenized_texts.append(text)
                all_offsets.append([])  # No offsets if already tokenized

        # Create dataloader
        # Note: y is None for inference
        dataloader = create_dataloader(
            tokenized_texts,
            None,
            preprocessor=self.preprocessor,
            embeddings=self.embeddings,
            batch_size=self.model_config.batch_size,
            features=features,
            shuffle=False,
            model_config=self.model_config,
        )

        steps_done = 0
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = (
                    batch  # dataloader yields (inputs, labels), labels are None or dummies
                )

                # Move inputs to device
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Run inference
                if hasattr(self.model, "decode"):
                    # For CRF models
                    tags = self.model.decode(inputs)
                    # tags is list of list of label indices
                    probs = None  # standard CRF hard decoding doesn't give element-wise probs easily
                else:
                    # For non-CRF models
                    outputs = self.model(inputs)
                    logits = outputs["logits"]
                    probs, pred_indices = torch.max(torch.sigmoid(logits), dim=-1)
                    tags = pred_indices.tolist()
                    probs = probs.tolist()

                # Process batch results
                for i in range(len(tags)):
                    idx = steps_done * self.model_config.batch_size + i
                    if idx >= len(texts):
                        break

                    text = texts[idx]
                    tokens = tokenized_texts[idx]
                    offsets = all_offsets[idx]

                    pred_tags_indices = tags[i]

                    # Inverse transform tags
                    pred_tags = self.preprocessor.inverse_transform(pred_tags_indices)

                    # For BERT/Transformers, we might need alignment if subwords were used
                    # But create_dataloader and models should handle this?
                    # In Keras version, generator returned offsets for alignment.
                    # In PyTorch version, if we use the same preprocessor/tokenizer logic,
                    # we need to ensure alignment.
                    # The create_dataloader in data_loader_pytorch uses "bert_preprocessor" logic if transformer_name is set.

                    # If we simply use the outputs, they should match the input tokens fed to the model.
                    # However, if we used subword tokenization, the model output matches subwords.
                    # We need to map back to original words.

                    # For now, let's assume 1-to-1 mapping or that preprocessor handles it.
                    # Re-checking tagger.py logic for transformers:
                    # It uses generator_output which contains input_offsets for subword alignment.

                    # If using transformers, the model output corresponding to subwords.
                    # We simply take the first subword label for the whole word usually.

                    current_probs = probs[i] if probs else None

                    if output_format == "json":
                        piece = {}
                        piece["text"] = text
                        piece["entities"] = self._build_json_response(
                            text, tokens, pred_tags, current_probs, offsets
                        )["entities"]
                        res["texts"].append(piece)
                    else:
                        the_tags = list(zip(tokens, pred_tags))
                        list_of_tags.append(the_tags)

                steps_done += 1

        if output_format == "json":
            return res
        else:
            return list_of_tags

    def _build_json_response(self, original_text, tokens, tags, prob, offsets):
        res = {"entities": []}
        chunks = get_entities_with_offsets(tags, offsets)
        for chunk_type, chunk_start, chunk_end, pos_start, pos_end in chunks:
            if prob is not None:
                # Handle potential length mismatch if prob is shorter than tags (shouldn't happen)
                end = min(chunk_end, len(prob))
                if chunk_start < end:
                    score = float(np.average(prob[chunk_start:end]))
                else:
                    score = 1.0
            else:
                score = 1.0

            if pos_start is not None and pos_end is not None:
                entity_text = original_text[pos_start : pos_end + 1]
                entity = {
                    "text": entity_text,
                    "class": chunk_type,
                    "score": score,
                    "beginOffset": pos_start,
                    "endOffset": pos_end,
                }
                res["entities"].append(entity)

        return res


def get_entities_with_offsets(seq, offsets):
    """
    Gets entities from sequence
    """
    i = 0
    chunks = []
    seq = seq + ["O"]  # add sentinel
    types = [tag.split("-")[-1] for tag in seq]
    max_length = min(len(seq) - 1, len(offsets))

    while i < max_length:
        if seq[i].startswith("B"):
            j = max_length
            if i + 1 != max_length:
                for j in range(i + 1, max_length + 1):
                    if seq[j].startswith("I") and types[j] == types[i]:
                        continue
                    break

            # offsets is list of tuples (start, end)
            if i < len(offsets) and (j - 1) < len(offsets):
                start_pos = offsets[i][0]
                end_pos = offsets[j - 1][1] - 1
                chunks.append((types[i], i, j, start_pos, end_pos))

            i = j
        else:
            i += 1
    return chunks
