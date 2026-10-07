import datetime
import logging

import numpy as np
import torch

from delft.sequenceLabelling.data_loader import create_dataloader
from delft.sequenceLabelling.windows import join_overlapping_windows
from delft.utilities.Tokenizer import tokenizeAndFilter

LOGGER = logging.getLogger(__name__)


def word_positions(word_starts):
    """
    The sub-token position of every word, in the order of the words, given how many
    words start on each sub-token: the words a sub-token spans all take its position.
    """
    return [position for position, nb_started in enumerate(word_starts) for _ in range(int(nb_started))]


class Tagger(object):
    def __init__(self, model, model_config, embeddings=None, preprocessor=None, device=None, nb_workers=0):
        """
        ``device`` is resolved by the caller, once — ``Sequence`` does it in its
        constructor and hands the result down. Resolving it here instead would
        repeat the work (and the device announcement) on every tag() call.
        When it is omitted we take the device the model already sits on rather
        than re-picking one, so a Tagger can never disagree with its model.

        ``nb_workers`` is the number of DataLoader worker *processes* used for
        inference. It defaults to 0 (in-process loading), which is the only
        safe value when DeLFT is embedded in a host process such as GROBID via
        JEP: a DataLoader is built on every tag() call, so any worker > 0 pays
        a process spawn per call and forks the host's interpreter.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings
        if device is None:
            device = next(model.parameters(), torch.empty(0)).device
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.nb_workers = max(0, nb_workers) if nb_workers is not None else 0

    def tag(self, texts, output_format, features=None, window_stride=None):
        """
        Label ``texts``. A sequence longer than ``max_sequence_length`` is cut into
        windows of that length, one every ``window_stride`` (in tokens, or in sub-tokens
        with a transformer), and labelled whole. Where windows overlap, a token takes
        the label of the window it is further from the edge of (see
        ``delft.sequenceLabelling.windows.join_overlapping_windows``).

        ``window_stride`` defaults to the one the model was trained with, saved in its
        configuration. A model trained without has none, and a sequence longer than it
        takes is truncated then, its last tokens left without a label.
        """
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

        if window_stride is None:
            window_stride = self._saved_window_stride()

        # Create dataloader
        # Note: y is None for inference
        dataloader = create_dataloader(
            tokenized_texts,
            None,
            preprocessor=self.preprocessor,
            embeddings=self.embeddings,
            batch_size=self.model_config.batch_size,
            features=features,
            num_workers=self.nb_workers,
            shuffle=False,
            model_config=self.model_config,
            role="tag",
            window_stride=window_stride,
        )

        # the label indices, and scores, of every example of the loader, in its order: a
        # sequence, or a window of one
        example_tags = []
        example_probs = []
        self.model.eval()

        # inference_mode rather than no_grad: it additionally skips view and
        # version-counter tracking, which is pure overhead here since nothing
        # leaves this loop but lists of tag indices.
        with torch.inference_mode():
            for batch in dataloader:
                inputs, _ = batch  # dataloader yields (inputs, labels), labels are None or dummies

                # Move inputs to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

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
                    # these models are trained with a softmax over the labels: the score of a
                    # label is its probability among them, which the sigmoid of its logit is not
                    probs, pred_indices = torch.max(torch.softmax(logits, dim=-1), dim=-1)
                    tags = pred_indices.tolist()
                    probs = probs.tolist()

                # transformers predict one label per sub-token: the data loader counts
                # the words starting on each sub-token, where the label of the word is
                word_starts = inputs["word_start_mask"].tolist() if "word_start_mask" in inputs else None

                for i in range(len(tags)):
                    pred_tags_indices = list(tags[i])
                    current_probs = probs[i] if probs else None

                    if word_starts is not None:
                        positions = word_positions(word_starts[i][: len(pred_tags_indices)])
                        pred_tags_indices = [pred_tags_indices[p] for p in positions]
                        if current_probs is not None:
                            current_probs = [current_probs[p] for p in positions]

                    example_tags.append(pred_tags_indices)
                    example_probs.append(current_probs)

        window_bounds = getattr(getattr(dataloader, "dataset", None), "window_bounds", None)
        if window_bounds is not None:
            # the windows of a sequence are labelled apart: put them back together, a
            # word of a transformer being a position as a token is
            example_tags = join_overlapping_windows(example_tags, window_bounds)
            if all(current_probs is not None for current_probs in example_probs):
                example_probs = join_overlapping_windows(example_probs, window_bounds)
            else:
                example_probs = [None] * len(example_tags)

        truncated = []  # (index, number of tokens, number of labels) of the sequences cut
        for idx, (text, pred_tags_indices, current_probs) in enumerate(zip(texts, example_tags, example_probs)):
            tokens = tokenized_texts[idx]
            offsets = all_offsets[idx]
            pred_tags = self.preprocessor.inverse_transform(pred_tags_indices)

            # without windows, the data loader cuts a sequence at max_sequence_length:
            # the tokens after the cut get no label and are left out of the result
            if len(pred_tags) < len(tokens):
                truncated.append((idx, len(tokens), len(pred_tags)))

            if output_format == "json":
                piece = {}
                piece["text"] = text
                piece["entities"] = self._build_json_response(text, tokens, pred_tags, current_probs, offsets)[
                    "entities"
                ]
                res["texts"].append(piece)
            else:
                the_tags = list(zip(tokens, pred_tags))
                list_of_tags.append(the_tags)

        if truncated:
            idx, nb_tokens, nb_labels = truncated[0]
            LOGGER.warning(
                "%d of %d sequences are longer than the model takes (max_sequence_length=%d%s) and were truncated: "
                "their last tokens are not labelled, e.g. sequence %d has %d tokens and %d labels. "
                "Give a window_stride to label them whole, or cut long sequences before sending them.",
                len(truncated),
                len(texts),
                self.model_config.max_sequence_length,
                " sub-tokens" if self.model_config.transformer_name else "",
                idx,
                nb_tokens,
                nb_labels,
            )

        if output_format == "json":
            return res
        else:
            return list_of_tags

    def _saved_window_stride(self):
        """
        The stride the model was trained with, no longer than the windows: the caller
        may have made ``max_sequence_length`` shorter than the model was trained with,
        which a stride it did not ask for should not make an error.
        """
        window_stride = getattr(self.model_config, "window_stride", None)
        max_sequence_length = self.model_config.max_sequence_length
        if window_stride and max_sequence_length:
            return min(window_stride, max_sequence_length)
        return window_stride

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
