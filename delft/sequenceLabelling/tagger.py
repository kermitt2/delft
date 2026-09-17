import datetime

import numpy as np
import torch

from delft.sequenceLabelling.data_loader import create_dataloader
from delft.utilities.Tokenizer import tokenizeAndFilter

# Number of tokens two consecutive windows of a long sequence share. A model sees
# little context at the edge of a window, so each shared token takes its label from
# the window where it is further from that edge.
WINDOW_OVERLAP = 50


class Tagger(object):
    def __init__(
        self,
        model,
        model_config,
        embeddings=None,
        preprocessor=None,
        device=None,
        nb_workers=0,
        window_overlap=WINDOW_OVERLAP,
    ):
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

        ``window_overlap`` is the number of tokens shared by two consecutive
        windows when a sequence is too long to be labelled in one pass, see
        ``_predict``.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings
        if device is None:
            device = next(model.parameters(), torch.empty(0)).device
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.nb_workers = max(0, nb_workers) if nb_workers is not None else 0
        self.window_overlap = max(0, window_overlap)

    def tag(self, texts, output_format, features=None):
        """
        Label ``texts``, either strings (tokenized here) or lists of tokens, with one
        label per token whatever the length of a text.
        """
        to_tokeniz = len(texts) > 0 and isinstance(texts[0], str)

        tokenized_texts = []
        all_offsets = []
        for text in texts:
            if to_tokeniz:
                tokens, offsets = tokenizeAndFilter(text)
                tokenized_texts.append(tokens)
                all_offsets.append(offsets)
            else:
                tokenized_texts.append(text)
                all_offsets.append([])  # No offsets if already tokenized

        self.model.eval()
        all_tags, all_probs = self._predict(tokenized_texts, features)

        if output_format == "json":
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model_config.model_name,
                "texts": [],
            }
            for text, tokens, tags, probs, offsets in zip(texts, tokenized_texts, all_tags, all_probs, all_offsets):
                piece = {}
                piece["text"] = text
                piece["entities"] = self._build_json_response(text, tokens, tags, probs, offsets)["entities"]
                res["texts"].append(piece)
            return res

        return [list(zip(tokens, tags)) for tokens, tags in zip(tokenized_texts, all_tags)]

    def _predict(self, tokenized_texts, features=None):
        """
        Return the labels, and the scores when the model gives some, of every token
        of every text.

        A model labels ``max_sequence_length`` tokens at most, and a transformer that
        many sub-tokens, which is fewer words: the data loader cuts what is beyond. A
        text that was cut is labelled again from ``window_overlap`` tokens before the
        point its labels stop, until all its tokens have one. Texts that fit, the
        usual case, take the single pass they always took.
        """
        tags = [[] for _ in tokenized_texts]
        probs = [[] for _ in tokenized_texts]
        with_probs = True
        starts = [0] * len(tokenized_texts)
        pending = list(range(len(tokenized_texts)))

        while pending:
            windows = [tokenized_texts[d][starts[d] :] for d in pending]
            window_features = None if features is None else [features[d][starts[d] :] for d in pending]
            predictions = self._predict_windows(windows, window_features)
            if len(predictions) != len(pending):
                raise RuntimeError(f"{len(pending)} sequences to label, {len(predictions)} labelled")

            still_pending = []
            for d, (window_tags, window_probs) in zip(pending, predictions):
                nb_tokens = len(tokenized_texts[d])
                if len(window_tags) == 0 and starts[d] < nb_tokens:
                    raise RuntimeError(f"no label predicted from token {starts[d]} of sequence {d}")

                # of the tokens this window shares with the previous one, the first
                # half keeps the label it has and the second half takes the new one
                keep = starts[d] + (len(tags[d]) - starts[d]) // 2
                tags[d] = tags[d][:keep] + window_tags[keep - starts[d] :]
                if window_probs is None:
                    with_probs = False
                else:
                    probs[d] = probs[d][:keep] + window_probs[keep - starts[d] :]

                if len(tags[d]) < nb_tokens:
                    # sharing half a window at most, the next window always starts
                    # further than this one
                    starts[d] = len(tags[d]) - min(self.window_overlap, len(window_tags) // 2)
                    still_pending.append(d)
            pending = still_pending

        return tags, probs if with_probs else [None] * len(tokenized_texts)

    def _predict_windows(self, windows, window_features=None):
        """
        One pass of the model: return, for each window and in the same order, the
        labels and the scores (None for a CRF) of the tokens the model could take.
        """
        dataloader = create_dataloader(
            windows,
            None,
            preprocessor=self.preprocessor,
            embeddings=self.embeddings,
            batch_size=self.model_config.batch_size,
            features=window_features,
            num_workers=self.nb_workers,
            shuffle=False,
            model_config=self.model_config,
            role="tag",
        )

        predictions = []

        # inference_mode rather than no_grad: it additionally skips view and
        # version-counter tracking, which is pure overhead here since nothing
        # leaves this loop but lists of tag indices.
        with torch.inference_mode():
            for inputs, _ in dataloader:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                if hasattr(self.model, "decode"):
                    # CRF models: hard decoding, no score per token
                    rows = self.model.decode(inputs)
                    rows_probs = None
                else:
                    logits = self.model(inputs)["logits"]
                    rows_probs, rows = torch.max(torch.sigmoid(logits), dim=-1)
                    rows_probs = rows_probs.tolist()

                word_starts = inputs["word_start_mask"].tolist() if "word_start_mask" in inputs else None
                lengths = inputs["length"].reshape(-1).tolist() if "length" in inputs else None

                for i, row in enumerate(rows):
                    if word_starts is not None:
                        # transformers predict one label per sub-token, the label of
                        # a word being the one of its first sub-token
                        positions = [p for p, is_start in enumerate(word_starts[i][: len(row)]) if is_start]
                    else:
                        # a row is padded to the longest sequence of its batch, and a
                        # single token sequence is fed as two
                        nb_tokens = min(len(row), len(windows[len(predictions)]))
                        if lengths is not None:
                            nb_tokens = min(nb_tokens, lengths[i])
                        positions = range(nb_tokens)

                    window_tags = self.preprocessor.inverse_transform([row[p] for p in positions])
                    window_probs = None if rows_probs is None else [rows_probs[i][p] for p in positions]
                    predictions.append((window_tags, window_probs))

        return predictions

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
