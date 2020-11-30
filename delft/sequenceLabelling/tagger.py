from collections import defaultdict
import numpy as np
import datetime
from delft.sequenceLabelling.data_generator import DataGenerator
from delft.utilities.Tokenizer import tokenizeAndFilter


class Tagger(object):

    def __init__(self, 
                model, 
                model_config, 
                embeddings=None, 
                preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.embeddings = embeddings

    def tag(self, texts, output_format, features=None):
        assert isinstance(texts, list)

        if output_format is 'json':
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model_config.model_name,
                "texts": []
            }
        else:
           list_of_tags = []

        to_tokeniz = False
        if (len(texts)>0 and isinstance(texts[0], str)):
            to_tokeniz = True

        if 'bert' in self.model_config.model_type.lower():
            preds = self.model.predict(texts, fold_id=-1)
            for i in range(0,len(preds)):
                pred = preds[i]
                text = texts[i]

                if (isinstance(text, str)):
                   tokens, offsets = tokenizeAndFilter(text)
                else:
                    # it is a list of string, so a string already tokenized
                    # note that in this case, offset are not present and json output is impossible
                    tokens = text
                    offsets = []

                tags = pred
                prob = None

                if output_format is 'json':
                    piece = {}
                    piece["text"] = text
                    piece["entities"] = self._build_json_response(text, tokens, tags, prob, offsets)["entities"]
                    res["texts"].append(piece)
                else:
                    the_tags = list(zip(tokens, tags))
                    list_of_tags.append(the_tags)

        else:
            predict_generator = DataGenerator(texts, None, 
                batch_size=self.model_config.batch_size, 
                preprocessor=self.preprocessor, 
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, tokenize=to_tokeniz, shuffle=False, features=features)

            nb_workers = 6
            multiprocessing = True
            # multiple workers will not work with ELMo due to GPU memory limit (with GTX 1080Ti 11GB)
            if self.embeddings.use_ELMo:
                # worker at 0 means the training will be executed in the main thread
                nb_workers = 0
                multiprocessing = False
                # dump token context independent data for train set, done once for the training

            steps_done = 0
            steps = len(predict_generator)
            for generator_output in predict_generator:
                if steps_done == steps:
                    break
                preds = self.model.predict_on_batch(generator_output[0])

                for i in range(0, len(preds)):
                    pred = [preds[i]]
                    text = texts[i+(steps_done*self.model_config.batch_size)]

                    #if (isinstance(text, str)):
                    if to_tokeniz:
                       tokens, offsets = tokenizeAndFilter(text)
                    else:
                        # it is a list of string, so a string already tokenized
                        # note that in this case, offset are not present and json output is impossible
                        tokens = text
                        offsets = []

                    tags = self._get_tags(pred)
                    prob = self._get_prob(pred)

                    if output_format is 'json':
                        piece = {}
                        piece["text"] = text
                        piece["entities"] = self._build_json_response(text, tokens, tags, prob, offsets)["entities"]
                        res["texts"].append(piece)
                    else:
                        the_tags = list(zip(tokens, tags))
                        list_of_tags.append(the_tags)
                steps_done += 1

        if output_format is 'json':
            return res
        else:
            return list_of_tags

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_json_response(self, original_text, tokens, tags, prob, offsets):
        res = {
            "entities": []
        }
        chunks = get_entities_with_offsets(tags, offsets)
        # LF: This could be combined with line 145, however currently the output list of spaces has one element missing
        # spaces = [offsets[offsetIndex-1][1] != offsets[offsetIndex][0] for offsetIndex in range(1, len(offsets))]

        for chunk_type, chunk_start, chunk_end, pos_start, pos_end in chunks:
            if prob is not None:
                score = float(np.average(prob[chunk_start:chunk_end]))
            else:
                score = 1.0

            # LF: reconstruct the text considering initial spaces - remove space a the end of the entity
            # text_from_tokens = ''.join([tokens[idx] + (' ' if spaces[idx] else '') for idx in range(chunk_start, chunk_end)])
            # if text_from_tokens.endswith(' '):
            #     text_from_tokens = text_from_tokens[0:-1]
            entity_text = original_text[pos_start: pos_end+1]

            entity = {
                "text": entity_text,
                "class": chunk_type,
                "score": score,
                "beginOffset": pos_start,
                "endOffset": pos_end
            }
            res["entities"].append(entity)

        return res


def get_entities_with_offsets(seq, offsets):
    """
    Gets entities from sequence

    Args:
        seq (list): sequence of labels.
        offsets (list of integer pair): sequence of offset position

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> offsets = [(0,10), (11, 15), (16, 29), (30, 41)]
        >>> print(get_entities(seq))
        [('PER', 0, 2, 0, 15), ('LOC', 3, 4, 30, 41)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    max_length = min(len(seq)-1, len(offsets))
    while i < max_length:
        if seq[i].startswith('B'):
            # if we are at the end of the offsets, we can stop immediately
            j = max_length
            if i+2 != max_length:
                for j in range(i+1, max_length+1):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
            start_pos = offsets[i][0]
            end_pos = offsets[j-1][1]-1
            chunks.append((types[i], i, j, start_pos, end_pos))
            i = j
        else:
            i += 1
    return chunks
