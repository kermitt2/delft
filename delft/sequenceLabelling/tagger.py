import datetime

import numpy as np

from delft.sequenceLabelling.data_generator import DataGeneratorTransformers
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.utilities.Tokenizer import tokenizeAndFilter


class Tagger(object):

    def __init__(self, 
                model, 
                model_config, 
                embeddings=None, 
                preprocessor: Preprocessor=None,
                transformer_preprocessor=None):

        self.model = model
        self.preprocessor = preprocessor
        self.transformer_preprocessor = transformer_preprocessor
        self.model_config = model_config
        self.embeddings = embeddings

    def tag(self, texts, output_format, features=None):

        if output_format == 'json':
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
        
        # dirty fix warning! in the particular case of using tf-addons CRF layer and having a 
        # single sequence in the input batch, a tensor shape error can happen in the CRF 
        # viterbi_decoding loop. So to prevent this, we add a dummy second sequence in the batch
        # that we will remove after prediction
        dummy_case = False
        if self.model_config.use_crf and not self.model_config.use_chain_crf and len(texts) == 1:
            if features == None:
                if to_tokeniz:
                    texts.append(texts[0])
                else:
                    texts.append(["dummy"])
            else:
                texts.append(texts[0])
                # add a dummy feature vector for the token dummy...
                features.append(features[0])
            dummy_case = True
        # end of dirty fix

        generator = self.model.get_generator()
        predict_generator = generator(texts, None, 
            batch_size=self.model_config.batch_size, 
            preprocessor=self.preprocessor, 
            bert_preprocessor=self.transformer_preprocessor,
            char_embed_size=self.model_config.char_embedding_size,
            max_sequence_length=self.model_config.max_sequence_length,
            embeddings=self.embeddings, tokenize=to_tokeniz, shuffle=False, 
            features=features, output_input_offsets=True, 
            use_chain_crf=self.model_config.use_chain_crf)

        steps_done = 0
        steps = len(predict_generator)
        for generator_output in predict_generator:
            if dummy_case and steps_done==1:
                break

            if steps_done == steps:
                break

            if isinstance(predict_generator, DataGeneratorTransformers):
                # the model uses transformer embeddings, so we need the input tokens to realign correctly the 
                # labels and the inpit label texts 

                # we need to remove one vector of the data corresponding to the marked tokens, this vector is not 
                # expected by the model, but we need it to restore correctly the labels (which are produced
                # according to the sub-segmentation of wordpiece, not the expected segmentation)
                data = generator_output[0]

                input_offsets = data[-1]
                data = data[:-1]

                y_pred_batch = self.model.predict_on_batch(data)
                #y_pred_batch = np.argmax(y_pred_batch, -1)

                # results have been produced by a model using a transformer layer, so a few things to do
                # the labels are sparse, so integers and not one hot encoded
                # we need to restore back the labels for wordpiece to the labels for normal tokens
                # for this we can use the marked tokens provided by the generator 
                new_y_pred_batch = []
                for y_pred_text, offsets_text in zip(y_pred_batch, input_offsets):
                    new_y_pred_text = []
                    # this is the result per sequence, realign labels:
                    for q in range(len(offsets_text)):
                        if offsets_text[q][0] == 0 and offsets_text[q][1] == 0:
                            # special token
                            continue
                        if offsets_text[q][0] != 0: 
                            # added sub-token
                            continue
                        new_y_pred_text.append(y_pred_text[q]) 
                    new_y_pred_batch.append(new_y_pred_text)
                preds = new_y_pred_batch
            else:
                # no weirdness changes on the input 
                preds = self.model.predict_on_batch(generator_output[0])

            for i in range(0, len(preds)):
                pred = [preds[i]]
                text = texts[i+(steps_done*self.model_config.batch_size)]

                if to_tokeniz:
                   tokens, offsets = tokenizeAndFilter(text)
                else:
                    # it is a list of string, so a string already tokenized
                    # note that in this case, offset are not present and json output is impossible
                    tokens = text
                    offsets = []

                if not self.model_config.use_crf or self.model_config.use_chain_crf:
                    tags = self._get_tags(pred)
                    prob = self._get_prob(pred)
                else:
                    tags = self._get_tags_sparse(pred)
                    prob = self._get_prob_sparse(pred)

                if output_format == 'json':
                    piece = {}
                    piece["text"] = text
                    piece["entities"] = self._build_json_response(text, tokens, tags, prob, offsets)["entities"]
                    res["texts"].append(piece)
                else:
                    the_tags = list(zip(tokens, tags))
                    list_of_tags.append(the_tags)
            steps_done += 1

        if output_format == 'json':
            return res
        else:
            return list_of_tags

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])
        return tags

    def _get_tags_sparse(self, pred):
        tags = self.preprocessor.inverse_transform(pred[0])
        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]
        return prob

    def _get_prob_sparse(self, pred):
        return [1.0] * len(pred[0])

    def _build_json_response(self, original_text, tokens, tags, prob, offsets):
        res = {
            "entities": []
        }
        chunks = get_entities_with_offsets(tags, offsets)
        for chunk_type, chunk_start, chunk_end, pos_start, pos_end in chunks:
            if prob is not None:
                score = float(np.average(prob[chunk_start:chunk_end]))
            else:
                score = 1.0

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
            if i+1 != max_length:
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
