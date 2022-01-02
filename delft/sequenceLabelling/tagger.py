from collections import defaultdict
import numpy as np
import datetime
from delft.sequenceLabelling.data_generator import DataGeneratorTransformers
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

class Tagger(object):

    def __init__(self, 
                model, 
                model_config, 
                embeddings=None, 
                preprocessor=None, 
                bert_preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.bert_preprocessor = bert_preprocessor
        self.model_config = model_config
        self.embeddings = embeddings

    def tag(self, texts, output_format, features=None):
        assert isinstance(texts, list)

        # if the model uses a transformer layer, we cannot use a batch generator, so
        # we rely on a tag function customized for transformers usage 
        '''
        if self.model_config.transformer != None:
            return self.tag_without_generator(texts, output_format, features=features)
        '''

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
        
        generator = self.model.get_generator()
        predict_generator = generator(texts, None, 
            batch_size=self.model_config.batch_size, 
            preprocessor=self.preprocessor, 
            bert_preprocessor=self.bert_preprocessor,
            char_embed_size=self.model_config.char_embedding_size,
            max_sequence_length=self.model_config.max_sequence_length,
            embeddings=self.embeddings, tokenize=to_tokeniz, shuffle=False, 
            features=features, output_input_tokens=True)

        steps_done = 0
        steps = len(predict_generator)
        for generator_output in predict_generator:
            if steps_done == steps:
                break

            if isinstance(predict_generator, DataGeneratorTransformers):
                # the model uses transformer embeddings, so we need the input tokens to realign correctly the 
                # labels and the inpit label texts 

                # we need to remove one vector of the data corresponding to the marked tokens, this vector is not 
                # expected by the model, but we need it to restore correctly the labels (which are produced
                # according to the sub-segmentation of wordpiece, not the expected segmentation)
                data = generator_output[0]

                input_tokens = data[-1]
                data = data[:-1]

                y_pred_batch = self.model.predict_on_batch(data)
                y_pred_batch = np.argmax(y_pred_batch, -1)

                # results have been produced by a model using a transformer layer, so a few things to do
                # the labels are sparse, so integers and not one hot encoded
                # we need to restore back the labels for wordpiece to the labels for normal tokens
                # for this we can use the marked tokens provided by the generator 
                new_y_pred_batch = []
                for y_pred_text, tokens_text in zip(y_pred_batch, input_tokens):
                    new_y_pred_text = []
                    # this is the result per sequence, realign labels:
                    for q in range(len(y_pred_text)):
                        if tokens_text[q] == '[SEP]':
                            break
                        if tokens_text[q] in ['[PAD]', '[CLS]']:
                            continue
                        if tokens_text[q].startswith("##"): 
                            continue
                        new_y_pred_text.append(y_pred_text[q]) 
                    new_y_pred_batch.append(new_y_pred_text)
                preds = new_y_pred_batch

                #preds = [self.preprocessor.inverse_transform(y) for y in y_pred_batch]
            else:
                # no weirdness changes on the input 
                preds = self.model.predict_on_batch(generator_output[0])
                #preds = [self.preprocessor.inverse_transform(y) for y in preds]

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

                if isinstance(predict_generator, DataGeneratorTransformers):
                    tags = self._get_tags_sparse(pred)
                    prob = self._get_prob_sparse(pred)
                else:
                    tags = self._get_tags(pred)
                    prob = self._get_prob(pred)

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


    def tag_without_generator(self, texts, output_format='json', features=None):

        if output_format == 'json':
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model_config.model_name,
                "texts": []
            }
        else:
           list_of_tags = []

        if texts is None or len(texts) == 0:
            return res

        if not self.preprocessor.return_features:
            features = None

        to_tokeniz = False
        if (len(texts)>0 and isinstance(texts[0], str)):
            # we need to tokenize these texts
            to_tokeniz = True
            texts_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in texts
            ]
        else:
            texts_tokenized = texts

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                #bound = min(n, )
                yield l[i:i + n]
        
        batch_idx = 0
        # segment in batches corresponding to self.predict_batch_size
        for text_batch in list(chunks(texts_tokenized, self.model_config.batch_size)):
            features_batch = None
            if features is not None:
                upper_bound = min(len(texts_tokenized), (batch_idx+1)*self.model_config.batch_size)
                features_batch = features[batch_idx*self.model_config.batch_size:upper_bound]

            num_current_batch = len(text_batch)

            batches = self.preprocessor.transform(text_batch)
            chars_batch = np.asarray(batches[0])

            if features is not None:
                features_batch = self.preprocessor.transform_features(features_batch)

            input_ids, input_masks, input_segments, input_chars, input_features, input_labels, input_tokens = self.bert_preprocessor.tokenize_and_align_features_and_labels(
                                                                        text_batch, 
                                                                        chars_batch,
                                                                        features_batch, 
                                                                        None,
                                                                        maxlen=self.model_config.max_sequence_length)

            batch_x = np.asarray(input_ids, dtype=np.int32)
            batch_x_masks = np.asarray(input_masks, dtype=np.int32)
            batch_c = np.asarray(input_chars, dtype=np.int32)

            if features is None:
                if self.preprocessor.return_chars:
                    results = self.model.predict_on_batch([batch_x, batch_c, batch_x_masks])
                else:
                    results = self.model.predict_on_batch([batch_x, batch_x_masks])
            else:
                batch_f = np.asarray(input_features, dtype=np.int32)
                if self.preprocessor.return_chars:
                    results = self.model.predict_on_batch([batch_x, batch_c, batch_f, batch_x_masks])
                else:
                    results = self.model.predict_on_batch([batch_x, batch_f, batch_x_masks])

            p = 0
            for i, prediction in enumerate(results):
                if p == num_current_batch:
                    break
                #predicted_labels = self._get_tags([prediction]) 
                text = texts[p+(batch_idx*self.model_config.batch_size)]

                y_pred_result = []
                for q in range(len(input_tokens)):
                    if input_tokens[i][q] == '[SEP]':
                        break
                    if input_tokens[i][q] in ['[PAD]', '[CLS]']:
                        continue
                    if input_tokens[i][q].startswith("##"): 
                        continue
                    y_pred_result.append(prediction[q]) 
                p += 1

                pred = y_pred_result
                
                if to_tokeniz:
                   tokens, offsets = tokenizeAndFilter(text)
                else:
                    # it is a list of string, so a string already tokenized
                    # note that in this case, offset are not present and json output is impossible
                    tokens = text
                    offsets = []

                tags = self._get_tags([pred])
                prob = self._get_prob([pred])

                if output_format == 'json':
                    piece = {}
                    piece["text"] = text
                    piece["entities"] = self._build_json_response(text, tokens, tags, prob, offsets)["entities"]
                    res["texts"].append(piece)
                else:
                    the_tags = list(zip(tokens, tags))
                    list_of_tags.append(the_tags)
            batch_idx += 1

        if output_format == 'json':
            return res
        else:
            return list_of_tags


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
