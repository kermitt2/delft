from collections import defaultdict
import numpy as np
import datetime
from sequenceLabelling.metrics import get_entities
from sequenceLabelling.tokenizer import tokenizeAndFilter

class Tagger(object):

    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, tokens):
        length = np.array([len(tokens)])
        X = self.preprocessor.transform([tokens])
        pred = self.model.predict(X, length)

        return pred

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response_old(self, tokens, tags, prob):
        res = {
            "software": "DeLFT",
            "date": datetime.datetime.now().isoformat(),
            "model": self.model.model_config.model_name,
            "tokens": tokens,
            "entities": []
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            # TODO: get the original string rather than regenerating it from tokens
            entity = {
                "text": ' '.join(tokens[chunk_start: chunk_end]),
                "class": chunk_type,
                "score": float(np.average(prob[chunk_start: chunk_end])),
                "beginOffset": chunk_start,
                "endOffset": chunk_end
            }
            res["entities"].append(entity)

        return res

    def _build_json_response(self, tokens, tags, prob):
        res = {
            "entities": []
        }
        chunks = get_entities(tags)
        for chunk_type, chunk_start, chunk_end in chunks:
            # TODO: get the original string rather than regenerating it from tokens
            entity = {
                "text": ' '.join(tokens[chunk_start: chunk_end]),
                "class": chunk_type,
                "score": float(np.average(prob[chunk_start: chunk_end])),
                "beginOffset": chunk_start,
                "endOffset": chunk_end
            }
            res["entities"].append(entity)

        return res

    def analyze_old(self, tokens):
        assert isinstance(tokens, list)

        pred = self.predict(tokens)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(tokens, tags, prob)

        return res

    def analyze(self, texts, output_format):
        assert isinstance(texts, list)

        if output_format is 'json':
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model.config.model_name,
                "texts": []
            }
        else:
           list_of_tags = []

        for text in texts:
            tokens = tokenizeAndFilter(text)

            pred = self.predict(tokens)
            tags = self._get_tags(pred)
            prob = self._get_prob(pred)
            #entities = self._build_response(tokens, tags, prob)
            
            if output_format is 'json':
                piece = {}
                piece["text"] = text
                piece["entities"] = self._build_json_response(tokens, tags, prob)["entities"]
                res["texts"].append(piece)
            else:
                the_tags = list(zip(tokens, tags))
                list_of_tags.append(the_tags)

        if output_format is 'json':
            return res
        else:
            return list_of_tags

    def tag_old(self, tokens):
        """Tags a sentence named entities.

        Args:
            sent: a sentence

        Return:
            labels_pred: list of (token, tag) for a sentence

        Example:
            >>> sent = 'President Obama is speaking at the White House.'
            >>> print(self.tag(sent))
            [('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
             ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
             ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
        """
        assert isinstance(tokens, list)

        pred = self.predict(tokens)
        tags = self._get_tags(pred)
        #tags = [t.split('-')[-1] for t in tags]  # remove prefix: e.g. B-Person -> Person

        return list(zip(tokens, tags))

    def get_entities(self, tokens):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        assert isinstance(tokens, list)

        pred = self.predict(tokens)
        entities = self._get_chunks(tokens, pred)

        return entities

    def _get_chunks(self, tokens, tags):
        """
        Args:
            tokens: sequence of tokens
            tags: sequence of labels

        Returns:
            dict of entities for a sequence

        Example:
            tokens = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
            tags = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        chunks = get_entities(tags)
        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(tokens[chunk_start: chunk_end]))  # todo delimiter changeable

        return res
