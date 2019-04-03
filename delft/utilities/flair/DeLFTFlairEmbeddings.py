import re
import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Dict

#import gensim
import numpy as np
import torch
#from deprecated import deprecated

#from pytorch_pretrained_bert.tokenization import BertTokenizer
#from pytorch_pretrained_bert.modeling import BertModel, PRETRAINED_MODEL_ARCHIVE_MAP

from flair.nn import LockedDropout, WordDropout
from flair.data import Dictionary, Token, Sentence
from flair.embeddings import TokenEmbeddings

class DeLFTFlairEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self, model, detach = True, use_cache = False):
        """
        Custom initialization of FLAIR emebeddings for DeLFT - avoid all the caching/downloading
        steps proper to FLAIR
        """
        super().__init__()
        
        if not Path(model).exists():
            raise ValueError('The given model ' + model + ' is not available or is not a valid path.')

        self.name = str(model)
        self.static_embeddings = detach

        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach

        self.is_forward_lm = self.lm.is_forward_lm

        # embed a dummy sentence to determine embedding_length
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(embedded_dummy[0].get_token(1).get_embedding())

        # set to eval mode
        self.eval()

    
    def train(self, mode=True):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state['cache'] = None
        return state

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):

        with torch.no_grad():

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences]

            longest_character_sequence_in_batch: int = len(max(text_sentences, key=len))

            # pad strings with whitespaces to longest sentence
            sentences_padded: List[str] = []
            append_padded_sentence = sentences_padded.append

            start_marker = '\n'

            end_marker = ' '
            extra_offset = len(start_marker)
            for sentence_text in text_sentences:
                pad_by = longest_character_sequence_in_batch - len(sentence_text)
                if self.is_forward_lm:
                    padded = '{}{}{}{}'.format(start_marker, sentence_text, end_marker, pad_by * ' ')
                    append_padded_sentence(padded)
                else:
                    padded = '{}{}{}{}'.format(start_marker, sentence_text[::-1], end_marker, pad_by * ' ')
                    append_padded_sentence(padded)

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()

                offset_forward = extra_offset
                offset_backward = len(sentence_text) + extra_offset

                for token in sentence.tokens:
                    #token: Token = token
                    offset_forward += len(token.text)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    # if self.tokenized_lm or token.whitespace_after:
                    offset_forward += 1
                    offset_backward -= 1

                    offset_backward -= len(token.text)

                    token.set_embedding(self.name, embedding)

        return sentences

    def __str__(self):
        return self.name
    