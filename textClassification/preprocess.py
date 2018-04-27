import itertools
import re
import numpy as np
from sklearn.externals import joblib
from keras.preprocessing import text, sequence

class TextPreprocessor(object):

    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 vocab_init=None,
                 max_words=300000):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.vocab_word = None
        self.vocab_char = None
        self.vocab_tag  = None
        self.vocab_init = vocab_init or {}

        self.tokenizer = text.Tokenizer(num_words=max_words)

    def clean_text(x):
        x_ascii = unidecode(x)
        x_clean = special_character_removal.sub('',x_ascii)
        return x_clean

    def lower(self, word):
        return word.lower() if self.lowercase else word

    def normalize_num(self, word):
        if self.num_norm:
            return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
        else:
            return word

    def word_index(self):
        return self.tokenizer.word_index

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p
