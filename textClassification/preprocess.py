import itertools
import regex as re
import numpy as np
from sklearn.externals import joblib
from keras.preprocessing import text, sequence
from unidecode import unidecode

class TextPreprocessor(object):

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 vocab_init=None,
                 max_words=300000):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.vocab_word = None
        self.vocab_init = vocab_init or {}

        self.tokenizer = text.Tokenizer(num_words=max_words)

        self.special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)

    def clean_text(self, text):
        x_ascii = unidecode(text)
        x_clean = self.special_character_removal.sub('',x_ascii)
        return x_clean

    def fit(self, texts):
        for i in range(0,len(texts)):
            texts[i] = self.clean_text(texts[i])

        #print(len(texts))
        #print(texts)
        
        self.tokenizer.fit_on_texts(texts)
        self.vocab_word = self.tokenizer.word_index

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

    def to_sequence(self, texts, maxlen=300):
        list_tokenized = self.tokenizer.texts_to_sequences(texts)
        X_t = sequence.pad_sequences(list_tokenized, maxlen=maxlen)
        return X_t

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p

def prepare_preprocessor(X, vocab_init):
    p = TextPreprocessor(vocab_init=vocab_init)
    p.fit(X)
    
    return p
