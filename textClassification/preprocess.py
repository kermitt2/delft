import itertools
import regex as re
import numpy as np
#from sklearn.externals import joblib
#from keras.preprocessing import text, sequence
from unidecode import unidecode
from utilities.Tokenizer import tokenizeAndFilterSimple

special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)

def to_vector_single(text, embeddings, maxlen=300, embed_size=300):
    """
    Given a string, tokenize it, then convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    tokens = tokenizeAndFilterSimple(clean_text(text))
    window = tokens[-maxlen:]
    
    x = np.zeros((maxlen, embed_size), )

    for i, word in enumerate(window):
        x[i,:] = get_word_vector(word, embeddings, embed_size).astype('float32')

    return x

def get_word_vector(word, embeddings, embed_size):
    if word in embeddings:
        return embeddings[word]
    else:
        # for unknown word, we use a vector filled with 0.0
        return np.zeros((embed_size,), dtype=np.float32)
        # alternatively, initialize with random negative values
        #return np.random.uniform(low=-0.5, high=0.0, size=(embeddings.shape[1],))

def clean_text(text):
    x_ascii = unidecode(text)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean

def lower(word):
    return word.lower() 

def normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)