import itertools
import regex as re
import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)

from unidecode import unidecode
from utilities.Tokenizer import tokenizeAndFilterSimple

special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)

def to_vector_single(text, embeddings, maxlen=300):
    """
    Given a string, tokenize it, then convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    tokens = tokenizeAndFilterSimple(clean_text(text))
    window = tokens[-maxlen:]
    
    # TBD: use better initializers (uniform, etc.) 
    x = np.zeros((maxlen, embeddings.embed_size), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        x[i,:] = embeddings.get_word_vector(word).astype('float32')

    return x

def clean_text(text):
    x_ascii = unidecode(text)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean

def lower(word):
    return word.lower() 

def normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)