import itertools
import regex as re
import numpy as np

from unidecode import unidecode
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

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

def create_single_input_bert(text, maxlen=512, transformer_tokenizer=None):
    '''
    Note: use batch method preferably for better performance
    '''

    # TBD: exception if tokenizer is not valid/None
    encoded_tokens = transformer_tokenizer.encode_plus(text, truncation=True, add_special_tokens=True, 
                                                max_length=maxlen, padding='max_length')
    # note: [CLS] and [SEP] are added by the tokenizer

    ids = encoded_tokens["input_ids"]
    masks = encoded_tokens["token_type_ids"]
    segments = encoded_tokens["attention_mask"]

    return ids, masks, segments

def create_batch_input_bert(texts, maxlen=512, transformer_tokenizer=None):
    # TBD: exception if tokenizer is not valid/None

    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    encoded_tokens = transformer_tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True, 
                                                max_length=maxlen, padding='max_length')

    # note: special tokens like [CLS] and [SEP] are added by the tokenizer

    ids = encoded_tokens["input_ids"]
    masks = encoded_tokens["token_type_ids"]
    segments = encoded_tokens["attention_mask"]

    return ids, masks, segments
