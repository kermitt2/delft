# some convenient methods for all models
import regex as re
import numpy as np
import pandas as pd
import sys
import os.path

from keras.preprocessing import text
from keras import backend as K

#from nltk.tokenize import wordpunct_tokenize
#from nltk.stem.snowball import EnglishStemmer

from tqdm import tqdm 
from gensim.models import FastText
from gensim.models import KeyedVectors
import langdetect
from textblob import TextBlob
from textblob.translate import NotTranslated
from xml.sax.saxutils import escape


def dot_product(x, kernel):
    """
    Wrapper for dot product operation used inthe attention layers, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


# read list of words (one per line), e.g. stopwords, badwords
def read_words(words_file):
    return [line.replace('\n','').lower() for line in open(words_file, 'r')]


# preprocessing used for twitter-trained glove embeddings
def glove_preprocess(text):
    """
    adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    """
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?:* ", "<URL>", text)
    text = re.sub("www.* ", "<URL>", text)
    text = re.sub("\[\[User(.*)\|", '<USER>', text)
    text = re.sub("<3", '<HEART>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
    text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
    text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
    text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
    text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
    text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
    text = re.sub("/", " / ", text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub("([!]){2,}", "! <REPEAT>", text)
    text = re.sub("([?]){2,}", "? <REPEAT>", text)
    text = re.sub("([.]){2,}", ". <REPEAT>", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <ELONG>", text)

    return text


# stemming 
"""
stemmer = EnglishStemmer()

def stem_word(text):
    return stemmer.stem(text)

def lemmatize_word(text):
    return lemmatizer.lemmatize(text)

def apply_stopword(text):
    return '' if text.lower() in stopwords else text

def reduce_text(conversion, text):
    return " ".join(map(conversion, wordpunct_tokenize(text.lower())))

def reduce_texts(conversion, texts):
    return [reduce_text(conversion, str(text))
            for text in tqdm(texts)]
"""

url_regex = re.compile(r"https?:\/\/[a-zA-Z0-9_\-\.]+(?:com|org|fr|de|uk|se|net|edu|gov|int|mil|biz|info|br|ca|cn|in|jp|ru|au|us|ch|it|nl|no|es|pl|ir|cz|kr|co|gr|za|tw|hu|vn|be|mx|at|tr|dk|me|ar|fi|nz)\/?\b")

# language detection with langdetect package
def detect_lang(x):
    try:
        language = langdetect.detect(x)
    except:
        language = 'unk'
    return language

# language detection with textblob package
def detect_lang_textBlob(x):
    #try:
    theBlob = TextBlob(x)
    language = theBlob.detect_language()
    #except:
    #    language = 'unk'
    return language

def translate(comment):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)


# generate the list of out of vocabulary words present in the Toxic dataset 
# with respect to 3 embeddings: fastText, Gloves and word2vec
def generateOOVEmbeddings():
    # read the (DL cleaned) dataset and build the vocabulary
    print('loading dataframes...')
    train_df = pd.read_csv('../data/training/train2.cleaned.dl.csv')
    test_df = pd.read_csv('../data/eval/test2.cleaned.dl.csv')

    # ps: forget memory and runtime, it's python here :D
    list_sentences_train = train_df["comment_text"].values
    list_sentences_test = test_df["comment_text"].values
    list_sentences_all = np.concatenate([list_sentences_train, list_sentences_test])

    tokenizer = text.Tokenizer(num_words=400000)
    tokenizer.fit_on_texts(list(list_sentences_all))
    print('word_index size:', len(tokenizer.word_index), 'words')
    word_index = tokenizer.word_index

    # load fastText - only the words
    print('loading fastText embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('fastText embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('fastText embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-fastText.txt", "w") as oovFile:
        for w in oov:
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()

    # load gloves - only the words
    print('loading gloves embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/glove.840B.300d.txt')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        voc.add(word)
    f.close()
    print('gloves embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('gloves embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-gloves.txt", "w") as oovFile:
        for w in oov:
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()

    # load word2vec - only the words
    print('loading word2vec embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/GoogleNews-vectors-negative300.vec')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('word2vec embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)
    
    print('word2vec embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-w2v.txt", "w") as oovFile:
        for w in oov:    
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()
    
     # load numberbatch - only the words
    print('loading numberbatch embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/numberbatch-en-17.06.txt')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('numberbatch embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)
    
    print('numberbatch embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-numberbatch.txt", "w") as oovFile:
        for w in oov:    
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()


# tests...
#if __name__ == "__main__":
    # get the argument

