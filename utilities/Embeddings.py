# Build custom embeddings on textual content of the data sets

from keras.preprocessing import text, sequence
import numpy as np
import sys
import os.path

# based on https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
def make_embeddings(embeddingspath, max_features, embed_size, word_index, skipHeader):
    embeddings_index = {}
    f = open(embeddingspath)
    begin = False
    if skipHeader:
        begin = True
    for line in f:
        if begin:
            # skip header
            begin = False
            continue
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def make_embeddings_simple(embeddingspath):
    model = {}
    embed_size = 0
    nbWords = 0
    print('loading embeddings...')
    with open(embeddingspath) as f:
        begin = True
        for line in f:
            line = line.split(' ')
            if begin:
                # first line gives the nb of words and the embedding size
                nbWords = line[0]
                embed_size = line[1].replace("\n", "")
                begin = False
            else:
                word = line[0]
                vector = np.array([float(val) for val in line[1:len(line)-1]])
                # note: above is working fine with FastText, but with Glove the -1 would need to be removed 
                model[word] = vector
    print('embeddings loaded for', nbWords, "words and", embed_size, "dimensions")
    return embed_size, model

def make_embeddings_fastText(embeddingspath, max_features, embed_size, word_index):
    fastTextModel = FastText.load_fasttext_format(embeddingspath)

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        # normally the following will give a vector for oov via ngram stuffs in fastText
        embedding_vector = fastTextModel[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# extend the given embeddings with oov embeddings produced by mimick, see
# https://github.com/yuvalpinter/Mimick
def make_embeddings_with_oov(embeddingspath, oov_embeddings_path, max_features, embed_size, word_index, skipHeader):
    embeddings_index = {}
    f = open(embeddingspath)
    begin = False
    if skipHeader:
        begin = True
    for line in f:
        if begin:
            # skip header
            begin = False
            continue
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    oov_embeddings_index = {}
    f = open(oov_embeddings_path)
    begin = True
    for line in f:
        if begin:
            # always skip header with the mimick embeddings
            begin = False
            continue
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        oov_embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = oov_embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
    return embedding_matrix
