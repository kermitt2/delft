# Manage pre-trained embeddings 

from keras.preprocessing import text, sequence
import numpy as np
import sys
import os.path
import json


class Embeddings(object):

    def __init__(self, name, path='./embedding-registry.json'):
        self.name = name
        self.embed_size = 0
        self.model = {}
        self.registry = self._load_embedding_registry(path)
        self.make_embeddings_simple(name)
        
    def __getattr__(self, name):
        return getattr(self.model, name)

    def _load_embedding_registry(self, path='./embedding-registry.json'):
        """
        Load the description of available embeddings. Each description provides a name, 
        a file path (used only if necessary) and a embeddings type (to take into account
        small variation of format)
        """
        registry_json = open(path).read()
        return json.loads(registry_json)

    def make_embeddings_simple(self, name="fasttext-crawl", hasHeader=True):
        nbWords = 0
        print('loading embeddings...')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            print("path:", embeddings_path)
            if embeddings_type == "glove":
                hasHeader = False
            with open(embeddings_path) as f:
                for line in f:
                    line = line.split(' ')
                    if begin:
                        if hasHeader:
                            # first line gives the nb of words and the embedding size
                            nbWords = int(line[0])
                            self.embed_size = int(line[1].replace("\n", ""))
                            begin = False
                            continue
                        else:
                            begin = False
                    word = line[0]
                    if embeddings_type == 'glove':
                        vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                    else:
                        vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
                    if self.embed_size == 0:
                        self.embed_size = len(vector)
                    self.model[word] = vector
            if nbWords == 0:
                nbWords = len(self.model)
            print('embeddings loaded for', nbWords, "words and", self.embed_size, "dimensions")

    def _get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        return None

    def get_word_vector(self, word):
        if word in self.model:
            return self.model[word]
        else:
            # for unknown word, we use a vector filled with 0.0
            return np.zeros((self.embed_size,), dtype=np.float32)
            # alternatively, initialize with random negative values
            #return np.random.uniform(low=-0.5, high=0.0, size=(embeddings.shape[1],))
            # alternatively use fasttext OOV ngram possibilities (if ngram available)

# based on https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# not used
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


# extend the given embeddings with oov embeddings produced by mimick, see
# https://github.com/yuvalpinter/Mimick
# not used
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
