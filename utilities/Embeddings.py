# Manage pre-trained embeddings 

from keras.preprocessing import text, sequence
import numpy as np
import sys
import os.path
import json
import lmdb
import io
import pickle
from tqdm import tqdm
import mmap

# this is the default init size of a lmdb database for embeddings
# based on https://github.com/kermitt2/nerd/blob/master/src/main/java/com/scienceminer/nerd/kb/db/KBDatabase.java
# and https://github.com/kermitt2/nerd/blob/0.0.3/src/main/java/com/scienceminer/nerd/kb/db/KBDatabaseFactory.java#L368
map_size = 100 * 1024 * 1024 * 1024 

class Embeddings(object):

    def __init__(self, name, path='./embedding-registry.json'):
        self.name = name
        self.embed_size = 0
        self.vocab_size = 0
        #self.model = {}
        self.registry = self._load_embedding_registry(path)
        self.embedding_lmdb_path = None
        if self.registry is not None:
            self.embedding_lmdb_path = self.registry["embedding-lmdb-path"]
        self.env = None
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

    def make_embeddings_simple_old(self, name="fasttext-crawl", hasHeader=True):
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


    def make_embeddings_lmdb(self, name="fasttext-crawl", hasHeader=True):
        nbWords = 0
        print('\nCompiling embeddings... (this is done only one time per embeddings)')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            print("path:", embeddings_path)
            if embeddings_type == "glove":
                hasHeader = False
            txn = self.env.begin(write=True)
            batch_size = 1024
            i = 0
            nb_lines = 0
            with open(embeddings_path) as f:
                for line in f:
                    nb_lines += 1

            with open(embeddings_path) as f:
                #for line in f:
                for line in tqdm(f, total=nb_lines):
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

                                   
                    if len(word.encode(encoding='UTF-8')) < self.env.max_key_size():   
                        txn.put(word.encode(encoding='UTF-8'), _serialize_pickle(vector))  
                        #txn.put(word.encode(encoding='UTF-8'), _serialize_byteio(vector))
                        i += 1

                    # commit batch
                    if i % batch_size == 0:
                        txn.commit()
                        txn = self.env.begin(write=True)

            #if i % batch_size != 0:
            txn.commit()   
            if nbWords == 0:
                nbWords = i
            self.vocab_size = nbWords
            print('embeddings loaded for', nbWords, "words and", self.embed_size, "dimensions")


    def make_embeddings_simple(self, name="fasttext-crawl", hasHeader=True):
        if self.embedding_lmdb_path is None:
            raise (OSError('Path to embedding database does not exist'))
        # check if the lmdb database exists
        envFilePath = os.path.join(self.embedding_lmdb_path, name)
        if os.path.isdir(envFilePath):
            # open the database in read mode
            self.env = lmdb.open(envFilePath, readonly=True)
            # we need to set self.embed_size and self.vocab_size
            with self.env.begin() as txn:
                stats = txn.stat()
                size = stats['entries']
                self.vocab_size = size

            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    vector = _deserialize_pickle(value)
                    self.embed_size = vector.shape[0]
                    break
                cursor.close()

            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env.close()
            self.env = lmdb.open(envFilePath, readonly=True)
        else: 
            # create and load the database in write mode
            self.env = lmdb.open(envFilePath, map_size=map_size)
            self.make_embeddings_lmdb(name, hasHeader)


    def _get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        return None


    def get_word_vector(self, word):
        if self.env is None:
            # db not available
            raise (OSError('The embedding database does not exist'))
        try:    
            with self.env.begin() as txn:
                vector = txn.get(word.encode(encoding='UTF-8'))
                if vector:
                    word_vector = _deserialize_pickle(vector)
                    vector = None
                else:
                    word_vector =  np.zeros((self.embed_size,), dtype=np.float32)
                    # alternatively, initialize with random negative values
                    #return np.random.uniform(low=-0.5, high=0.0, size=(embeddings.shape[1],))
                    # alternatively use fasttext OOV ngram possibilities (if ngram available)
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env.close()
            envFilePath = os.path.join(self.embedding_lmdb_path, self.name)
            self.env = lmdb.open(envFilePath, readonly=True)
            return self.get_word_vector(word)
        return word_vector

    def get_word_vector_old(self, word):
        if word in self.model:
            return self.model[word]
        else:
            # for unknown word, we use a vector filled with 0.0
            return np.zeros((self.embed_size,), dtype=np.float32)
            # alternatively, initialize with random negative values
            #return np.random.uniform(low=-0.5, high=0.0, size=(embeddings.shape[1],))
            # alternatively use fasttext OOV ngram possibilities (if ngram available)

def _serialize_byteio(array):
    memfile = io.BytesIO()
    np.save(memfile, array)
    memfile.seek(0)
    return memfile.getvalue()

def _deserialize_byteio(serialized):
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile)

def _serialize_pickle(a):
    return pickle.dumps(a)

def _deserialize_pickle(serialized):
    return pickle.loads(serialized)

def _get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
