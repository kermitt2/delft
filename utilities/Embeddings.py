# Manage pre-trained embeddings 

from keras.preprocessing import text, sequence
import numpy as np
import sys
import os
import os.path
import json
import lmdb
import io
import pickle
from tqdm import tqdm
import mmap
import tensorflow as tf
import keras.backend as K

# for ELMo embeddings
from utilities.bilm.data import Batcher
from utilities.bilm.model import BidirectionalLanguageModel
from utilities.bilm.elmo import weight_layers

# gensim is used to exploit .bin FastText embeddings, in particular the OOV with the provided ngrams
#from gensim.models import FastText

# this is the default init size of a lmdb database for embeddings
# based on https://github.com/kermitt2/nerd/blob/master/src/main/java/com/scienceminer/nerd/kb/db/KBDatabase.java
# and https://github.com/kermitt2/nerd/blob/0.0.3/src/main/java/com/scienceminer/nerd/kb/db/KBDatabaseFactory.java#L368
map_size = 100 * 1024 * 1024 * 1024 

class Embeddings(object):

    def __init__(self, name, path='./embedding-registry.json', lang='en', use_ELMo=False):
        self.name = name
        self.embed_size = 0
        self.vocab_size = 0
        self.model = {}
        self.registry = self._load_embedding_registry(path)
        self.lang = lang
        self.embedding_lmdb_path = None
        if self.registry is not None:
            self.embedding_lmdb_path = self.registry["embedding-lmdb-path"]
        self.env = None
        self.make_embeddings_simple(name)
        self.bilm = None
        self.use_ELMo = use_ELMo
        if use_ELMo:
            self.make_ELMo()
            self.embed_size = 1024

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

    def make_embeddings_simple_in_memory(self, name="fasttext-crawl", hasHeader=True):
        nbWords = 0
        print('loading embeddings...')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            self.lang = description["lang"]
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

    
    '''
    def make_embeddings_fasttext_bin(self, name="wiki.en.bin"):
        nbWords = 0
        print('loading embeddings...')
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            print("path:", embeddings_path)

        self.model = load_fasttext_format(embeddings_path)
    '''

    def make_embeddings_lmdb(self, name="fasttext-crawl", hasHeader=True):
        nbWords = 0
        print('\nCompiling embeddings... (this is done only one time per embeddings)')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            self.lang = description["lang"]
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
        if self.embedding_lmdb_path is None or self.embedding_lmdb_path == "None":
            print("embedding_lmdb_path is not specified in the embeddings registry, so the embeddings will be loaded in memory...")
            self.make_embeddings_simple_in_memory(name, hasHeader)
        else:    
            # check if the lmdb database exists
            envFilePath = os.path.join(self.embedding_lmdb_path, name)
            if os.path.isdir(envFilePath):
                description = self._get_description(name)
                if description is not None:
                    self.lang = description["lang"]

                # open the database in read mode
                self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=4)
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
                self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=2)
            else: 
                # create and load the database in write mode
                self.env = lmdb.open(envFilePath, map_size=map_size)
                self.make_embeddings_lmdb(name, hasHeader)

    def make_ELMo(self):
        # Location of pretrained BiLM for the specified language
        # TBD check if ELMo language resources are present
        description = self._get_description('elmo-en')
        if description is not None:
            self.lang = description["lang"]
            vocab_file = description["path-vocab"]
            options_file = description["path-config"]
            weight_file = description["path_weights"]

            print('init ELMo')

            # Create a Batcher to map text to character ids
            self.batcher = Batcher(vocab_file, 50)

            # Build the biLM graph.
            self.bilm = BidirectionalLanguageModel(options_file, weight_file)

            # Input placeholders to the biLM.
            self.character_ids = tf.placeholder('int32', shape=(None, None, 50))
            self.embeddings_op = self.bilm(self.character_ids)
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                # the reuse=True scope reuses weights from the whole context 
                elmo_input = weight_layers('input', self.embeddings_op, l2_coef=0.0)
        

    def get_sentence_vector_ELMo(self, tokens_list):
        if not self.use_ELMo:
            print("Warning: ELMo embeddings requested but embeddings object wrongly initialised")
            return

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # the reuse=True scope reuses weights from the whole context 
            elmo_input = weight_layers('input', self.embeddings_op, l2_coef=0.0)

        # Create batches of data
        token_ids = self.batcher.batch_sentences(tokens_list)

        with tf.Session() as sess:
            # weird, for this cpu is faster than gpu (1080Ti !)
            with tf.device("/cpu:0"):
                # It is necessary to initialize variables once before running inference
                sess.run(tf.global_variables_initializer())

                # Compute ELMo representations 
                elmo_result = sess.run(
                    elmo_input['weighted_op'],
                    feed_dict={self.character_ids: token_ids}
                )
        return elmo_result


    def _get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        for emb in self.registry["embeddings-contextualized"]:
            if emb["name"] == name:
                return emb
        return None


    def get_word_vector(self, word):
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.env is None:
            # db not available, the embeddings should be available in memory (normally!)
            return self.get_word_vector_in_memory(word)
        try:    
            with self.env.begin() as txn:
                txn = self.env.begin()   
                vector = txn.get(word.encode(encoding='UTF-8'))
                if vector:
                    word_vector = _deserialize_pickle(vector)
                    vector = None
                else:
                    word_vector = np.zeros((self.embed_size,), dtype=np.float32)
                    # alternatively, initialize with random negative values
                    #word_vector = np.random.uniform(low=-0.5, high=0.0, size=(self.embed_size,))
                    # alternatively use fasttext OOV ngram possibilities (if ngram available)
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env.close()
            envFilePath = os.path.join(self.embedding_lmdb_path, self.name)
            self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self.get_word_vector(word)
        return word_vector

    def get_word_vector_in_memory(self, word):
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if word in self.model:
            return self.model[word]
        else:
            # for unknown word, we use a vector filled with 0.0
            return np.zeros((self.embed_size,), dtype=np.float32)
            # alternatively, initialize with random negative values
            #return np.random.uniform(low=-0.5, high=0.0, size=(self.embed_size,))
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
