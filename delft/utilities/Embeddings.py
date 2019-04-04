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
import hashlib, struct
from tqdm import tqdm
import mmap
import tensorflow as tf
import keras.backend as K

# for fasttext binary embeddings
fasttext_support = True
try:
    import fastText
except ImportError as e:
    fasttext_support = False

# for ELMo embeddings
from delft.utilities.bilm.data import Batcher, TokenBatcher
from delft.utilities.bilm.model import BidirectionalLanguageModel, dump_token_embeddings
from delft.utilities.bilm.elmo import weight_layers

from delft.utilities.Tokenizer import tokenizeAndFilterSimple

# for FLAIR embeddings
from delft.utilities.flair.DeLFTFlairEmbeddings import DeLFTFlairEmbeddings
#from flair.embeddings import FlairEmbeddings
from flair.data import Sentence, Token

# gensim is used to exploit .bin FastText embeddings, in particular the OOV with the provided ngrams
#from gensim.models import FastText

# this is the default init size of a lmdb database for embeddings
# based on https://github.com/kermitt2/nerd/blob/master/src/main/java/com/scienceminer/nerd/kb/db/KBDatabase.java
# and https://github.com/kermitt2/nerd/blob/0.0.3/src/main/java/com/scienceminer/nerd/kb/db/KBDatabaseFactory.java#L368
map_size = 100 * 1024 * 1024 * 1024 

# dim of ELMo embeddings (2 times the dim of the LSTM for LM)
ELMo_embed_size = 1024

# for FLAIR, we distinguish direction of the models
FORWARD = 0
BACKWARD = 1

class Embeddings(object):

    def __init__(self, name, path='./embedding-registry.json', lang='en', extension='vec', use_ELMo=False, use_FLAIR=False):
        self.name = name
        self.embed_size = 0
        self.static_embed_size = 0
        self.vocab_size = 0
        self.model = {}
        self.registry = self._load_embedding_registry(path)
        self.lang = lang
        self.extension = extension
        self.embedding_lmdb_path = None
        if self.registry is not None:
            self.embedding_lmdb_path = self.registry["embedding-lmdb-path"]
        self.env = None
        self.make_embeddings_simple(name)
        self.static_embed_size = self.embed_size
        self.bilm = None

        # below init for using ELMo embeddings
        self.use_ELMo = use_ELMo
        self.use_ELMo_cache = False
        self.embedding_ELMo_cache = None
        self.env_ELMo = None
        if use_ELMo:
            self.make_ELMo()
            self.embed_size = ELMo_embed_size + self.embed_size
            description = self._get_description('elmo-'+self.lang)
            # clean possible remaining cache
            self.clean_ELMo_cache()
            if description:
                if description['cache-training']:
                    self.use_ELMo_cache = True
                    self.embedding_ELMo_cache = os.path.join(description["path-cache"], "cache")        
                    # create and load a cache in write mode, it will be used only for training
                    self.env_ELMo = lmdb.open(self.embedding_ELMo_cache, map_size=map_size)

        # below init for using FLAIR embeddings
        self.use_FLAIR = use_FLAIR
        self.use_FLAIR_cache_forward = False
        self.use_FLAIR_cache_backward = False
        self.embedding_FLAIR_cache_forward = None
        self.embedding_FLAIR_cache_backward = None
        self.env_FLAIR_forward = None
        self.env_FLAIR_backward = None
        if use_FLAIR:
            self.make_FLAIR()
            self.embed_size = self.flair_forward.embedding_length + self.flair_backward.embedding_length + self.embed_size
            description_forward = self._get_description('flair-forward-en-news')
            description_backward = self._get_description('flair-backward-en-news')
            # clean possible remaining cache
            self.clean_FLAIR_cache()
            if description_forward:
                if description_forward['cache-training']:
                    self.use_FLAIR_cache_forward = True
                    self.embedding_FLAIR_cache_forward = os.path.join(description_forward["path-cache"], "cache_forward")
                    # create and load a cache in write mode, it will be used only for training
                    self.env_FLAIR_forward = lmdb.open(self.embedding_FLAIR_cache_forward, map_size=map_size)
            if description_backward:
                if description_backward['cache-training']:
                    self.use_FLAIR_cache_backward = True
                    self.embedding_FLAIR_cache_backward = os.path.join(description_backward["path-cache"], "cache_backward")
                    # create and load a cache in write mode, it will be used only for training
                    self.env_FLAIR_backward = lmdb.open(self.embedding_FLAIR_cache_backward, map_size=map_size)


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
        """
        Store simple word embbedings in memory, this approach should be avoided for performance
        reasons
        """
        nbWords = 0
        print('loading embeddings...')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            self.lang = description["lang"]
            print("path:", embeddings_path)
            if self.extension == 'bin':
                self.model = fastText.load_model(embeddings_path)
                nbWords = len(self.model.get_words())
                self.embed_size = self.model.get_dimension()
            else:
                if embeddings_type == "glove":
                    hasHeader = False
                with open(embeddings_path, encoding='utf8') as f:
                    for line in f:
                        line = line.strip()
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
                        #if embeddings_type == 'glove':
                        vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                        #else:
                        #    vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
                        if self.embed_size == 0:
                            self.embed_size = len(vector)
                        self.model[word] = vector
                if nbWords == 0:
                    nbWords = len(self.model)
            print('embeddings loaded for', nbWords, "words and", self.embed_size, "dimensions")

    def make_embeddings_lmdb(self, name="fasttext-crawl", hasHeader=True):
        """
        Store simple word embbedings in LMDB
        """
        nbWords = 0
        print('\nCompiling embeddings... (this is done only one time per embeddings at first launch)')
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
            with open(embeddings_path, encoding='utf8') as f:
                for line in f:
                    nb_lines += 1

            with open(embeddings_path, encoding='utf8') as f:
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
                    #if embeddings_type == 'glove':
                    try:
                        if line[len(line)-1] == '\n':
                            vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
                        else:
                            vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                    
                        #vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                    except:
                        print(len(line))
                        print(line[1:len(line)])
                    #else:
                    #    vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
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
        """
        Init simple word embeddings (e.g. Glove, FastText, w2v)
        """
        description = self._get_description(name)
        if description is not None:
            self.extension = description["format"]

        if self.extension == "bin":
            if fasttext_support == True:
                print("embeddings are of .bin format, so they will be loaded in memory...")
                self.make_embeddings_simple_in_memory(name, hasHeader)
            else:
                if not (sys.platform == 'linux' or sys.platform == 'darwin'):
                    raise ValueError('FastText .bin format not supported for your platform')
                else:
                    raise ValueError('Go to the documentation to get more information on how to install FastText .bin support')

        elif self.embedding_lmdb_path is None or self.embedding_lmdb_path == "None":
            print("embedding_lmdb_path is not specified in the embeddings registry, so the embeddings will be loaded in memory...")
            self.make_embeddings_simple_in_memory(name, hasHeader)
        else:    
            # check if the lmdb database exists
            envFilePath = os.path.join(self.embedding_lmdb_path, name)
            load_db = True
            if os.path.isdir(envFilePath):
                description = self._get_description(name)
                if description is not None:
                    self.lang = description["lang"]

                # open the database in read mode
                self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=4)
                if self.env:
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

                    if self.vocab_size != 0 and self.embed_size != 0:
                        load_db = False

                        # no idea why, but we need to close and reopen the environment to avoid
                        # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
                        # when opening new transaction !
                        self.env.close()
                        self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=2)

            if load_db: 
                # create and load the database in write mode
                self.env = lmdb.open(envFilePath, map_size=map_size)
                self.make_embeddings_lmdb(name, hasHeader)

    def make_ELMo(self):
        # Location of pretrained BiLM for the specified language
        # TBD check if ELMo language resources are present
        description = self._get_description('elmo-'+self.lang)
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

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                # the reuse=True scope reuses weights from the whole context 
                self.embeddings_op = self.bilm(self.character_ids)
                self.elmo_input = weight_layers('input', self.embeddings_op, l2_coef=0.0)

    def make_FLAIR(self):
        # Location of pretrained BiLM for the specified language
        # TBD check if FLAIR language resources are present
        #description_forward = self._get_description('flair-forward-en-mix')
        #description_backward = self._get_description('flair-backward-en-mix')

        description_forward = self._get_description('flair-forward-en-news')
        description_backward = self._get_description('flair-backward-en-news')

        if description_forward is not None:
            self.lang = description_forward["lang"]
            model_file = description_forward["path_weights"]
            print('init FLAIR forward LM')
            self.flair_backward = DeLFTFlairEmbeddings(model_file)

        if description_backward is not None:
            model_file = description_backward["path_weights"]
            print('init FLAIR backward LM')
            self.flair_forward = DeLFTFlairEmbeddings(model_file)

    def get_sentence_vector_only_ELMo(self, token_list):
        """
        Return the ELMo embeddings only, for a full sentence
        """

        if not self.use_ELMo:
            print("Warning: ELMo embeddings requested but embeddings object wrongly initialised")
            return

        # Create batches of data
        local_token_ids = self.batcher.batch_sentences(token_list)
        max_size_sentence = local_token_ids[0].shape[0]
        # check lmdb cache
        elmo_result = self.get_ELMo_lmdb_vector(token_list, max_size_sentence)
        if elmo_result is not None:
            return elmo_result

        with tf.Session() as sess:
            # weird, for this cpu is faster than gpu (1080Ti !)
            with tf.device("/cpu:0"):
                # It is necessary to initialize variables once before running inference
                sess.run(tf.global_variables_initializer())

                # Compute ELMo representations (2 times as a heavy warm-up)
                elmo_result = sess.run(
                    self.elmo_input['weighted_op'],
                    feed_dict={self.character_ids: local_token_ids}
                )
                elmo_result = sess.run(
                    self.elmo_input['weighted_op'],
                    feed_dict={self.character_ids: local_token_ids}
                )
                # if required, cache computation
                if self.use_ELMo_cache:
                    self.cache_ELMo_lmdb_vector(token_list, elmo_result)
        return elmo_result

    def get_sentence_vector_with_ELMo(self, token_list):
        """
        Return a concatenation of standard embeddings (e.g. Glove) and ELMo embeddings 
        for a full sentence, this is the usual usage
        """
        if not self.use_ELMo:
            print("Warning: ELMo embeddings requested but embeddings object wrongly initialised")
            return

        # Create batches of data
        local_token_ids = self.batcher.batch_sentences(token_list)
        max_size_sentence = local_token_ids[0].shape[0]

        # check lmdb cache
        elmo_result = self.get_ELMo_lmdb_vector(token_list, max_size_sentence) 
        if elmo_result is None:
            with tf.Session() as sess:
                # weird, for this cpu is faster than gpu (1080Ti !)
                with tf.device("/cpu:0"):
                    # It is necessary to initialize variables once before running inference
                    sess.run(tf.global_variables_initializer())

                    # Compute ELMo representations (2 times as a heavy warm-up)
                    elmo_result = sess.run(
                        self.elmo_input['weighted_op'],
                        feed_dict={self.character_ids: local_token_ids}
                    )
                    elmo_result = sess.run(
                        self.elmo_input['weighted_op'],
                        feed_dict={self.character_ids: local_token_ids}
                    )
                    # if required, cache computation
                    if self.use_ELMo_cache:
                        self.cache_ELMo_lmdb_vector(token_list, elmo_result)
        
        concatenated_result = np.zeros((len(token_list), max_size_sentence-2, self.embed_size), dtype=np.float32)
        for i in range(0, len(token_list)):
            for j in range(0, len(token_list[i])):
                concatenated_result[i][j] = np.concatenate((elmo_result[i][j], self.get_word_vector(token_list[i][j]).astype('float32')), )
        return concatenated_result

    def get_sentence_vector_with_FLAIR(self, token_list):
        """
        Return a concatenation of standard embeddings (e.g. Glove) and FLAIR embeddings (forward and backward LM) 
        for a full sentence (recommended usage from FLAIR developers)
        """
        if not self.use_FLAIR:
            print("Warning: FLAIR embeddings requested but embeddings object wrongly initialised")
            return

        FLAIR_embed_size = self.flair_backward.embedding_length

        max_size_sentence = 0
        for tokens in token_list:
            if len(tokens) > max_size_sentence:
                max_size_sentence = len(tokens)

        sentences = []
        for tokens in token_list:
            sentence = Sentence()
            for token in tokens:
                sentence.add_token(Token(token))
            sentences.append(sentence)

        # check lmdb cache
        flair_result_forward = self.get_FLAIR_lmdb_vector(token_list, max_size_sentence, FORWARD) 
        if flair_result_forward is None:
            flair_result_forward = np.zeros((len(token_list), max_size_sentence, FLAIR_embed_size), dtype='float32')
            self.flair_forward.embed(sentences)
            for i in range(len(sentences)):
                j = 0
                for token in sentences[i]:
                    flair_result_forward[i,j] = token.embedding.numpy()
                    token.clear_embeddings()
                    j += 1

            # if required, cache computation
            if self.use_FLAIR_cache_forward:
                self.cache_FLAIR_lmdb_vector(token_list, flair_result_forward, FORWARD)

        flair_result_backward = self.get_FLAIR_lmdb_vector(token_list, max_size_sentence, BACKWARD) 
        if flair_result_backward is None:
            flair_result_backward = np.zeros((len(token_list), max_size_sentence, FLAIR_embed_size), dtype='float32')
            self.flair_backward.embed(sentences)
            for i in range(len(sentences)):
                j = 0
                for token in sentences[i]:
                    flair_result_backward[i,j] = token.embedding.numpy()
                    token.clear_embeddings()
                    j += 1

            # if required, cache computation
            if self.use_FLAIR_cache_backward:
                self.cache_FLAIR_lmdb_vector(token_list, flair_result_backward, BACKWARD)

        concatenated_result = np.zeros((len(token_list), max_size_sentence, self.embed_size), dtype=np.float32)
        for i in range(0, len(token_list)):
            for j in range(0, len(token_list[i])):
                concatenated_result[i][j] = np.concatenate(
                                (flair_result_forward[i][j], 
                                 flair_result_backward[i][j], 
                                 self.get_word_vector(token_list[i][j]).astype('float32')), )
        return concatenated_result


    def _get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        for emb in self.registry["embeddings-contextualized"]:
            if emb["name"] == name:
                return emb
        return None

    def get_word_vector(self, word):
        """
        Get simple static embeddings (e.g. glove) for a given token
        """
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.env is None or self.extension == 'bin':
            # db not available or embeddings in bin format, the embeddings should be available in memory (normally!)
            return self.get_word_vector_in_memory(word)
        try:    
            with self.env.begin() as txn:
                txn = self.env.begin()   
                vector = txn.get(word.encode(encoding='UTF-8'))
                if vector:
                    word_vector = _deserialize_pickle(vector)
                    vector = None
                else:
                    word_vector = np.zeros((self.static_embed_size,), dtype=np.float32)
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

    def get_ELMo_lmdb_vector(self, token_list, max_size_sentence):
        """
        Try to get the ELMo embeddings for a sequence cached in LMDB
        """
        if self.env_ELMo is None:
            # db cache not available, we don't cache ELMo stuff
            return None
        if not self.use_ELMo_cache:
            return None
        try:    
            ELMo_vector = np.zeros((len(token_list), max_size_sentence-2, ELMo_embed_size), dtype='float32')
            with self.env_ELMo.begin() as txn:
                for i in range(0, len(token_list)):
                    txn = self.env_ELMo.begin()
                    # get a hash for the token_list
                    the_hash = list_digest(token_list[i])
                    vector = txn.get(the_hash.encode(encoding='UTF-8'))
                    if vector:
                        # adapt expected shape/padding
                        local_embeddings = _deserialize_pickle(vector)
                        if local_embeddings.shape[0] > max_size_sentence-2:
                            # squeeze the extra padding space
                            ELMo_vector[i] = local_embeddings[:max_size_sentence-2,]
                        elif local_embeddings.shape[0] == max_size_sentence-2:
                            # bingo~!
                            ELMo_vector[i] = local_embeddings
                        else:
                            # fill the missing space with padding
                            filler = np.zeros((max_size_sentence-(local_embeddings.shape[0]+2), ELMo_embed_size), dtype='float32')
                            ELMo_vector[i] = np.concatenate((local_embeddings, filler))
                        vector = None
                    else:
                        return None
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env_ELMo.close()
            self.env_ELMo = lmdb.open(self.embedding_ELMo_cache, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self.get_ELMo_lmdb_vector(token_list)
        return ELMo_vector

    def get_FLAIR_lmdb_vector(self, token_list, max_size_sentence, direction):
        """
            Try to get the ELMo embeddings for a sequence cached in LMDB
        """
        if self.env_FLAIR_forward is None and direction == FORWARD:
            # db cache not available, we don't cache FLAIR stuff
            return None
        if self.env_FLAIR_backward is None and direction == BACKWARD:
            # db cache not available, we don't cache FLAIR stuff
            return None    
        if not self.use_FLAIR_cache_forward and direction == FORWARD:
            return None
        if not self.use_FLAIR_cache_backward and direction == BACKWARD:
            return None

        if direction == FORWARD:
            env_FLAIR = self.env_FLAIR_forward
            FLAIR_embed_size = self.flair_forward.embedding_length
        else:
            env_FLAIR = self.env_FLAIR_backward
            FLAIR_embed_size = self.flair_backward.embedding_length

        try:    
            FLAIR_vector = np.zeros((len(token_list), max_size_sentence, FLAIR_embed_size), dtype='float32')
            with env_FLAIR.begin() as txn:
                for i in range(0, len(token_list)):
                    txn = env_FLAIR.begin()
                    # get a hash for the token_list
                    the_hash = list_digest(token_list[i])
                    vector = txn.get(the_hash.encode(encoding='UTF-8'))
                    if vector:
                        # adapt expected shape/padding
                        local_embeddings = _deserialize_pickle(vector)
                        if local_embeddings.shape[0] > max_size_sentence:
                            # squeeze the extra padding space
                            FLAIR_vector[i] = local_embeddings[:max_size_sentence,]
                        elif local_embeddings.shape[0] == max_size_sentence:
                            # bingo~!
                            FLAIR_vector[i] = local_embeddings
                        else:
                            # fill the missing space with padding
                            filler = np.zeros((max_size_sentence-(local_embeddings.shape[0]), FLAIR_embed_size), dtype='float32')
                            FLAIR_vector[i] = np.concatenate((local_embeddings, filler))
                        vector = None
                    else:
                        return None
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            env_FLAIR.close()
            if direction == FORWARD:
                env_FLAIR = lmdb.open(self.embedding_FLAIR_cache_forward, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            else:
                env_FLAIR = lmdb.open(self.embedding_FLAIR_cache_backward, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self.get_FLAIR_lmdb_vector(token_list)
        return FLAIR_vector

    def cache_ELMo_lmdb_vector(self, token_list, ELMo_vector):
        """
            Cache in LMDB the ELMo embeddings for a given sequence 
        """
        if self.env_ELMo is None:
            # db cache not available, we don't cache ELMo stuff
            return None
        if not self.use_ELMo_cache:
            return None 
        txn = self.env_ELMo.begin(write=True)
        for i in range(0, len(token_list)):
            # get a hash for the token_list
            the_hash = list_digest(token_list[i])
            txn.put(the_hash.encode(encoding='UTF-8'), _serialize_pickle(ELMo_vector[i]))  
        txn.commit()

    def cache_FLAIR_lmdb_vector(self, token_list, FLAIR_vector, direction):
        """
            Cache in LMDB the FLAIR embeddings for a given sequence 
        """
        if direction == FORWARD:
            env_FLAIR = self.env_FLAIR_forward
        else:
            env_FLAIR = self.env_FLAIR_backward
        if env_FLAIR is None:
            # db cache not available, we don't cache FLAIR stuff
            return None
        if not self.use_FLAIR_cache_forward and direction == FORWARD:
            return None
        if not self.use_FLAIR_cache_backward and direction == BACKWARD:
            return None
        txn = env_FLAIR.begin(write=True)
        for i in range(0, len(token_list)):
            # get a hash for the token_list
            the_hash = list_digest(token_list[i])
            txn.put(the_hash.encode(encoding='UTF-8'), _serialize_pickle(FLAIR_vector[i]))  
        txn.commit()

    def clean_ELMo_cache(self):
        """
        Delete ELMo embeddings cache, this takes place normally after the completion of a training
        """
        if self.env_ELMo is not None:
            self.env_ELMo.close()
            self.env_ELMo = None
        if self.embedding_ELMo_cache is not None:
            for file in os.listdir(self.embedding_ELMo_cache): 
                file_path = os.path.join(self.embedding_ELMo_cache, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.embedding_ELMo_cache)
            self.embedding_ELMo_cache = None

    def clean_FLAIR_cache(self):
        """
            Delete FLAIR embeddings cache, this takes place normally after the completion of a training
        """
        if self.env_FLAIR_forward is not None:
            self.env_FLAIR_forward.close()
            self.env_FLAIR_forward = None
        if self.embedding_FLAIR_cache_forward is not None:
            for file in os.listdir(self.embedding_FLAIR_cache_forward): 
                file_path = os.path.join(self.embedding_FLAIR_cache_forward, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.embedding_FLAIR_cache_forward)
            self.embedding_FLAIR_cache_forward = None

        if self.env_FLAIR_backward is not None:
            self.env_FLAIR_backward.close()
            self.env_FLAIR_backward = None
        if self.embedding_FLAIR_cache_backward is not None:
            for file in os.listdir(self.embedding_FLAIR_cache_backward): 
                file_path = os.path.join(self.embedding_FLAIR_cache_backward, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.embedding_FLAIR_cache_backward)
            self.embedding_FLAIR_cache_backward = None

    def get_word_vector_in_memory(self, word):
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.extension == 'bin':
            return self.model.get_word_vector(word)
        if word in self.model:
            return self.model[word]
        else:
            # for unknown word, we use a vector filled with 0.0
            return np.zeros((self.static_embed_size,), dtype=np.float32)
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


def list_digest(strings):
    hash = hashlib.sha1()
    for s in strings:
        hash.update(struct.pack("I", len(s)))
        hash.update(s.encode(encoding='UTF-8'))
    return hash.hexdigest()


def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def test():
    embeddings = Embeddings("glove-840B", use_ELMo=True)
    token_list = [['This', 'is', 'a', 'test', '.']]
    vect = embeddings.get_sentence_vector_ELMo(token_list)
    embeddings.cache_ELMo_lmdb_vector(token_list, vect)
    vect = embeddings.get_sentence_vector_ELMo(token_list)

    embeddings.clean_ELMo_cache()
