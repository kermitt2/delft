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

# gensim is used to exploit .bin FastText embeddings, in particular the OOV with the provided ngrams
#from gensim.models import FastText

# this is the default init size of a lmdb database for embeddings
# based on https://github.com/kermitt2/nerd/blob/master/src/main/java/com/scienceminer/nerd/kb/db/KBDatabase.java
# and https://github.com/kermitt2/nerd/blob/0.0.3/src/main/java/com/scienceminer/nerd/kb/db/KBDatabaseFactory.java#L368
map_size = 100 * 1024 * 1024 * 1024 

# dim of ELMo embeddings (2 times the dim of the LSTM for LM)
ELMo_embed_size = 1024

class Embeddings(object):

    def __init__(self, name, path='./embedding-registry.json', lang='en', extension='vec', use_ELMo=False):
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
        if use_ELMo:
            self.make_ELMo()
            self.embed_size = ELMo_embed_size + self.embed_size
            description = self._get_description('elmo-en')
            self.env_ELMo = None
            if description:
                self.embedding_ELMo_cache = os.path.join(description["path-cache"], "cache")
                # clean possible remaining cache
                self.clean_ELMo_cache()
                # create and load a cache in write mode, it will be used only for training
                self.env_ELMo = lmdb.open(self.embedding_ELMo_cache, map_size=map_size)

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

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                # the reuse=True scope reuses weights from the whole context 
                self.embeddings_op = self.bilm(self.character_ids)
                self.elmo_input = weight_layers('input', self.embeddings_op, l2_coef=0.0)

    def dump_ELMo_token_embeddings(self, x_train):
        if not self.use_ELMo:
            print("Warning: ELMo embeddings dump requested but embeddings object wrongly initialised")
            return

        description = self._get_description('elmo-en')
        if description is not None:
            print("Building ELMo token dump")

            self.lang = description["lang"]
            options_file = description["path-config"]
            weight_file = description["path_weights"]
            working_path = description["path-cache"]

            all_tokens = set(['<S>', '</S>'])
            for i in range(0, len(x_train)):
                # as it is training, it is already tokenized
                tokens = x_train[i]
                for token in tokens:
                    if token not in all_tokens:
                       all_tokens.add(token)

            vocab_file = os.path.join(working_path, 'vocab_small.txt')
            with open(vocab_file, 'w') as fout:
                fout.write('\n'.join(all_tokens))

            tf.reset_default_graph()
            token_embedding_file = os.path.join(working_path, 'elmo_token_embeddings.hdf5')
            dump_token_embeddings(
                vocab_file, options_file, weight_file, token_embedding_file
            )
            tf.reset_default_graph()

            self.batcher_token_dump = TokenBatcher(vocab_file)

            self.bilm_token_dump = BidirectionalLanguageModel(
                options_file,
                weight_file,
                use_character_inputs=False,
                embedding_weight_file=token_embedding_file
            )

            self.token_ids = tf.placeholder('int32', shape=(None, None))
            self.embeddings_op_token_dump = self.bilm_token_dump(self.token_ids)
            """
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                # the reuse=True scope reuses weights from the whole context 
                self.elmo_input_token_dump = weight_layers('input', self.embeddings_op_token_dump, l2_coef=0.0)
            """
            print("ELMo token dump completed")

    def get_sentence_vector_only_ELMo(self, token_list):
        """
            Return the ELMo embeddings only for a full sentence
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
                #cache computation
                self.cache_ELMo_lmdb_vector(token_list, elmo_result)
        return elmo_result

    def get_sentence_vector_with_ELMo(self, token_list):
        """
            Return a concatenation of standard embeddings (e.g. Glove) and ELMo embeddings 
            for a full sentence
        """
        if not self.use_ELMo:
            print("Warning: ELMo embeddings requested but embeddings object wrongly initialised")
            return
        """
        # trick to extend the context for short sentences
        token_list_extended = token_list.copy()
        #print("token_list_extended before: ", token_list_extended)
        for i in range(0, len(token_list_extended)):
            local_list = token_list_extended[i]
            j = i
            while len(local_list) <= 5:
                #print(j, local_list)
                if j < len(token_list_extended)-1:
                    local_list = local_list + token_list_extended[j+1]
                else:
                    break
                j = j + 1
            token_list_extended[i] = local_list
        #print("token_list_extended after: ", token_list_extended)

        max_size_sentence = 0
        for i in range(0, len(token_list)):
            local_length = len(token_list[i])
            if local_length > max_size_sentence:
                max_size_sentence = local_length
        """

        # Create batches of data

        #print("\ntoken_list:", token_list)
        local_token_ids = self.batcher.batch_sentences(token_list)
        #print("local_token_ids:", local_token_ids)
        max_size_sentence = local_token_ids[0].shape[0]

        '''
        i = 0
        j = 1 # <s>
        k = 1 # start of word
        for sentence in token_list:
            print('\nsentence:', sentence)
            #print('local_token_ids[i]:', local_token_ids[i])
            for token in sentence:
                print('\ntoken:', token)
                print('local_token_ids[i,j]:', local_token_ids[i,j])
                for character in token:
                    print(character, ":", local_token_ids[i][j][k])
                    k += 1
                k = 1
                j += 1
            j = 1 
            i += 1
        '''

        #elmo_result = np.zeros((len(token_list), max_size_sentence-2, ELMo_embed_size), dtype='float32')
        #elmo_result = np.random.rand(len(token_list), max_size_sentence-2, ELMo_embed_size)
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
                    #cache computation
                    self.cache_ELMo_lmdb_vector(token_list, elmo_result)
        
        concatenated_result = np.zeros((len(token_list), max_size_sentence-2, self.embed_size), dtype=np.float32)
        #concatenated_result = np.random.rand(elmo_result.shape[0], max_size_sentence-2, self.embed_size)
        for i in range(0, len(token_list)):
            for j in range(0, len(token_list[i])):
                #if is_int(token_list[i][j]) or is_float(token_list[i][j]):
                #dummy_result = np.zeros((elmo_result.shape[2]), dtype=np.float32)
                #concatenated_result[i][j] = np.concatenate((dummy_result, self.get_word_vector(token_list[i][j])), )
                #else:
                concatenated_result[i][j] = np.concatenate((elmo_result[i][j], self.get_word_vector(token_list[i][j]).astype('float32')), )
                #concatenated_result[i][j] = np.concatenate((self.get_word_vector(token_list[i][j]), elmo_result[i][j]), )
        return concatenated_result

    def get_sentence_vector_ELMo_with_token_dump(self, token_list):
        if not self.use_ELMo:
            print("Warning: ELMo embeddings requested but embeddings object wrongly initialised")
            return

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # the reuse=True scope reuses weights from the whole context 
            self.elmo_input_token_dump = weight_layers('input', self.embeddings_op_token_dump, l2_coef=0.0)

        # Create batches of data
        local_token_ids = self.batcher_token_dump.batch_sentences(token_list)

        with tf.Session() as sess:
            # weird, for this cpu is faster than gpu (1080Ti !)
            with tf.device("/cpu:0"):
                # It is necessary to initialize variables once before running inference
                sess.run(tf.global_variables_initializer())

                # Compute ELMo representations 
                elmo_result = sess.run(
                    self.elmo_input_token_dump['weighted_op'],
                    feed_dict={self.token_ids: local_token_ids}
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
        """
            Get static embeddings (e.g. glove) for a given token
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

    def cache_ELMo_lmdb_vector(self, token_list, ELMo_vector):
        """
            Cache in LMDB the ELMo embeddings for a given sequence 
        """
        if self.env_ELMo is None:
            # db cache not available, we don't cache ELMo stuff
            return None
        txn = self.env_ELMo.begin(write=True)
        for i in range(0, len(token_list)):
            # get a hash for the token_list
            the_hash = list_digest(token_list[i])
            txn.put(the_hash.encode(encoding='UTF-8'), _serialize_pickle(ELMo_vector[i]))  
        txn.commit()

    def clean_ELMo_cache(self):
        """
            Delete ELMo embeddings cache, this takes place normally after the completion of a training
        """
        if self.env_ELMo is None:
            # db cache not available, nothing to clean
            return
        else: 
            self.env_ELMo.close()
            self.env_ELMo = None
            for file in os.listdir(self.embedding_ELMo_cache): 
                file_path = os.path.join(self.embedding_ELMo_cache, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.embedding_ELMo_cache)

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
