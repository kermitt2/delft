import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)
import keras
from sequenceLabelling.preprocess import to_vector_single
from utilities.Tokenizer import tokenizeAndFilterSimple

# generate batch of data to feed sequence labelling model, both for training and prediction

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, 
                batch_size=24, 
                preprocessor=None, 
                word_embed_size=300, 
                char_embed_size=25, 
                embeddings=(), 
                tokenize=False, 
                shuffle=True):
        'Initialization'
        self.x = x
        self.y = y
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        batch_x, batch_c, batch_l, batch_y = self.__data_generation(index)
        return [batch_x, batch_c, batch_l], batch_y

    def shuffle_pair(self, a, b):
        # generate permutation index array
        permutation = np.random.permutation(a.shape[0])
        # shuffle the two arrays
        return a[permutation], b[permutation]

    def on_epoch_end(self):
        # shuffle dataset at each epoch
        if self.shuffle == True:
            if self.y is None:
                np.random.shuffle(self.x)
            else:      
                self.shuffle_pair(self.x,self.y)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' 
        max_iter = min(self.batch_size, len(self.x)-self.batch_size*index)

        # restrict data to index window
        sub_x = self.x[(index*self.batch_size):(index*self.batch_size)+max_iter]

        # tokenize texts in self.x if not already done
        max_length_x = 0
        if self.tokenize:
            x_tokenized = []
            for i in range(0, max_iter):
                tokens = tokenizeAndFilterSimple(sub_x[i])
                if len(tokens) > max_length_x:
                    max_length_x = len(tokens)
                x_tokenized.append(tokens)
        else:
            for tokens in sub_x:
                if len(tokens) > max_length_x:
                    max_length_x = len(tokens)
            x_tokenized = sub_x

        batch_x = np.zeros((max_iter, max_length_x, self.word_embed_size), dtype='float32')
        #batch_c = np.zeros((max_iter, max_length_x, self.char_embed_size), dtype='float32')

        batch_y = None
        max_length_y = max_length_x
        if self.y is not None:
            # note: tags are always already "tokenized",
            batch_y = np.zeros((max_iter, max_length_y), dtype='float32')

        # generate data
        for i in range(0, max_iter):
            # store sample embeddings 
            batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x, self.word_embed_size)

            # store tag embeddings
            if self.y is not None:
                # hot one encoding for tags
                batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]

        if self.y is not None:
            batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y)
        else:
            batches = self.preprocessor.transform(x_tokenized)
        batch_c = batches[0]
        batch_length = batches[1]
        
        return batch_x, np.asarray(batch_c), batch_length, batch_y
