import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)
import keras
from delft.textClassification.preprocess import to_vector_single, to_vector_simple_with_elmo, to_vector_simple_with_bert
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

# generate batch of data to feed text classification model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=256, maxlen=300, list_classes=[], embeddings=(), shuffle=True):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.embeddings = embeddings
        self.list_classes = list_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        batch_x, batch_y = self.__data_generation(index)
        return batch_x, batch_y

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

        batch_x = np.zeros((max_iter, self.maxlen, self.embeddings.embed_size), dtype='float32')
        batch_y = None
        if self.y is not None:
            batch_y = np.zeros((max_iter, len(self.list_classes)), dtype='float32')

        x_tokenized = []
        for i in range(0, max_iter):
            tokens = tokenizeAndFilterSimple(sub_x[i])
            x_tokenized.append(tokens)

        if self.embeddings.use_ELMo:     
            #batch_x = to_vector_elmo(x_tokenized, self.embeddings, max_length_x)
            batch_x = to_vector_simple_with_elmo(x_tokenized, self.embeddings, self.maxlen)

        if self.embeddings.use_BERT:     
            batch_x = to_vector_simple_with_bert(x_tokenized, self.embeddings, self.maxlen)

        # Generate data
        for i in range(0, max_iter):
            # Store sample
            if not self.embeddings.use_ELMo and not self.embeddings.use_BERT:    
                batch_x[i] = to_vector_single(self.x[(index*self.batch_size)+i], self.embeddings, self.maxlen)

            # Store class
            # classes are numerical, so nothing to vectorize for y
            if self.y is not None:
                batch_y[i] = self.y[(index*self.batch_size)+i]

        return batch_x, batch_y
