import numpy as np
import tensorflow.keras as keras

from delft.utilities.numpy import shuffle_triple_with_view
from delft.textClassification.preprocess import to_vector_single
from delft.textClassification.preprocess import create_single_input_bert, create_batch_input_bert
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

class DataGenerator(keras.utils.Sequence):
    """
    Generate batch of data to feed text classification model, both for training and prediction.
    For Keras input based on word embeddings, we keep embeddings application outside the model 
    to make it considerably more compact and avoid duplication of embeddings layers.

    When the Keras input will feed a BERT layer, sentence piece tokenization is kept outside 
    the model so that we can serialize the model and have it more compact.  
    """
    def __init__(self, x, y, batch_size=256, maxlen=300, list_classes=[], embeddings=(), shuffle=True, bert_data=False, transformer_tokenizer=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.embeddings = embeddings
        self.list_classes = list_classes
        self.shuffle = shuffle
        self.bert_data = bert_data
        self.transformer_tokenizer = transformer_tokenizer
        self.on_epoch_end()

    def __len__(self):
        """
        Give the number of batches per epoch
        """
        # The number of batches is set so that each training sample is seen at most once per epoch
        if self.x is None:
            return 0
        elif (len(self.x) % self.batch_size) == 0:
            return int(np.floor(len(self.x) / self.batch_size))
        else:
            return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        batch_x, batch_y = self.__data_generation(index)
        return batch_x, batch_y

    def on_epoch_end(self):
        """
        In case we are training, we can shuffle the training data for the next epoch.
        """
        # If we are predicting, we don't need to shuffle
        if self.y is None:
            return

        # other shuffle dataset for next epoch
        if self.shuffle:
            self.x, self.y, _ = shuffle_triple_with_view(self.x, self.y)

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples
        """
        max_iter = min(self.batch_size, len(self.x)-self.batch_size*index)

        if not self.bert_data:
            batch_x = np.zeros((max_iter, self.maxlen, self.embeddings.embed_size), dtype='float32')
        batch_y = None
        if self.y is not None:
            batch_y = np.zeros((max_iter, len(self.list_classes)), dtype='float32')

        # Generate data
        if not self.bert_data:
            for i in range(0, max_iter):
                # for input as word embeddings: 
                batch_x[i] = to_vector_single(self.x[(index*self.batch_size)+i], self.embeddings, self.maxlen)
        else:
            # for input as sentence piece token index for BERT layer
            input_ids, input_masks, input_segments = create_batch_input_bert(self.x[(index*self.batch_size):(index*self.batch_size)+max_iter], 
                                                                             maxlen=self.maxlen, 
                                                                             transformer_tokenizer=self.transformer_tokenizer)
            # we can use only input indices, but could be reconsidered
            batch_x = np.asarray(input_ids, dtype=np.int32)
            #batch_x_masks = np.asarray(input_masks, dtype=np.int32)
            #batch_x_segments = np.asarray(input_segments, dtype=np.int32)

        # classes are numerical, so nothing to vectorize for y
        for i in range(0, max_iter):
            if self.y is not None:
                batch_y[i] = self.y[(index*self.batch_size)+i]

        return batch_x, batch_y
