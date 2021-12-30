import numpy as np
from delft.utilities.Utilities import truncate_batch_values
from delft.utilities.numpy import shuffle_triple_with_view

# seed is fixed for reproducibility
np.random.seed(7)

import tensorflow.keras as keras
from delft.sequenceLabelling.preprocess import to_vector_single, to_casing_single
from delft.utilities.Tokenizer import tokenizeAndFilterSimple
import tensorflow as tf

# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                bert_preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None):
        # self.x and self.y are shuffled view of self.original_x and self.original_y
        self.original_x = self.x = x
        self.original_y = self.y = y
        # features here are optional additional features provided in the case of GROBID input for instance
        self.original_features = self.features = features
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.bert_preprocessor = bert_preprocessor
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        self.on_epoch_end()

    def __len__(self):
        '''
        Give the number of batches per epoch
        '''
        # The number of batches is set so that each training sample is seen at most once per epoch
        if self.original_x is None:
            return 0
        elif (len(self.original_x) % self.batch_size) == 0:
            return int(np.floor(len(self.original_x) / self.batch_size))
        else:
            return int(np.floor(len(self.original_x) / self.batch_size) + 1)

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        batch_x, batch_c, batch_f, batch_a, batch_l, batch_y = self.__data_generation(index)
        if self.preprocessor.return_word_embeddings:
            if self.preprocessor.return_casing:
                return [batch_x, batch_c, batch_a, batch_l], batch_y
            elif self.preprocessor.return_features:
                return [batch_x, batch_c, batch_f, batch_l], batch_y
            else:
                return [batch_x, batch_c, batch_l], batch_y

        if self.preprocessor.return_bert_embeddings:
            if self.preprocessor.return_features:  
                return [batch_x, batch_f, batch_l], batch_y
            else:
                return [batch_x, batch_l], batch_y


    def on_epoch_end(self):
        '''
        In case we are training, we can shuffle the training data for the next epoch.
        '''
        # If we are predicting, we don't need to shuffle
        if self.original_y is None:
            return

        # shuffle dataset at each epoch
        if self.shuffle:
            self.x, self.y, self.features = shuffle_triple_with_view(self.original_x, self.original_y, self.original_features)

    def __data_generation(self, index):
        '''
        Generates data containing batch_size samples
        '''
        max_iter = min(self.batch_size, len(self.original_x)-self.batch_size * index)

        # restrict data to index window
        sub_x = self.x[(index * self.batch_size):(index * self.batch_size) + max_iter]

        # tokenize texts in self.x if not already done
        if self.tokenize:
            x_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in sub_x
            ]
        else:
            x_tokenized = sub_x

        max_length_f = max_length_x = max((len(tokens) for tokens in x_tokenized))

        if self.max_sequence_length and max_length_x > self.max_sequence_length:
            max_length_x = self.max_sequence_length
            # truncation of sequence at max_sequence_length
            x_tokenized = np.asarray(truncate_batch_values(x_tokenized, self.max_sequence_length), dtype=object)

        # prevent sequence of length 1 alone in a batch (this causes an error in tf)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        # generate data
        batch_a = np.zeros((max_iter, max_length_x), dtype='float32')
        batch_y = None
        batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
        
        # store tag embeddings
        if self.y is not None:
            # note: tags are always already "tokenized" by input token
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            max_length_y = max((len(y_row) for y_row in batch_y))

            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                # truncation of sequence at max_sequence_length
                 batch_y = np.asarray(truncate_batch_values(batch_y, self.max_sequence_length), dtype=object)

        # Note: for the moment it's word embeddings or transformers embeddings, not both !
        if self.preprocessor.return_word_embeddings:
            for i in range(0, max_iter):
                batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x)
        elif self.preprocessor.return_bert_embeddings:
            # for input as sentence piece token index for BERT layer

            if self.y == None:
                input_ids, input_masks, input_segments, _ = self.bert_preprocessor.create_batch_input_bert(
                                                                            x_tokenized, 
                                                                            maxlen=self.max_sequence_length)
            else:
                input_ids, input_masks, input_segments, input_labels, _ = self.bert_preprocessor.tokenize_and_align_labels(
                                                                            x_tokenized, 
                                                                            batch_y,
                                                                            maxlen=self.max_sequence_length)

            # we can use only input indices, but could be reconsidered
            batch_x = np.asarray(input_ids, dtype=np.int32)
            #batch_x_masks = np.asarray(input_masks, dtype=np.int32)
            #batch_x_segments = np.asarray(input_segments, dtype=np.int32)

            if self.y != None:
                batch_y = np.asarray(input_labels, dtype=np.int32)

        if self.preprocessor.return_casing:
            for i in range(0, max_iter):
                batch_a[i] = to_casing_single(x_tokenized[i], max_length_x)

        batch_f = np.zeros((batch_x.shape[0:2]), dtype='int32')

        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            if self.max_sequence_length and max_length_f > self.max_sequence_length:
                max_length_f = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                sub_f = truncate_batch_values(sub_f, self.max_sequence_length)

            batch_f = self.preprocessor.transform_features(sub_f, extend=extend)

        if self.y is not None:
            batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend)
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0])
        batch_l = batches[1]

        return batch_x, batch_c, batch_f, batch_a, batch_l, batch_y
