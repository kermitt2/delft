import numpy as np
# seed is fixed for reproducibility
from delft.sequenceLabelling.config import ModelConfig
from delft.utilities.numpy import shuffle_arrays

np.random.seed(7)
import keras
from delft.sequenceLabelling.preprocess import to_vector_single, to_casing_single, to_vector_elmo, \
    to_vector_simple_with_elmo, to_vector_bert, to_vector_simple_with_bert, pad_sequences, dense_to_one_hot, \
    _pad_sequences
from delft.utilities.Tokenizer import tokenizeAndFilterSimple
import tensorflow as tf
tf.set_random_seed(7)


# generate batch of data to feed sequence labelling model, both for training and prediction
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None):
        'Initialization'
        self.x = x
        self.y = y
        # features here are optional additional features provided in the case of GROBID input for instance
        self.features = features
        self.preprocessor = preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # The number of batches is set so that each training sample is seen at most once per epoch
        if self.x is None:
            return 0
        elif (len(self.x) % self.batch_size) == 0:
            return int(np.floor(len(self.x) / self.batch_size))
        else:
            return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate data for the current batch index
        batch_x, batch_c, batch_f, batch_a, batch_l, batch_y = self.__data_generation(index)
        if self.preprocessor.return_casing:
            return [batch_x, batch_c, batch_a, batch_l], batch_y
        elif self.preprocessor.return_features:
            return [batch_x, batch_c, batch_f, batch_l], batch_y
        else:
            return [batch_x, batch_c, batch_l], batch_y

    def _shuffle_dataset(self):
        arrays_to_shuffle = [self.x]
        if self.y is not None:
            arrays_to_shuffle.append(self.y)
        if self.features is not None:
            arrays_to_shuffle.append(self.features)
        shuffle_arrays(arrays_to_shuffle)

    def on_epoch_end(self):
        # If we are predicting, we don't need to shuffle
        if self.y is None:
            return

        # shuffle dataset at each epoch
        if self.shuffle:
            self._shuffle_dataset()

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
                if len(tokens) > self.max_sequence_length:
                    max_length_x = self.max_sequence_length
                    # truncation of sequence at max_sequence_length
                    tokens = tokens[:self.max_sequence_length]
                elif len(tokens) > max_length_x:
                    max_length_x = len(tokens)
                x_tokenized.append(tokens)
        else:
            for tokens in sub_x:
                if len(tokens) > max_length_x:
                    max_length_x = len(tokens)
            x_tokenized = sub_x

        # prevent sequence of length 1 alone in a batch (this causes an error in tf)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
        batch_a = np.zeros((max_iter, max_length_x), dtype='float32')

        batch_y = None
        max_length_y = max_length_x
        if self.y is not None:
            # note: tags are always already "tokenized",
            batch_y = np.zeros((max_iter, max_length_y), dtype='float32')

        if self.embeddings.use_ELMo:
            #batch_x = to_vector_elmo(x_tokenized, self.embeddings, max_length_x)
            batch_x = to_vector_simple_with_elmo(x_tokenized, self.embeddings, max_length_x, extend=extend)
        elif self.embeddings.use_BERT:
            #batch_x = to_vector_bert(x_tokenized, self.embeddings, max_length_x)
            batch_x = to_vector_simple_with_bert(x_tokenized, self.embeddings, max_length_x, extend=extend)

        # generate data
        for i in range(0, max_iter):
            # store sample embeddings
            if not self.embeddings.use_ELMo and not self.embeddings.use_BERT:
                batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x)

            if self.preprocessor.return_casing:
                batch_a[i] = to_casing_single(x_tokenized[i], max_length_x)

        # store tag embeddings
        if self.y is not None:
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]

        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            batch_f_transformed, features_length = self.preprocessor.transform_features(sub_f, extend=extend)
            batch_f_padded, _ = pad_sequences(batch_f_transformed, [0]*features_length)
            batch_f_asarray = np.asarray(batch_f_padded)
            # batch_f_list_one_hot = [
            #     dense_to_one_hot(np.asarray(batch), ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE, nlevels=2) for batch in
            #     batch_f_asarray]
            # batch_f_4dimentions = np.asarray(batch_f_list_one_hot)
            # batch_f_shape = np.asarray(batch_f_padded).shape
            # batch_f = batch_f_4dimentions.reshape(batch_f_shape[0], batch_f_shape[1],
            #                                       batch_f_shape[2] * batch_f_shape[3])

            batch_f = batch_f_asarray
            ## I obtain a vector that is 20 x token number (which is padded in line 138 with empty vectors) and number of features x number of one hot encode
            ## For a case where the number of features are 7, I got something like 20, num_tokens, (12x7) = 20, n, 84

        else:
            batch_f = np.zeros((batch_x.shape[0:2]), dtype='int32')
            # batch_f = dense_to_one_hot(batch_f_asarray, ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE, nlevels=2)
            # batch_f_padded, _ = pad_sequences(batch_f_asarray, [0])
            # batch_f_asarray = np.asarray(batch_f_padded).reshape(max_iter, 1, ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE)

        # batch_f = batch_f.reshape(batch_f.shape[0], batch_f.shape[1] * batch_f.shape[2])
        if self.y is not None:
            batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend)
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0])
        batch_l = batches[1]

        return batch_x, batch_c, batch_f, batch_a, batch_l, batch_y
