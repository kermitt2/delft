import numpy as np
from delft.utilities.Utilities import truncate_batch_values, len_until_first_pad
from delft.utilities.numpy import shuffle_triple_with_view

import tensorflow.keras as keras
from delft.sequenceLabelling.preprocess import to_vector_single, to_casing_single, to_vector_simple_with_elmo, \
    Preprocessor, BERTPreprocessor
from delft.utilities.Tokenizer import tokenizeAndFilterSimple


class BaseGenerator(keras.utils.Sequence):
    """
    Abstract class for data generator.

    Generate batch of data to feed sequence labeling model, both for training and prediction.
    
    This generator is for input based on word embeddings. We keep embeddings application outside the 
    model to make it considerably more compact and avoid duplication of embeddings layers.
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor: Preprocessor = None,
                bert_preprocessor: BERTPreprocessor = None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize: bool =False,
                shuffle: bool =True,
                features=None,
                output_input_offsets: bool=False,
                use_chain_crf: bool =False):
        # self.x and self.y are shuffled view of self.original_x and self.original_y
        self.original_x = self.x = x
        self.original_y = self.y = y
        # features here are optional additional features provided in the case of GROBID input for instance
        self.original_features = self.features = features
        self.preprocessor = preprocessor
        self.bert_preprocessor = bert_preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        self.output_input_offsets = output_input_offsets
        self.use_chain_crf = use_chain_crf

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

    @property
    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        raise NotImplementedError("Subclasses should implement this")

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

    @property
    def __data_generation(self, index):
        '''
        Generates data containing batch_size samples
        '''
        raise NotImplementedError("Subclasses should implement this")


class DataGenerator(BaseGenerator):
    """    
    This generator is for input based on word embeddings. We keep embeddings application outside the 
    model to make it considerably more compact and avoid duplication of embeddings layers.
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                bert_preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None,
                output_input_offsets=False,
                use_chain_crf=False):

        super().__init__(x, y, 
                        batch_size=batch_size, 
                        preprocessor=preprocessor,
                        bert_preprocessor=bert_preprocessor, 
                        char_embed_size=char_embed_size, 
                        embeddings=embeddings, 
                        max_sequence_length=max_sequence_length, 
                        tokenize=tokenize, 
                        shuffle=shuffle, 
                        features=features,
                        output_input_offsets=output_input_offsets,
                        use_chain_crf=use_chain_crf)
        self.on_epoch_end()

    def __getitem__(self, index):
        '''
        Generate one batch of data, batch_l always last input, so that it can be used easily by the training scorer
        '''
        batch_x, batch_c, batch_f, batch_a, batch_l, batch_y = self.__data_generation(index)
        if self.preprocessor.return_casing:
            return [batch_x, batch_c, batch_a, batch_l], batch_y
        elif self.preprocessor.return_features:
            return [batch_x, batch_c, batch_f, batch_l], batch_y
        else:
            return [batch_x, batch_c, batch_l], batch_y

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

        # prevent sequence of length 1 alone in a batch (this causes an error in the Chain CRF layer)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        # generate data
        if self.embeddings and self.embeddings.use_ELMo:
            batch_x = to_vector_simple_with_elmo(x_tokenized, self.embeddings, max_length_x, extend=extend)
        else:
            batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
            for i in range(0, max_iter):
                batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x)

        # store tag embeddings
        batch_y = None
        if self.y is not None:
            # note: tags are always already "tokenized" by input token
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            max_length_y = max((len(y_row) for y_row in batch_y))

            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                # truncation of sequence at max_sequence_length
                batch_y = np.asarray(truncate_batch_values(batch_y, self.max_sequence_length), dtype=object)

        batch_f = np.zeros((batch_x.shape[0:2]), dtype=np.int32)
        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            if self.max_sequence_length and max_length_f > self.max_sequence_length:
                max_length_f = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                sub_f = truncate_batch_values(sub_f, self.max_sequence_length)
            batch_f = self.preprocessor.transform_features(sub_f, extend=extend)
        
        batch_a = np.zeros((max_iter, max_length_x), dtype=np.int32)
        if self.preprocessor.return_casing:
            for i in range(0, max_iter):
                batch_a[i] = to_casing_single(x_tokenized[i], max_length_x) 

        if self.y is not None:
            if self.use_chain_crf:
                batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend, label_indices=False)
            else:
                batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend, label_indices=True)
            batch_y = np.asarray(truncate_batch_values(batch_y, max_length_x), dtype=np.int32)
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0], dtype=np.int32)
        batch_l = batches[1]

        return batch_x, batch_c, batch_f, batch_a, batch_l, batch_y


class DataGeneratorTransformers(BaseGenerator):
    """
    Generate batch of data to feed sequence labeling model, both for training and prediction.
    
    This generator is for input based on transformer embeddings. We keep embeddings application 
    outside the model so that we can serialize the model more easily.  
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                bert_preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None,
                output_input_offsets=False,
                use_chain_crf=False):

        super().__init__(x, y, 
                        batch_size=batch_size, 
                        preprocessor=preprocessor, 
                        bert_preprocessor=bert_preprocessor, 
                        char_embed_size=char_embed_size, 
                        embeddings=embeddings, 
                        max_sequence_length=max_sequence_length, 
                        tokenize=tokenize, 
                        shuffle=shuffle, 
                        features=features,
                        output_input_offsets=output_input_offsets,
                        use_chain_crf=use_chain_crf)

        if self.bert_preprocessor.empty_features_vector is None:
            self.bert_preprocessor.empty_features_vector = self.preprocessor.empty_features_vector()

        self.on_epoch_end()

    def __getitem__(self, index):
        '''
        Generate one batch of data. These data are the input of the models but can also be used by the training scorer
        '''
        batch_x, batch_x_types, batch_x_masks, batch_c, batch_f, batch_l, batch_input_offsets, batch_y = self.__data_generation(index)

        # careful with the order of data arrays, double-check the models input as defined in models.py before
        # modifying this

        return_data = [batch_x]

        if self.preprocessor.return_chars:  
            return_data += [batch_c]

        if self.preprocessor.return_features:  
            return_data += [batch_f]

        return_data += [batch_x_types]

        return_data += [batch_x_masks]

        if self.output_input_offsets:
            # always last input, used when prediction are done to be able to restore the correct labeled sequence
            # this is never routed directly to a model input
            return_data += [batch_input_offsets]

        return return_data, batch_y


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
            x_tokenized = truncate_batch_values(x_tokenized, self.max_sequence_length)

        # generate data
        batch_y = None
        
        # tag embeddings
        if self.y is not None:
            # note: tags are always already "tokenized" by input token
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            max_length_y = max((len(y_row) for y_row in batch_y))

            if self.max_sequence_length and max_length_y > self.max_sequence_length:
                # truncation of sequence at max_sequence_length
                 batch_y = np.asarray(truncate_batch_values(batch_y, self.max_sequence_length), dtype=object)

        # features
        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            if self.max_sequence_length and max_length_f > self.max_sequence_length:
                max_length_f = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                sub_f = truncate_batch_values(sub_f, self.max_sequence_length)
            sub_f = self.preprocessor.transform_features(sub_f)
        else:
            sub_f = None

        # chars and length
        batches = self.preprocessor.transform(x_tokenized)
        batch_c = batches[0]
        batch_l = batches[1]

        # to have input as sentence piece token index for transformer layer
        input_ids, token_type_ids, attention_mask, input_chars, input_features, input_labels, input_offsets = self.bert_preprocessor.tokenize_and_align_features_and_labels(
                                                                        x_tokenized, 
                                                                        batch_c,
                                                                        sub_f,
                                                                        batch_y,
                                                                        maxlen=self.max_sequence_length)

        # truncate the batch input vectors for the max length in batch after sub-tokenization
        max_length_x = max((len_until_first_pad(tokens, 0) for tokens in input_ids))

        batch_x = np.asarray(truncate_batch_values(input_ids, max_length_x), dtype=np.int32)
        batch_x_types = np.asarray(truncate_batch_values(token_type_ids, max_length_x), dtype=np.int32)
        batch_x_masks = np.asarray(truncate_batch_values(attention_mask, max_length_x), dtype=np.int32)
        batch_c = np.asarray(truncate_batch_values(input_chars, max_length_x), dtype=np.int32)
        batch_input_offsets = np.asarray(truncate_batch_values(input_offsets, max_length_x), dtype=object)

        if self.y is not None:
            __, batch_y = self.preprocessor.transform(x_tokenized, input_labels, label_indices=True)
            batch_y = np.asarray(truncate_batch_values(batch_y, max_length_x), dtype=np.int32)

        if self.preprocessor.return_features:
            batch_f = np.asarray(truncate_batch_values(input_features, max_length_x), dtype=np.int32)
        else:    
            batch_f = np.zeros((batch_x.shape[0:2]), dtype=np.int32)            

        return batch_x, batch_x_types, batch_x_masks, batch_c, batch_f, batch_l, batch_input_offsets, batch_y

