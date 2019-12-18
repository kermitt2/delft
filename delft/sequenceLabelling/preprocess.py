import itertools
import logging
import re
from functools import partial
from typing import List, Iterable, Set

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from delft.sequenceLabelling.config import ModelConfig

LOGGER = logging.getLogger(__name__)

np.random.seed(7)
#from tensorflow import set_random_seed
#set_random_seed(7)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import tensorflow as tf
tf.set_random_seed(7)

# this is derived from https://github.com/Hironsan/anago/blob/master/anago/preprocess.py

UNK = '<UNK>'
PAD = '<PAD>'

case_index = {'<PAD>': 0, 'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7}


def calculate_cardinality(feature_vector):
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    for index_column in range(index, len(feature_vector[0][0])):
        values = set()
        for index_document in range(0, len(feature_vector)):
            for index_row in range(0, len(feature_vector[index_document])):
                value = feature_vector[index_document][index_row][index_column]
                if value != " ":
                    values.add(value)

        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            values_to_int[values_list[val_num]] = val_num

        columns_length.append((index, values_to_int))
        index += 1

    return columns_length

def cardinality_to_index_map(columns_length, features_max_vector_size):
    # Filter out the columns that are not fitting
    columns_index = []
    for index, column_content_cardinality in columns_length:
        if len(column_content_cardinality) <= features_max_vector_size:
            columns_index.append((index, column_content_cardinality))
    # print(columns_index)
    index_list = [ind[0] for ind in columns_index if ind[0]]
    val_to_int_list = {value[0]: value[1] for value in columns_index}

    return index_list, val_to_int_list


def reduce_features_to_indexes(feature_vector, features_max_vector_size):
    cardinality = calculate_cardinality(feature_vector)
    index_list, map_to_integers = cardinality_to_index_map(cardinality, features_max_vector_size)

    return index_list, map_to_integers


def reduce_features_vector_old(feature_vector, features_max_vector_size):
    '''
    Reduce the features vector.
    First it calculates cardinalities for each value that each feature can assume, then
    removes features with cardinality above features_max_vector_size.

    :param feature_vector: feature vector to be reduced
    :param features_max_vector_size maximum size of the one-hot-encoded values
    :return:
    '''

    # Compute frequencies for each column
    columns_length = calculate_cardinality(feature_vector)

    # print("Column: " + str(index_column) + " Len:  " + str(len(values)))
    index_list, val_to_int_list = cardinality_to_index_map(columns_length, features_max_vector_size)

    # create a reduced vector feature value
    reduced_features_vector = []
    for index_document in range(0, len(feature_vector)):
        # print(len(f_train[index_document]))
        for index_row in range(0, len(feature_vector[index_document])):
            reduced_features_vector.append([val_to_int_list[index][
                                            feature_vector[index_document][index_row][index_column]] for
                                        index, index_column in enumerate(index_list)])

    return reduced_features_vector


def reduce_features_vector(feature_vector, features_max_vector_size):
    '''
    Reduce the features vector.
    First it calculates cardinalities for each value that each feature can assume, then
    removes features with cardinality above features_max_vector_size.

    :param feature_vector: feature vector to be reduced
    :param features_max_vector_size maximum size of the one-hot-encoded values
    :return:
    '''

    # Compute frequencies for each column
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    for index_column in range(index, len(feature_vector[0])):
        values = set()
        for index_row in range(0, len(feature_vector)):
            value = feature_vector[index_row][index_column]
            if value != " ":
                values.add(value)

        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            values_to_int[values_list[val_num]] = val_num

        columns_length.append((index, values_to_int))
        index += 1
        # print("Column: " + str(index_column) + " Len:  " + str(len(values)))

    # Filter out the columns that are not fitting
    columns_index = []
    for index, column_content_cardinality in columns_length:
        if len(column_content_cardinality) <= features_max_vector_size:
            columns_index.append((index, column_content_cardinality))
    # print(columns_index)
    index_list = [ind[0] for ind in columns_index if ind[0]]

    # Assign indexes to each feature value
    reduced_features_vector = []
    for index_row in range(0, len(feature_vector)):
        reduced_features_vector.append(
            [feature_vector[index_row][index_column]for index, index_column in enumerate(index_list)])

    return reduced_features_vector


def to_dict(value_list_batch: List[list], feature_indices: Set[int] = None,
            features_vector_size: int = ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE):

    if not feature_indices:
        matrix = reduce_features_vector(value_list_batch, features_vector_size)

        return [{index: value for index, value in enumerate(value_list)} for value_list in matrix]
    else:
        return [{index: value for index, value in enumerate(value_list) if index in feature_indices} for value_list
                in value_list_batch]


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_indices: Iterable[int] = None,
                 features_vector_size = ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE):
        # feature_indices_set = None
        self.features_max_vector_size = features_vector_size
        self.features_indices = features_indices
        self.feature_count = 0

        # List of mappings to integers (corresponding to each feature column) of features values
        self.features_map_to_index = []

    def fit(self, X):
        if not self.features_indices:
            indexes, mapping = reduce_features_to_indexes(X, self.features_max_vector_size)
            self.features_indices = indexes
            self.features_map_to_index = mapping
            self.feature_count = len(self.features_indices)

        return self

    def transform(self, X):
        output = [[[self.features_map_to_index[index][value] for index, value in enumerate(value_list) if index in self.features_indices] for
                    value_list in document] for document in X]

        return output, len(self.features_indices)

class FeaturesPreprocessor2(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices: Iterable[int] = None,
                 features_vector_size = ModelConfig.DEFAULT_FEATURES_VECTOR_SIZE):
        feature_indices_set = None
        self.features_vector_size = features_vector_size

        if feature_indices:
            feature_indices_set = set(feature_indices)
        to_dict_fn = partial(to_dict, feature_indices=feature_indices_set, features_vector_size=features_vector_size)
        self.pipeline = Pipeline(steps=[
            ('to_dict', FunctionTransformer(to_dict_fn, validate=False)),
            ('vectorize', DictVectorizer(sparse=False))
        ])

    def fit(self, X):
        flattened_features = [word_features for sentence_features in X for word_features in sentence_features]
        # LOGGER.debug('flattened_features: %s', flattened_features)
        self.pipeline.fit(flattened_features)
        return self

    def transform(self, X):
        # LOGGER.debug('transform, X: %s', X)
        return np.asarray([
            self.pipeline.transform(sentence_features)
            for sentence_features in X
        ])


class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 use_char_feature=True,
                 padding=True,
                 return_lengths=True,
                 return_casing=False,
                 return_features=False,
                 max_char_length=30,
                 feature_preprocessor: FeaturesPreprocessor = None
                 ):

        self.use_char_feature = use_char_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.return_casing = return_casing
        self.return_features = return_features
        self.vocab_char = None
        self.vocab_tag  = None
        self.vocab_case = [k for k, v in case_index.items()]
        self.max_char_length = max_char_length
        self.feature_preprocessor = feature_preprocessor

    def fit(self, X, y):
        chars = {PAD: 0, UNK: 1}
        tags  = {PAD: 0}

        for w in set(itertools.chain(*X)):
            if not self.use_char_feature:
                continue
            for c in w:
                if c not in chars:
                    chars[c] = len(chars)

        for t in itertools.chain(*y):
            if t not in tags:
                tags[t] = len(tags)

        self.vocab_char = chars
        self.vocab_tag  = tags

        return self

    def transform(self, X, y=None, extend=False):
        """
        transforms input into sequence
        the optional boolean `extend` indicates that we need to avoid sequence of length 1 alone in a batch 
        (which would cause an error with tf)

        Args:
            X: list of list of word tokens
            y: list of list of tags

        Returns:
            numpy array: sentences with char sequences, and optionally length, casing and custom features  
            numpy array: sequence of tags
        """
        chars = []
        lengths = []
        for sent in X:
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                if self.use_char_feature:
                    char_ids.append(self.get_char_ids(w))
                    if extend:
                        char_ids.append([])

            if self.use_char_feature:
                chars.append(char_ids)

        if y is not None:
            pad_index = self.vocab_tag[PAD]
            LOGGER.debug('vocab_tag: %s', self.vocab_tag)
            y = [[self.vocab_tag.get(t, pad_index) for t in sent] for sent in y]
            if extend:
                y[0].append(pad_index)

        if self.padding:
            sents, y = self.pad_sequence(chars, y)
        else:
            sents = [chars]

        # lengths
        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

    def fit_features(self, features_batch):
        return self.feature_preprocessor.fit(features_batch)

    def transform_features(self, features_batch):
        return self.feature_preprocessor.transform(features_batch)

    def inverse_transform(self, y):
        """
        send back original label string
        """
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def pad_sequence(self, char_ids, labels=None):
        if labels:
            labels_padded, _ = pad_sequences(labels, 0)
            labels_asarray = np.asarray(labels_padded)
            labels_one_hot = dense_to_one_hot(labels_asarray, len(self.vocab_tag), nlevels=2)

        if self.use_char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2, max_char_length=self.max_char_length)
            char_ids = np.asarray(char_ids)
            return [char_ids], labels_one_hot
        else:
            return labels_one_hot

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok=0, nlevels=1, max_char_length=30):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max_char_length
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length


def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """
    Convert class labels from scalars to one-hot vectors

    Args:
        labels_dense: a dense set of values
        num_classes: the number of classes in output
        nlevels: the number of levels (??)
    """
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    elif nlevels == 3:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        something_else = labels_dense.shape[2]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # component_idx = np.tile(np.tile(np.arange(num_classes), (num_length,1)), (num_labels, 1, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

def prepare_preprocessor(X, y, model_config, features: np.array = None):
    '''
    Prepare the preprocessor. If features are passed, configure the feature preprocessor

    From: https://github.com/elifesciences/sciencebeam-trainer-delft/blob/5ceb89bdb9ae56c7f60d68b3aeb7e04dc34cd2be/sciencebeam_trainer_delft/sequence_labelling/preprocess.py#L81
    '''
    feature_preprocessor = None
    if not model_config.ignore_features and features is not None:
        feature_preprocessor = FeaturesPreprocessor(
            features_indices=model_config.features_indices,
            features_vector_size=model_config.features_vector_size
        )
    preprocessor = WordPreprocessor(
        max_char_length=model_config.max_char_length,
        feature_preprocessor=feature_preprocessor
    )
    preprocessor.fit(X, y)
    if features is not None:
        preprocessor.fit_features(features)
    return preprocessor

def to_vector_single(tokens, embeddings, maxlen=300, lowercase=False, num_norm=True):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    window = tokens[-maxlen:]

    # TBD: use better initializers (uniform, etc.) 
    x = np.zeros((maxlen, embeddings.embed_size), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        if lowercase:
            word = _lower(word)
        if num_norm:
            word = _normalize_num(word)
        x[i,:] = embeddings.get_word_vector(word).astype('float32')

    return x


def to_vector_elmo(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on ELMo contextualized embeddings
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(_lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_only_ELMo(subtokens)
    """
    if use_token_dump:
        return embeddings.get_sentence_vector_ELMo_with_token_dump(tokens)
    """


def to_vector_simple_with_elmo(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False, extend=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the concatenation of the provided static embeddings and 
    the ELMo contextualized embeddings, introducing <PAD> and <UNK> 
    padding token vector when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(_lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        if extend:
            local_tokens.append(UNK)
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_with_ELMo(subtokens)


def to_vector_bert(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False, extend=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the BERT contextualized embeddings, introducing
    padding token when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(_lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        if extend:
            local_tokens.append(UNK)
        subtokens.append(local_tokens)
    vector = embeddings.get_sentence_vector_only_BERT(subtokens)
    return vector


def to_vector_simple_with_bert(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the concatenation of the provided static embeddings and 
    the BERT contextualized embeddings, introducing padding token vector 
    when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(_lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_with_BERT(subtokens)


def to_casing_single(tokens, maxlen=300):
    """
    Given a list of tokens set the casing, introducing <PAD> and <UNK> padding 
    when appropriate
    """
    window = tokens[-maxlen:]

    # TBD: use better initializers (uniform, etc.) 
    x = np.zeros((maxlen), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        x[i] = float(_casing(word))

    return x


def _casing(word):
        casing = 'other'

        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1
        digitFraction = numDigits / float(len(word))

        if word.isdigit():
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():
            casing = 'allLower'
        elif word.isupper():
            casing = 'allUpper'
        elif word[0].isupper():
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return case_index[casing]


def _lower(word):
    return word.lower()


def _normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)


