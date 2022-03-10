import itertools
import json
import logging
import re
from typing import List, Iterable, Set

import numpy as np

from delft.sequenceLabelling.config import ModelConfig

LOGGER = logging.getLogger(__name__)

from sklearn.base import BaseEstimator, TransformerMixin

UNK = '<UNK>'
PAD = '<PAD>'

case_index = {'<PAD>': 0, 'numeric': 1, 'allLower': 2, 'allUpper': 3, 'initialUpper': 4, 'other': 5,
              'mainly_numeric': 6, 'contains_digit': 7}

def calculate_cardinality(feature_vector, indices=None):
    """
    Calculate cardinality of each features

    :param feature_vector: three dimensional vector with features
    :param indices: list of indices of the features to be extracted
    :return: a map where each key is the index of the feature and the value is a map feature_value,
    value_index.
    For example
     [(0, {'feature1': 1, 'feature2': 2})]

     indicates that the feature is at index 0 and has two values, features1 and features2 with two
     unique index.

     NOTE: the features are indexed from 1 to n + 1. The 0 value is reserved as padding
    """
    columns_length = []
    index = 0
    if not len(feature_vector) > 0:
        return []

    for index_column in range(index, len(feature_vector[0][0])):
        if indices and index_column not in indices:
            index += 1
            continue

        values = set()
        for index_document in range(0, len(feature_vector)):
            for index_row in range(0, len(feature_vector[index_document])):
                value = feature_vector[index_document][index_row][index_column]
                if value != " ":
                    values.add(value)

        values = sorted(values)
        values_cardinality = len(values)

        values_list = list(values)
        values_to_int = {}
        for val_num in range(0, values_cardinality):
            # We reserve the 0 for the unseen features so the indexes will go from 1 to cardinality + 1
            values_to_int[values_list[val_num]] = val_num + 1

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
    max_index_value = features_max_vector_size * len(columns_index) + 1

    index_list = [ind[0] for ind in columns_index if ind[0] >= 0]
    val_to_int_map = {
        value[0]: {val_features: idx_features + (index * features_max_vector_size) for val_features, idx_features in
                   value[1].items()} for index, value in enumerate(columns_index)}

    return index_list, val_to_int_map


def reduce_features_to_indexes(feature_vector, features_max_vector_size, indices=None):
    cardinality = calculate_cardinality(feature_vector, indices=indices)
    index_list, map_to_integers = cardinality_to_index_map(cardinality, features_max_vector_size)

    return index_list, map_to_integers


def reduce_features_vector(feature_vector, features_max_vector_size):
    """
    Reduce the features vector.
    First it calculates the cardinality for each value that each feature can assume, then
    removes features with cardinality above features_max_vector_size.
    Finally it assign indices values for each features, assuming 0 as padding (feature value out of bound or invalid),
    and values from 1 to features_max_vector_size * number of features.

    :param feature_vector: feature vector to be reduced
    :param features_max_vector_size maximum size of the one-hot-encoded values
    :return:
    """

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
            [feature_vector[index_row][index_column] for index, index_column in enumerate(index_list)])

    return reduced_features_vector


def get_map_to_index(X, features_indices, features_vector_size):
    pass


def to_dict(value_list_batch: List[list], feature_indices: Set[int] = None,
            features_vector_size: int = ModelConfig.DEFAULT_FEATURES_VOCABULARY_SIZE):
    if not feature_indices:
        matrix = reduce_features_vector(value_list_batch, features_vector_size)

        return [{index: value for index, value in enumerate(value_list)} for value_list in matrix]
    else:
        return [{index: value for index, value in enumerate(value_list) if index in feature_indices} for value_list
                in value_list_batch]


class FeaturesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_indices: Iterable[int] = None,
                 features_vocabulary_size=ModelConfig.DEFAULT_FEATURES_VOCABULARY_SIZE,
                 features_map_to_index=None):
        # feature_indices_set = None
        if features_map_to_index is None:
            features_map_to_index = {}
        self.features_vocabulary_size = features_vocabulary_size
        self.features_indices = features_indices

        # List of mappings to integers (corresponding to each feature column) of features values
        # This value could be provided (model has been loaded) or not (first-time-training)
        self.features_map_to_index = features_map_to_index

    def fit(self, X):
        if not self.features_indices:
            indexes, mapping = reduce_features_to_indexes(X, self.features_vocabulary_size)
        else:
            indexes, mapping = reduce_features_to_indexes(X, self.features_vocabulary_size,
                                                          indices=self.features_indices)

        self.features_map_to_index = mapping
        self.features_indices = indexes
        return self

    def transform(self, X, extend=False):
        """
        Transform the features into a vector, return the vector and the extracted number of features

        :param extend: when set to true it's adding an additional empty feature list in the sequence.
        """
        features_vector = [[[self.features_map_to_index[index][
                                 value] if index in self.features_map_to_index and value in self.features_map_to_index[
            index] else 0
                             for index, value in enumerate(value_list) if index in self.features_indices] for
                            value_list in document] for document in X]

        features_count = len(self.features_indices)

        if extend:
            for out in features_vector:
                out.append([0] * features_count)

        features_vector_padded, _ = pad_sequences(features_vector, [0] * features_count)
        output = np.asarray(features_vector_padded)

        return output

    def empty_features_vector(self) -> Iterable[int]:
        features_count = len(self.features_indices)
        return [0] * features_count


class BERTPreprocessor(object):
    """
    Generic BERT preprocessor for a sequence labelling data set.
    Input are pre-tokenized texts, possibly with features and labels to re-align with the sub-tokenization. 
    Rely on transformers library tokenizer
    """

    def __init__(self, tokenizer, empty_features_vector=None, empty_char_vector=None):
        self.tokenizer = tokenizer
        self.empty_features_vector = empty_features_vector
        self.empty_char_vector = empty_char_vector


    def tokenize_and_align_features_and_labels(self, texts, chars, text_features, text_labels, maxlen=512):
        """
        Training/evaluation usage with features: sub-tokenize+convert to ids/mask/segments input texts, realign labels
        and features given new tokens introduced by the wordpiece sub-tokenizer.
        texts is a list of texts already pre-tokenized
        """
        target_ids = []
        target_type_ids = []
        target_attention_mask = []
        input_tokens = []
        target_chars = []

        target_features = None
        if text_features is not None:
            target_features = []
        
        target_labels = None
        if text_labels is not None:
            target_labels = [] 
        
        for i, text in enumerate(texts):
            
            local_chars = chars[i]

            features = None
            if text_features is not None:
                features = text_features[i]
            
            label_list = None
            if text_labels is not None:
                label_list = text_labels[i]

            input_ids, token_type_ids, attention_mask, chars_block, feature_blocks, target_tags, tokens = self.convert_single_text(text, 
                                                                                                                    local_chars, 
                                                                                                                    features, 
                                                                                                                    label_list, 
                                                                                                                    maxlen)
            target_ids.append(input_ids)
            target_type_ids.append(token_type_ids)
            target_attention_mask.append(attention_mask)
            input_tokens.append(tokens)
            target_chars.append(chars_block)

            if target_features is not None:
                target_features.append(feature_blocks)
            
            if target_labels is not None:
                target_labels.append(target_tags)                

        return target_ids, target_type_ids, target_attention_mask, target_chars, target_features, target_labels, input_tokens


    def convert_single_text(self, text_tokens, chars_tokens, features_tokens, label_tokens, max_seq_length):
        """
        Converts a single sequence input into a single transformer input format using generic tokenizer
        of the transformers library, align other channel input to the new sub-tokenization
        """
        if label_tokens is None:
            # we create a dummy label list to facilitate
            label_tokens = []
            while len(label_tokens) < len(text_tokens):
                label_tokens.append(0)

        if features_tokens is None:
            # we create a dummy feature list to facilitate
            features_tokens = []
            while len(features_tokens) < len(text_tokens):
                features_tokens.append(self.empty_features_vector)

        if chars_tokens is None:
            # we create a dummy feature list to facilitate
            chars_tokens = []
            while len(chars_tokens) < len(text_tokens):
                chars_tokens.append(self.empty_char_vector)
        
        # sub-tokenization
        encoded_result = self.tokenizer(text_tokens, add_special_tokens=True, is_split_into_words=True,
            max_length=max_seq_length, truncation=True, return_offsets_mapping=True)

        input_ids = encoded_result.input_ids
        offsets = encoded_result.offset_mapping
        if "token_type_ids" in encoded_result:
            token_type_ids = encoded_result.token_type_ids
        else:
            token_type_ids = [0] * len(input_ids)
        attention_mask = encoded_result.attention_mask
        label_ids = []
        chars_blocks = []
        feature_blocks = []

        # trick to support sentence piece tokenizer like GPT2, roBERTa, CamemBERT, etc. which encode prefixed 
        # spaces in the tokens (the encoding symbol for this space varies from one model to another)
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        new_offsets = []
        for i in range(0, len(input_ids)):
            if len(self.tokenizer.decode(input_ids[i])) != 0:
                # if a decoded token has a length of 0, it is typically a space added for sentence piece/camembert/GPT2 
                # which happens to be then sometimes a single token for unknown reason when with is_split_into_words=True
                # we need to skip this but also remove it from attention_mask, token_type_ids and offsets to stay 
                # in sync
                new_input_ids.append(input_ids[i])
                new_attention_mask.append(attention_mask[i])
                new_token_type_ids.append(token_type_ids[i])
                new_offsets.append(offsets[i])
        input_ids = new_input_ids
        attention_mask = new_attention_mask
        token_type_ids = new_token_type_ids
        offsets = new_offsets

        word_idx = -1
        for i, offset in enumerate(offsets):
            if offset[0] == 0 and offset[1] == 0:
                # this is a special token
                label_ids.append("<PAD>")
                chars_blocks.append(self.empty_char_vector)
                feature_blocks.append(self.empty_features_vector)
            else:
                if offset[0] == 0:
                    word_idx += 1

                    # new token
                    label_ids.append(label_tokens[word_idx])
                    feature_blocks.append(features_tokens[word_idx])
                    chars_blocks.append(chars_tokens[word_idx])
                else:
                    # propagate the data to the new sub-token or 
                    # dummy/empty input for sub-tokens
                    label_ids.append("<PAD>")
                    chars_blocks.append(self.empty_char_vector)
                    # 2 possibilities, either empty features for sub-tokens or repeating the 
                    # feature vector of the prefix sub-token 
                    #feature_blocks.append(self.empty_features_vector)
                    feature_blocks.append(features_tokens[word_idx])

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            token_type_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            label_ids.append("<PAD>")
            chars_blocks.append(self.empty_char_vector)
            feature_blocks.append(self.empty_features_vector)

        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(chars_blocks) == max_seq_length
        assert len(feature_blocks) == max_seq_length

        return input_ids, token_type_ids, attention_mask, chars_blocks, feature_blocks, label_ids, offsets


    def convert_single_text_bert(self, text_tokens, chars_tokens, features_tokens, label_tokens, max_seq_length):
        """
        Converts a single sequence input into a single BERT input format and align other channel input to this 
        new sub-tokenization

        The original BERT implementation works as follow:

        input:
            tokens: [Jim,Henson,was,a,puppeteer]
            labels: [I-PER,I-PER,O,O,O]
        The BERT tokenization will modify sentence and labels as follow:
            tokens: [Jim,Hen,##son,was,a,puppet,##eer]
            labels: [I-PER,I-PER,X,O,O,O,X]

        Here, we don't introduce the X label for added sub-tokens and simply extend the previous label
        and produce:
            tokens: [Jim,Hen,##son,was,a,puppet,##eer]
            labels: [I-PER,I-PER,I-PER,O,O,O,O]

        Notes: input and output labels are text labels, they need to be changed into indices after
        text conversion.
        """

        tokens = []
        # the following is to keep track of additional tokens added by BERT tokenizer,
        # only some of them has a prefix ## that allows to identify them downstream in the process
        tokens_marked = []
        labels = []
        features = []
        chars = []

        if label_tokens is None:
            # we create a dummy label list to facilitate
            label_tokens = []
            while len(label_tokens) < len(text_tokens):
                label_tokens.append(0)

        if features_tokens is None:
            # we create a dummy feature list to facilitate
            features_tokens = []
            while len(features_tokens) < len(text_tokens):
                features_tokens.append(self.empty_features_vector)

        if chars_tokens is None:
            # we create a dummy feature list to facilitate
            chars_tokens = []
            while len(chars_tokens) < len(text_tokens):
                chars_tokens.append(self.empty_char_vector)

        for text_token, label_token, chars_token, features_token in zip(text_tokens, label_tokens, chars_tokens, features_tokens):
            text_sub_tokens = self.tokenizer.tokenize(text_token, add_special_tokens=False)
            
            # we mark added sub-tokens with the "##" prefix in order to restore token back correctly,
            # otherwise the BERT tokenizer do not mark them all with this prefix 
            # (note: to be checked if it's the same with the non-original BERT tokenizer)
            text_sub_tokens_marked = self.tokenizer.tokenize(text_token, add_special_tokens=False)
            for i in range(len(text_sub_tokens_marked)):
                if i == 0:
                    continue
                tok = text_sub_tokens_marked[i]
                if not tok.startswith("##"):
                    text_sub_tokens_marked[i] = "##" + tok
            
            label_sub_tokens = [label_token] + [label_token] * (len(text_sub_tokens) - 1)
            chars_sub_tokens = [chars_token] + [chars_token] * (len(text_sub_tokens) - 1)
            feature_sub_tokens = [features_token] + [features_token] * (len(text_sub_tokens) - 1)

            tokens.extend(text_sub_tokens)
            tokens_marked.extend(text_sub_tokens_marked)
            labels.extend(label_sub_tokens)
            features.extend(feature_sub_tokens)
            chars.extend(chars_sub_tokens)

        if len(tokens) >= max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            tokens_marked = tokens_marked[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            features = features[0:(max_seq_length - 2)]
            chars = chars[0:(max_seq_length - 2)]

        input_tokens = []
        input_tokens_marked = []
        segment_ids = []
        label_ids = []
        chars_blocks = []
        feature_blocks = []

        # The convention in BERT is:
        # (a) For sequence pairs:
        #   tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #   type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #   tokens:   [CLS] the dog is hairy . [SEP]
        #   type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first sequence
        # or the second sequence.

        input_tokens.append(self.tokenizer.cls_token)
        input_tokens_marked.append(self.tokenizer.cls_token)
        segment_ids.append(0)
        label_ids.append("<PAD>")
        chars_blocks.append(self.empty_char_vector)
        feature_blocks.append(self.empty_features_vector)

        for i, token in enumerate(tokens):
            input_tokens.append(token)
            segment_ids.append(0)
            label_ids.append(labels[i])
            feature_blocks.append(features[i])
            chars_blocks.append(chars[i])

        for token in tokens_marked:
            input_tokens_marked.append(token)
 
        input_tokens.append(self.tokenizer.sep_token)
        input_tokens_marked.append(self.tokenizer.sep_token)
        segment_ids.append(0)
        label_ids.append("<PAD>")
        chars_blocks.append(self.empty_char_vector)
        feature_blocks.append(self.empty_features_vector)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            input_mask.append(self.tokenizer.pad_token_id)
            segment_ids.append(0)
            label_ids.append("<PAD>")
            chars_blocks.append(self.empty_char_vector)
            feature_blocks.append(self.empty_features_vector)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(chars_blocks) == max_seq_length
        assert len(feature_blocks) == max_seq_length

        return input_ids, input_mask, segment_ids, chars_blocks, feature_blocks, label_ids, input_tokens_marked


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 padding=True,
                 return_lengths=False,
                 return_word_embeddings=False,
                 return_casing=False,
                 return_features=False,
                 return_chars=False,
                 return_bert_embeddings=False,
                 max_char_length=30,
                 feature_preprocessor: FeaturesPreprocessor = None,
                 ):

        self.padding = padding
        self.return_lengths = return_lengths
        self.return_word_embeddings = return_word_embeddings
        self.return_casing = return_casing
        self.return_features = return_features
        self.return_chars = return_chars
        self.return_bert_embeddings = return_bert_embeddings
        self.vocab_char = None
        self.vocab_tag = None
        self.vocab_case = [k for k, v in case_index.items()]
        self.max_char_length = max_char_length
        self.feature_preprocessor = feature_preprocessor
        self.indice_tag = None

    def fit(self, X, y):
        chars = {PAD: 0, UNK: 1}
        tags = {PAD: 0}

        #if self.return_chars:

        temp_chars = {
            c
            for w in set(itertools.chain(*X))
            for c in w
        }

        sorted_chars = sorted(temp_chars)
        sorted_chars_dict = {
            c: idx + 2
            for idx, c in enumerate(sorted_chars)
        }
        chars = {**chars, **sorted_chars_dict}

        temp_tags = set(itertools.chain(*y))
        sorted_tags = sorted(temp_tags)
        sorted_tags_dict = {
            tag: idx + 1
            for idx, tag in enumerate(sorted_tags)
        }
        tags = {**tags, **sorted_tags_dict}

        self.vocab_char = chars
        self.vocab_tag = tags
        self.indice_tag = {i: t for t, i in self.vocab_tag.items()}

        return self

    def transform(self, X, y=None, extend=False, label_indices=False):
        """
        transforms input into sequence
        the optional boolean `extend` indicates that we need to avoid sequence of length 1 alone in a batch 
        (which would cause an error with tf)

        Args:
            X: list of list of word tokens
            y: list of list of tags

        Returns:
            numpy array: sentences with char sequences and length 
            numpy array: sequence of tags, either one hot encoded (default) or as indices

        if label_indices parameter is true, we encode tags with index integer, otherwise output hot one encoded tags
        """
        chars = []
        lengths = []
        for sent in X:
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                #if self.return_chars:
                char_ids.append(self.get_char_ids(w))
                if extend:
                    char_ids.append([])

            #if self.return_chars:
            chars.append(char_ids)

        if y is not None:
            pad_index = self.vocab_tag[PAD]
            y = [[self.vocab_tag.get(t, pad_index) for t in sent] for sent in y]
            if extend:
                y[0].append(pad_index)

        if self.padding:
            sents, y = self.pad_sequence(chars, y, label_indices=label_indices)
        else:
            sents = [chars]

        # lengths
        #if self.return_lengths:
        lengths = np.asarray(lengths, dtype=np.int32)
        lengths = lengths.reshape((lengths.shape[0], 1))
        sents.append(lengths)

        return (sents, y) if y is not None else sents

    def fit_features(self, features_batch):
        if self.feature_preprocessor is None:
            return

        return self.feature_preprocessor.fit(features_batch)

    def transform_features(self, features_batch, extend=False):
        return self.feature_preprocessor.transform(features_batch, extend=extend)

    def inverse_transform(self, y):
        """
        send back original label string from label index
        """
        if self.indice_tag == None:
            self.indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [self.indice_tag[y_] for y_ in y]

    def get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def pad_sequence(self, char_ids, labels=None, label_indices=False):
        '''
        pad char and label sequences

        Relatively to labels, if label_indices is True, we encode labels with integer,
        otherwise with hot one encoding
        '''
        labels_final = None
        if labels:
            labels_padded, _ = pad_sequences(labels, 0)
            labels_final = np.asarray(labels_padded)
            if not label_indices:
                labels_final = dense_to_one_hot(labels_final, len(self.vocab_tag), nlevels=2)

        #if self.return_chars:
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2, max_char_length=self.max_char_length)
        char_ids = np.asarray(char_ids)
        return [char_ids], labels_final
        #else:
        #    return labels_final

    def empty_features_vector(self) -> Iterable[int]:
        if self.feature_preprocessor is not None:
            return self.feature_preprocessor.empty_features_vector()
        else:
            return None

    def empty_char_vector(self) -> Iterable[int]:
        return [0] * self.max_char_length

    def save(self, file_path):
        variables = vars(self)
        output_dict = {}
        for var in variables.keys():
            if var == 'feature_preprocessor' and variables['feature_preprocessor'] is not None:
                output_dict[var] = variables[var].__dict__
            else:
                output_dict[var] = variables[var]

        with open(file_path, 'w') as fp:
            json.dump(output_dict, fp, sort_keys=False, indent=4)

    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            variables = json.load(f)
            self = cls()
            for key, val in variables.items():
                if key == 'feature_preprocessor' and val is not None:
                    preprocessor = FeaturesPreprocessor()
                    preprocessor.__dict__.update(val)
                    if 'features_map_to_index' in preprocessor.__dict__:
                        preprocessor.__dict__['features_map_to_index'] = {int(key): val for key, val in
                                                                          preprocessor.__dict__[
                                                                              'features_map_to_index'].items()}
                    setattr(self, key, preprocessor)
                else:
                    setattr(self, key, val)

        # fix typing issue for integer keys which become string :/
        self.indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return self


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
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))


def prepare_preprocessor(X, y, model_config, features: np.array = None):
    """
    Prepare the preprocessor. If features are passed, configure the feature preprocessor

    From: https://github.com/elifesciences/sciencebeam-trainer-delft/blob/5ceb89bdb9ae56c7f60d68b3aeb7e04dc34cd2be/sciencebeam_trainer_delft/sequence_labelling/preprocess.py#L81
    """
    feature_preprocessor = None
    if features is not None and str.endswith(model_config.architecture, "FEATURES"):
        feature_preprocessor = FeaturesPreprocessor(
            features_indices=model_config.features_indices,
            features_vocabulary_size=model_config.features_vocabulary_size
        )

    preprocessor = Preprocessor(
        max_char_length=model_config.max_char_length,
        feature_preprocessor=feature_preprocessor
    )
    preprocessor.fit(X, y)

    # Compute features and store information in the model config
    if feature_preprocessor is not None:
        preprocessor.fit_features(features)
        model_config.features_indices = preprocessor.feature_preprocessor.features_indices
        model_config.features_map_to_index = preprocessor.feature_preprocessor.features_map_to_index

    return preprocessor


def to_vector_single(tokens, embeddings, maxlen, lowercase=False, num_norm=True):
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
        x[i, :] = embeddings.get_word_vector(word).astype('float32')

    return x

def to_vector_elmo(tokens, embeddings, maxlen, lowercase=False, num_norm=False, extend=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on ELMo contextualized embeddings
    """
    subtokens = get_subtokens(tokens, maxlen, extend, lowercase)
    return embeddings.get_sentence_vector_only_ELMo(subtokens)


def to_vector_simple_with_elmo(tokens, embeddings, maxlen, lowercase=False, num_norm=False, extend=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the concatenation of the provided static embeddings and 
    the ELMo contextualized embeddings, introducing <PAD> and <UNK> 
    padding token vector when appropriate
    """
    subtokens = get_subtokens(tokens, maxlen, extend, lowercase)
    return embeddings.get_sentence_vector_with_ELMo(subtokens)


def get_subtokens(tokens, maxlen, extend=False, lowercase=False):
    """
    Extract the token list and eventually lowercase or truncate longest sequences

    :param tokens: input tokens
    :param maxlen: maximum length for each sub_token
    :param extend: when set to true, sub tokens will be padded with an additional element
    :param lowercase: when set to true the sub_tokens will be lowercased
    :return:
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
    return subtokens


def to_casing_single(tokens, maxlen):
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
