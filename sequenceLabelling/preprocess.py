import itertools
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

# this is loosely based on https://github.com/Hironsan/anago/blob/master/anago/preprocess.py

UNK = '<UNK>'
PAD = '<PAD>'

class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 lowercase=False,
                 num_norm=True,
                 use_char_feature=True,
                 padding=True,
                 return_lengths=True):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.use_char_feature = use_char_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.vocab_char = None
        self.vocab_tag  = None

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

    def transform(self, X, y=None):
        """
        transforms input into sequence

        Args:
            X: list of list of word tokens
            y: list of list of tags

        Returns:
            numpy array: sentences with char and length sequences
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

            if self.use_char_feature:
                chars.append(char_ids)

        if y is not None:
            y = [[self.vocab_tag[t] for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(chars, y)
        else:
            sents = [chars]

        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def pad_sequence(self, char_ids, labels=None):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = dense_to_one_hot(labels, len(self.vocab_tag), nlevels=2)

        if self.use_char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            char_ids = np.asarray(char_ids)
            return [char_ids], labels
        else:
            return labels

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


def pad_sequences(sequences, pad_tok, nlevels=1):
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
        max_length_word = max(len(max(seq, key=len)) for seq in sequences)
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


def prepare_preprocessor(X, y, use_char=True):
    p = WordPreprocessor()
    p.fit(X, y)

    return p

def to_vector_single(tokens, embeddings, maxlen=300, embed_size=300, lowercase=False, num_norm=True):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    window = tokens[-maxlen:]
    
    # TBD: use better initializers (uniform, etc.) 
    x = np.zeros((maxlen, embed_size), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        x[i,:] = get_word_vector(word, embeddings, embed_size, lowercase, num_norm).astype('float32')

    return x

def get_word_vector(word, embeddings, embed_size, lowercase=False, num_norm=True):
    if lowercase:
        word = _lower(word)
    if num_norm:
        word = _normalize_num(word)
    if word in embeddings:
        return embeddings[word]
    else:
        # for unknown word, we use a vector filled with 0.0
        return np.zeros((embed_size,), dtype=np.float32)
        # alternatively, initialize with random negative values
        #return np.random.uniform(low=-0.5, high=0.0, size=(embeddings.shape[1],))
        # alternatively use fasttext OOV ngram possibilities (if ngram available)

def _lower(word):
    return word.lower() 

def _normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
