import itertools
import re
import numpy as np
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


class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 use_char_feature=True,
                 padding=True,
                 return_lengths=True, 
                 return_casing=False, 
                 return_features=False, 
                 max_char_length=30
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
            y = [[self.vocab_tag[t] for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(chars, y)
        else:
            sents = [chars]

        # optional additional information
        # lengths
        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

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
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = dense_to_one_hot(labels, len(self.vocab_tag), nlevels=2)

        if self.use_char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2, max_char_length=self.max_char_length)
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


def prepare_preprocessor(X, y, model_config):
    p = WordPreprocessor(max_char_length=model_config.max_char_length)
    p.fit(X, y)

    return p


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


def to_vector_simple_with_elmo(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
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
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_with_ELMo(subtokens)


def to_vector_bert(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
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


class InputExample(object):
    """
    A single training/test example for simple BERT sequence classification.
    """
    def __init__(self,
                 guid,
                 text,
                 label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
    """
    Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """
    A single BERT set of features of data.
    """
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids     

class NERProcessor(object):
    """
    BERT Processor for a NER data set.
    """
    def __init__(self,
                 data_dir,
                 task_name):
        self.data_dir = data_dir
        self.task_name = task_name
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}".format(self.task_name), "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}".format(self.task_name), "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        data_path = os.path.join(self.data_dir, "test-{0}".format(self.task_name), "test-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        data_path = os.path.join(self.data_dir, "resource", "label.vocab")
        label_list = self._read_text(data_path)
        return label_list
    
    def _read_text(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "rb") as file:
                data_list = []
                for line in file:
                    data_list.append(line.decode("utf-8").strip())

                return data_list
        else:
            raise FileNotFoundError("data path not found")
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)
                return data_list
        else:
            raise FileNotFoundError("data path not found")
    
    def _get_example(self,
                     data_list):
        example_list = []
        for data in data_list:
            guid = data["id"]
            text = tokenization.convert_to_unicode(data["text"])
            label = tokenization.convert_to_unicode(data["label"])
            example = InputExample(guid=guid, text=text, label=label)
            example_list.append(example)
        
        return example_list

def convert_single_example(ex_index,
                           example,
                           label_list,
                           max_seq_length,
                           tokenizer):
    """
    Converts a single BERT `InputExample` into a single BERT `InputFeatures`.
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    text_tokens = example.text.split(" ")
    label_tokens = example.label.split(" ")
    
    tokens = []
    labels = []
    for text_token, label_token in zip(text_tokens, label_tokens):
        text_sub_tokens = tokenizer.tokenize(text_token)
        label_sub_tokens = [label_token] + ["X"] * (len(text_sub_tokens) - 1)
        tokens.extend(text_sub_tokens)
        labels.extend(label_sub_tokens)
    
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    
    if len(labels) > max_seq_length - 2:
        labels = labels[0:(max_seq_length - 2)]
    
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    
    input_tokens = []
    segment_ids = []
    label_ids = []
    
    input_tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    
    for i, token in enumerate(tokens):
        input_tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    
    input_tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer):
    """
    Convert a set of BERT `InputExample`s to a list of BERT `InputFeatures`.
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        features.append(feature)
    
    return features

def input_fn_builder(features,
                     seq_length,
                     is_training,
                     drop_remainder):
    """
    Creates a BERT `input_fn` closure to be passed to TPUEstimator.
    """
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)
    
    def input_fn(params):
        batch_size = params["batch_size"]
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "label_ids": tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })
        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))
        
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d
    
    return input_fn

def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_file):
    """
    Convert a set of BERT `InputExample`s to a TFRecord file.
    """
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    
    writer = tf.python_io.TFRecordWriter(output_file)
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
    
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_ids])
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        writer.write(tf_example.SerializeToString())
    
    writer.close()

def file_based_input_fn_builder(input_file,
                                seq_length,
                                is_training,
                                drop_remainder):
    """
    Creates a BERT  `input_fn` closure to be passed to TPUEstimator.
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    
    def _decode_record(record,
                       name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        
        return example
    
    def input_fn(params):
        """
        The actual BERT input function.
        """
        batch_size = params["batch_size"]
        
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))
        
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        
        return d
    
    return input_fn
