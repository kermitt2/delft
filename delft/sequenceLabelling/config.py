import json


# Model parameters

class ModelConfig(object):
    DEFAULT_FEATURES_VOCABULARY_SIZE = 12
    DEFAULT_FEATURES_EMBEDDING_SIZE = 4

    def __init__(self, 
                 model_name="",
                 architecture="BidLSTM_CRF",
                 embeddings_name="glove-840B",
                 word_embedding_size=300,
                 char_emb_size=25, 
                 char_lstm_units=25,
                 max_char_length=30,
                 word_lstm_units=100, 
                 max_sequence_length=300,
                 dropout=0.5, 
                 recurrent_dropout=0.3,
                 use_crf=False,
                 use_chain_crf=False,
                 fold_number=1,
                 batch_size=64,
                 use_ELMo=False,
                 features_vocabulary_size=DEFAULT_FEATURES_VOCABULARY_SIZE,
                 features_indices=None,
                 features_embedding_size=DEFAULT_FEATURES_EMBEDDING_SIZE,
                 features_lstm_units=DEFAULT_FEATURES_EMBEDDING_SIZE,
                 transformer_name=None):

        self.model_name = model_name
        self.architecture = architecture
        self.embeddings_name = embeddings_name

        self.char_vocab_size = None
        self.case_vocab_size = None

        self.char_embedding_size = char_emb_size
        self.num_char_lstm_units = char_lstm_units
        self.max_char_length = max_char_length

        # Features
        self.features_vocabulary_size = features_vocabulary_size    # maximum number of unique values per feature
        self.features_indices = features_indices
        self.features_embedding_size = features_embedding_size
        self.features_lstm_units = features_lstm_units

        self.max_sequence_length = max_sequence_length
        self.word_embedding_size = word_embedding_size
        self.num_word_lstm_units = word_lstm_units

        self.case_embedding_size = 5
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.use_crf = use_crf
        self.use_chain_crf = use_chain_crf
        self.fold_number = fold_number
        self.batch_size = batch_size # this is the batch size for prediction

        self.transformer_name = transformer_name

        self.use_ELMo = use_ELMo

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(vars(self), f, sort_keys=False, indent=4)

    @classmethod
    def load(cls, file):
        with open(file) as f:
            variables = json.load(f)
            self = cls()
            for key, val in variables.items():
                setattr(self, key, val)
        return self


# Training parameters
class TrainingConfig(object):

    def __init__(self, 
                 batch_size=20, 
                 optimizer='adam', 
                 learning_rate=0.001, 
                 lr_decay=0.9,
                 clip_gradients=5.0, 
                 max_epoch=50, 
                 early_stop=True,
                 patience=5,
                 max_checkpoints_to_keep=0,
                 multiprocessing=True):

        self.batch_size = batch_size # this is the batch size for training
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.clip_gradients = clip_gradients
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.patience = patience
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.multiprocessing = multiprocessing
