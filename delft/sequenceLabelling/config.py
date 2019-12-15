import json


# Model parameters
class ModelConfig(object):

    def __init__(self, 
                 model_name="",
                 model_type="BidLSTM_CRF",
                 embeddings_name="glove-840B",
                 word_embedding_size=300,
                 char_emb_size=25, 
                 char_lstm_units=25,
                 max_char_length=30,
                 word_lstm_units=100, 
                 max_sequence_length=None,
                 dropout=0.5, 
                 recurrent_dropout=0.3,
                 use_char_feature=True, 
                 use_crf=True,
                 fold_number=1,
                 batch_size=64,
                 use_ELMo=False,
                 use_BERT=False):

        self.model_name = model_name
        self.model_type = model_type
        self.embeddings_name = embeddings_name

        self.char_vocab_size = None
        self.case_vocab_size = None

        self.char_embedding_size = char_emb_size
        self.num_char_lstm_units = char_lstm_units
        self.max_char_length = max_char_length

        self.max_sequence_length = max_sequence_length
        self.word_embedding_size = word_embedding_size
        self.num_word_lstm_units = word_lstm_units

        self.case_embedding_size = 5
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.use_char_feature = use_char_feature
        self.use_crf = use_crf
        self.fold_number = fold_number
        self.batch_size = batch_size # this is the batch size for test and prediction

        self.use_ELMo = use_ELMo
        self.use_BERT = use_BERT

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
                 max_checkpoints_to_keep=5,
                 multiprocessing=True):

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.clip_gradients = clip_gradients
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.patience = patience
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.multiprocessing = multiprocessing
