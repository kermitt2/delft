import json


class ModelConfig(object):
    """ wrapper class for model parameters """

    def __init__(self, 
                 model_name="",
                 model_type="gru",
                 char_emb_size=0, 
                 word_emb_size=300, 
                 dropout=0.5, 
                 use_char_feature=False,
                 maxlen=300):

        self.model_name = model_name
        self.model_type = model_type

        # Number of unique words in the vocab (plus 2, for <UNK>, <PAD>).
        self.vocab_size = None
        self.char_vocab_size = None

        self.char_embedding_size = char_emb_size
        self.word_embedding_size = word_emb_size
        self.dropout = dropout
        self.maxlen = maxlen

        self.use_char_feature = use_char_feature

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


class TrainingConfig(object):
    """ wrapper class for training hyperparameters """

    def __init__(self, 
                 batch_size=20, 
                 optimizer='adam', 
                 learning_rate=0.001, 
                 lr_decay=0.9,
                 clip_gradients=5.0, 
                 max_epoch=30, 
                 patience=5,
                 use_roc_auc=False):

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.clip_gradients = clip_gradients
        self.max_epoch = max_epoch
        self.patience = patience
        self.use_roc_auc = use_roc_auc
