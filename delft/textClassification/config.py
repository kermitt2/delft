import codecs, json 

# Model parameter
class ModelConfig(object):

    def __init__(self, 
                 model_name="",
                 architecture="gru",
                 embeddings_name="glove-840B", 
                 list_classes=[],
                 char_emb_size=0, 
                 word_emb_size=300, 
                 dropout=0.5, 
                 recurrent_dropout=0.25,
                 use_char_feature=False,
                 maxlen=300,
                 fold_number=1,
                 batch_size=64,
                 dense_size=32,
                 transformer_name=None
                 ):

        self.model_name = model_name
        self.architecture = architecture
        self.embeddings_name = embeddings_name

        #self.vocab_size = None
        #self.char_vocab_size = None

        self.char_embedding_size = char_emb_size
        self.word_embedding_size = word_emb_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.maxlen = maxlen
        self.dense_size = dense_size

        self.use_char_feature = use_char_feature
        self.list_classes = list_classes
        self.fold_number = fold_number
        self.batch_size = batch_size # this is the batch size for test and prediction

        self.transformer_name = transformer_name

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


# Training parameter
class TrainingConfig(object):

    def __init__(self,
                 learning_rate,
                 batch_size=256, 
                 optimizer='adam',
                 lr_decay=0.9,
                 clip_gradients=5.0, 
                 max_epoch=50, 
                 patience=5,
                 early_stop=True,
                 use_roc_auc=True,
                 class_weights=None,
                 multiprocessing=True):

        self.batch_size = batch_size # this is the batch size for training
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.clip_gradients = clip_gradients
        self.max_epoch = max_epoch
        self.patience = patience
        self.use_roc_auc = use_roc_auc
        self.class_weights = class_weights
        self.multiprocessing = multiprocessing
        self.early_stop = early_stop
        