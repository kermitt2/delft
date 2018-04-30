import os

import numpy as np

from sequenceLabelling.config import ModelConfig, TrainingConfig
from sequenceLabelling.evaluator import Evaluator
from sequenceLabelling.models import SeqLabelling_BidLSTM_CRF
from sequenceLabelling.preprocess import prepare_preprocessor, WordPreprocessor
from sequenceLabelling.tagger import Tagger
from sequenceLabelling.trainer import Trainer
from utilities.Embeddings import filter_embeddings

# based on https://github.com/Hironsan/anago/blob/master/anago/wrapper.py
# with various modifcations

class Sequence(object):

    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_file = 'preprocessor.pkl'

    def __init__(self, 
                 model_name="",
                 char_emb_size=25, 
                 word_emb_size=300, 
                 char_lstm_units=25,
                 word_lstm_units=100, 
                 dropout=0.5, 
                 use_char_feature=True, 
                 use_crf=True,
                 batch_size=20, 
                 optimizer='adam', 
                 learning_rate=0.001, 
                 lr_decay=0.9,
                 clip_gradients=5.0, 
                 max_epoch=50, 
                 patience=5,
                 max_checkpoints_to_keep=5, 
                 log_dir=None,
                 embeddings=()):

        self.model_config = ModelConfig(model_name, char_emb_size, word_emb_size, char_lstm_units,
                                        word_lstm_units, dropout, use_char_feature, use_crf)
        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              patience, 
                                              max_checkpoints_to_keep)
        self.model = None
        self.models = None
        self.p = None
        self.log_dir = log_dir
        self.embeddings = embeddings 

    def train(self, x_train, y_train, x_valid=None, y_valid=None, vocab_init=None):
        self.p = prepare_preprocessor(x_train, y_train, vocab_init=vocab_init)
        embeddings = filter_embeddings(self.embeddings, self.p.vocab_word,
                                       self.model_config.word_embedding_size)
        self.model_config.vocab_size = len(self.p.vocab_word)
        self.model_config.char_vocab_size = len(self.p.vocab_char)

        self.model = SeqLabelling_BidLSTM_CRF(self.model_config, embeddings, len(self.p.vocab_tag))

        trainer = Trainer(self.model, 
                          self.models,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p)
        trainer.train(x_train, y_train, x_valid, y_valid)

    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, vocab_init=None, fold_number=10):
        self.p = prepare_preprocessor(x_train, y_train, vocab_init=vocab_init)
        embeddings = filter_embeddings(self.embeddings, self.p.vocab_word,
                                       self.model_config.word_embedding_size)
        self.model_config.vocab_size = len(self.p.vocab_word)
        self.model_config.char_vocab_size = len(self.p.vocab_char)

        for k in range(0,fold_number-1):
            self.model = SeqLabelling_BidLSTM_CRF(self.model_config, embeddings, len(self.p.vocab_tag))
            self.models.append(model)

        trainer = Trainer(self.model, 
                          self.models,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p)
        trainer.train_nfold(x_train, y_train, x_valid, y_valid)

    def eval(self, x_test, y_test):
        if self.model:
            evaluator = Evaluator(self.model, preprocessor=self.p)
            evaluator.eval(x_test, y_test)
        else:
            raise (OSError('Could not find a model.'))

    def analyze(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze(words)
        else:
            raise (OSError('Could not find a model.'))

    def tag(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.tag(words)
        else:
            raise (OSError('Could not find a model.'))

    def save(self, dir_path='data/models/sequenceLabelling/'):
        
        # create subfolder for the model if not already exists
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.p.save(os.path.join(directory, self.preprocessor_file))
        print('preprocessor saved')
        
        self.model_config.save(os.path.join(directory, self.config_file))
        print('model config file saved')
        
        self.model.save(os.path.join(directory, self.weight_file))
        print('model saved')

    def load(self, dir_path='data/models/sequenceLabelling//'):
        self.p = WordPreprocessor.load(os.path.join(dir_path, self.model_config.model_name, self.preprocessor_file))
        
        config = ModelConfig.load(os.path.join(dir_path, self.model_config.model_name, self.config_file))
        
        dummy_embeddings = np.zeros((config.vocab_size, config.word_embedding_size), dtype=np.float32)
        self.model = SeqLabelling_BidLSTM_CRF(config, dummy_embeddings, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, self.weight_file))

