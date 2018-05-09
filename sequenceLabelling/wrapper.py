import os

import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)

from sequenceLabelling.config import ModelConfig, TrainingConfig
from sequenceLabelling.models import BidLSTM_CRF
from sequenceLabelling.preprocess import prepare_preprocessor, WordPreprocessor
from sequenceLabelling.tagger import Tagger
from sequenceLabelling.trainer import Trainer
from sequenceLabelling.data_generator import DataGenerator
from sequenceLabelling.trainer import F1scorer

from utilities.Embeddings import Embeddings

# initially derived from https://github.com/Hironsan/anago/blob/master/anago/wrapper.py
# with various modifications

class Sequence(object):

    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_file = 'preprocessor.pkl'

    def __init__(self, 
                 model_name,
                 model_type="BidLSTM_CRF",
                 embeddings_name="glove-840B",
                 char_emb_size=25, 
                 char_lstm_units=25,
                 word_lstm_units=100, 
                 dropout=0.5, 
                 recurrent_dropout=0.25,
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
                 fold_number=1):

        self.model = None
        self.models = None
        self.p = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name
        self.embeddings = Embeddings(embeddings_name) 

        self.model_config = ModelConfig(model_name=model_name, 
                                        model_type=model_type, 
                                        embeddings_name=embeddings_name, 
                                        word_emb_size=self.embeddings.embed_size, 
                                        char_emb_size=char_emb_size, 
                                        char_lstm_units=char_lstm_units, 
                                        word_lstm_units=word_lstm_units, 
                                        dropout=dropout, 
                                        recurrent_dropout=recurrent_dropout, 
                                        use_char_feature=use_char_feature, 
                                        use_crf=use_crf, 
                                        fold_number=fold_number, 
                                        batch_size=batch_size)

        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              patience, max_checkpoints_to_keep)


    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        # TBD if valid is None, segment train to get one
        x_all = np.concatenate((x_train, x_valid), axis=0)
        y_all = np.concatenate((y_train, y_valid), axis=0)
        self.p = prepare_preprocessor(x_all, y_all)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model = BidLSTM_CRF(self.model_config, len(self.p.vocab_tag))

        trainer = Trainer(self.model, 
                          self.models,
                          self.embeddings,
                          self.model_config,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p
                          )
        trainer.train(x_train, y_train, x_valid, y_valid)


    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, fold_number=10):
        self.p = prepare_preprocessor(x_train, y_train)
        self.model_config.char_vocab_size = len(self.p.vocab_char)

        for k in range(0,fold_number-1):
            self.model = BidLSTM_CRF(self.model_config, len(self.p.vocab_tag))
            self.models.append(model)

        trainer = Trainer(self.model, 
                          self.models,
                          self.embeddings,
                          self.model_config,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p
                          )
        trainer.train_nfold(x_train, y_train, x_valid, y_valid)


    def eval(self, x_test, y_test):
        if self.model_config.fold_number is 1:
            if self.model:
                # Prepare test data(steps, generator)
                test_generator = DataGenerator(x_test, y_test, 
                  batch_size=self.training_config.batch_size, preprocessor=self.p, 
                  char_embed_size=self.model_config.char_embedding_size, 
                  embeddings=self.embeddings, shuffle=False)

                # Build the evaluator and evaluate the model
                f1scorer = F1scorer(test_generator, self.p, evaluation=True)
                f1scorer.model = self.model
                f1scorer.on_epoch_end(epoch=-1) 
            else:
                raise (OSError('Could not find a model.'))
        """
        else:
            if self.models is not None:
                total_f1 = 0
                total_correct_preds = 0
                total_total_correct = 0
                total_total_preds = 0
                for i in range(0, self.model_config.fold_number):
                    # Prepare test data(steps, generator)
                    test_generator = DataGenerator(x_test, y_test, 
                      batch_size=self.training_config.batch_size, preprocessor=self.p, 
                      word_embed_size=self.model_config.word_embedding_size, 
                      char_embed_size=self.model_config.char_embedding_size, 
                      embeddings=self.embeddings, shuffle=False)

                    # Build the evaluator and evaluate the model
                    f1scorer = F1scorer(test_generator, self.p)
                    f1scorer.model = self.models[i]
                    f1scorer.on_epoch_end(epoch=-1) 
                    f1 = f1scorer.f1
                    correct_preds = f1scorer.correct_preds
                    total_correct = f1scorer.total_correct
                    total_preds = f1scorer.total_preds

                macro_f1 = f1scorer.calc_f1(total_correct_preds, total_total_correct, total_total_preds)
                micro_f1 = total_f1 / self.model_config.fold_number
        """

    def tag(self, texts, output_format):
        if self.model:
            tagger = Tagger(self.model, self.model_config, self.embeddings, preprocessor=self.p)
            return tagger.tag(texts, output_format)
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


    def load(self, dir_path='data/models/sequenceLabelling/'):
        self.p = WordPreprocessor.load(os.path.join(dir_path, self.model_config.model_name, self.preprocessor_file))
        
        self.model_config = ModelConfig.load(os.path.join(dir_path, self.model_config.model_name, self.config_file))

        # load embeddings
        self.embeddings = Embeddings(self.model_config.embeddings_name) 

        self.model = BidLSTM_CRF(self.model_config, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, self.weight_file))
