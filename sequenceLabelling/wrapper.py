import os

import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)

# ask tensorflow to be quiet and not print hundred lines of logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.set_random_seed(7)

import keras.backend as K
# Initialize Keras session
#sess = tf.Session()
#K.set_session(sess)

from sequenceLabelling.config import ModelConfig, TrainingConfig
from sequenceLabelling.models import get_model
from sequenceLabelling.preprocess import prepare_preprocessor, WordPreprocessor
from sequenceLabelling.tagger import Tagger
from sequenceLabelling.trainer import Trainer
from sequenceLabelling.data_generator import DataGenerator
from sequenceLabelling.trainer import Scorer

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
                 embeddings_name=None,
                 char_emb_size=25, 
                 max_char_length=30,
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
                 early_stop=True,
                 patience=5,
                 max_checkpoints_to_keep=5, 
                 log_dir=None,
                 use_ELMo=True,
                 fold_number=1):

        self.model = None
        self.models = None
        self.p = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name

        word_emb_size = 0
        if embeddings_name is not None:
            self.embeddings = Embeddings(embeddings_name, use_ELMo=use_ELMo) 
            word_emb_size = self.embeddings.embed_size

        self.model_config = ModelConfig(model_name=model_name, 
                                        model_type=model_type, 
                                        embeddings_name=embeddings_name, 
                                        word_embedding_size=word_emb_size, 
                                        char_emb_size=char_emb_size, 
                                        char_lstm_units=char_lstm_units, 
                                        max_char_length=max_char_length,
                                        word_lstm_units=word_lstm_units, 
                                        dropout=dropout, 
                                        recurrent_dropout=recurrent_dropout, 
                                        use_char_feature=use_char_feature, 
                                        use_crf=use_crf, 
                                        fold_number=fold_number, 
                                        batch_size=batch_size,
                                        use_ELMo=use_ELMo)

        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              early_stop, patience, 
                                              max_checkpoints_to_keep)


    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        # TBD if valid is None, segment train to get one
        x_all = np.concatenate((x_train, x_valid), axis=0)
        y_all = np.concatenate((y_train, y_valid), axis=0)
        self.p = prepare_preprocessor(x_all, y_all, self.model_config)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

        """
        if self.embeddings.use_ELMo:
            # dump token context independent data for the train set, done once for the training
            x_train_local = x_train
            if not self.training_config.early_stop:
                # in case we want to train with the validation set too, we dump also
                # the ELMo embeddings for the token of the valid set
                x_train_local = np.concatenate((x_train, x_valid), axis=0)
            self.embeddings.dump_ELMo_token_embeddings(x_train_local)
        """
        self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
        trainer = Trainer(self.model, 
                          self.models,
                          self.embeddings,
                          self.model_config,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p
                          )
        trainer.train(x_train, y_train, x_valid, y_valid)
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()

    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, fold_number=10):
        if x_valid is not None and y_valid is not None:
            x_all = np.concatenate((x_train, x_valid), axis=0)
            y_all = np.concatenate((y_train, y_valid), axis=0)
            self.p = prepare_preprocessor(x_all, y_all, self.model_config)
        else:
            self.p = prepare_preprocessor(x_train, y_train, self.model_config)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)
        self.p.return_lengths = True
        
        #self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
        self.models = []

        for k in range(0, fold_number):
            model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
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
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()

    def eval(self, x_test, y_test):
        if self.model_config.fold_number > 1 and self.models and len(self.models) == self.model_config.fold_number:
            self.eval_nfold(x_test, y_test)
        else:
            self.eval_single(x_test, y_test)


    def eval_single(self, x_test, y_test):   
        if self.model:
            # Prepare test data(steps, generator)
            test_generator = DataGenerator(x_test, y_test, 
              batch_size=self.training_config.batch_size, preprocessor=self.p, 
              char_embed_size=self.model_config.char_embedding_size, 
              embeddings=self.embeddings, shuffle=False)

            # Build the evaluator and evaluate the model
            scorer = Scorer(test_generator, self.p, evaluation=True)
            scorer.model = self.model
            scorer.on_epoch_end(epoch=-1) 
        else:
            raise (OSError('Could not find a model.'))


    def eval_nfold(self, x_test, y_test):
        if self.models is not None:
            total_f1 = 0
            best_f1 = 0
            best_index = 0
            worst_f1 = 1
            worst_index = 0
            reports = []
            total_precision = 0
            total_recall = 0
            for i in range(0, self.model_config.fold_number):
                print('\n------------------------ fold ' + str(i) + '--------------------------------------')

                # Prepare test data(steps, generator)
                test_generator = DataGenerator(x_test, y_test, 
                  batch_size=self.training_config.batch_size, preprocessor=self.p, 
                  char_embed_size=self.model_config.char_embedding_size, 
                  embeddings=self.embeddings, shuffle=False)

                # Build the evaluator and evaluate the model
                scorer = Scorer(test_generator, self.p, evaluation=True)
                scorer.model = self.models[i]
                scorer.on_epoch_end(epoch=-1) 
                f1 = scorer.f1
                precision = scorer.precision
                recall = scorer.recall
                reports.append(scorer.report)
                
                if best_f1 < f1:
                    best_f1 = f1
                    best_index = i
                if worst_f1 > f1:
                    worst_f1 = f1
                    worst_index = i
                total_f1 += f1
                total_precision += precision
                total_recall += recall

            macro_f1 = total_f1 / self.model_config.fold_number
            macro_precision = total_precision / self.model_config.fold_number
            macro_recall = total_recall / self.model_config.fold_number

            print("\naverage over", self.model_config.fold_number, "folds")
            print("\tmacro f1 =", macro_f1)
            print("\tmacro precision =", macro_precision)
            print("\tmacro recall =", macro_recall, "\n")

            print("\n** Worst ** model scores - \n")
            print(reports[worst_index])

            self.model = self.models[best_index]
            print("\n** Best ** model scores - \n")
            print(reports[best_index])
        

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
        self.embeddings = Embeddings(self.model_config.embeddings_name, use_ELMo=self.model_config.use_ELMo) 
        self.model_config.word_embedding_size = self.embeddings.embed_size

        self.model = get_model(self.model_config, self.p, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, self.weight_file))
