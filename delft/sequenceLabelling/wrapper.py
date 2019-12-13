import os

from itertools import islice
import time
import json
import re
import math

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

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import prepare_preprocessor, WordPreprocessor
from delft.sequenceLabelling.tagger import Tagger
from delft.sequenceLabelling.trainer import Trainer
from delft.sequenceLabelling.data_generator import DataGenerator
from delft.sequenceLabelling.trainer import Scorer

from delft.utilities.Embeddings import Embeddings

# initially derived from https://github.com/Hironsan/anago/blob/master/anago/wrapper.py
# with various modifications


class Sequence(object):

    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    preprocessor_file = 'preprocessor.pkl'

    # number of parallel worker for the data generator when not using ELMo
    nb_workers = 6

    def __init__(self, 
                 model_name,
                 model_type="BidLSTM_CRF",
                 embeddings_name=None,
                 char_emb_size=25, 
                 max_char_length=30,
                 char_lstm_units=25,
                 word_lstm_units=100, 
                 max_sequence_length=None,
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
                 use_ELMo=False,
                 use_BERT=False,
                 fold_number=1):

        self.model = None
        self.models = None
        self.p = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name

        word_emb_size = 0
        if embeddings_name is not None:
            self.embeddings = Embeddings(embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT) 
            word_emb_size = self.embeddings.embed_size

        self.model_config = ModelConfig(model_name=model_name, 
                                        model_type=model_type, 
                                        embeddings_name=embeddings_name, 
                                        word_embedding_size=word_emb_size, 
                                        char_emb_size=char_emb_size, 
                                        char_lstm_units=char_lstm_units, 
                                        max_char_length=max_char_length,
                                        word_lstm_units=word_lstm_units,
                                        max_sequence_length=max_sequence_length, 
                                        dropout=dropout, 
                                        recurrent_dropout=recurrent_dropout, 
                                        use_char_feature=use_char_feature, 
                                        use_crf=use_crf, 
                                        fold_number=fold_number, 
                                        batch_size=batch_size,
                                        use_ELMo=use_ELMo,
                                        use_BERT=use_BERT)

        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              early_stop, patience, 
                                              max_checkpoints_to_keep)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        # TBD if valid is None, segment train to get one
        x_all = np.concatenate((x_train, x_valid), axis=0) if x_valid is not None else x_train
        y_all = np.concatenate((y_train, y_valid), axis=0) if y_valid is not None else y_train
        self.p = prepare_preprocessor(x_all, y_all, self.model_config)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

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
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

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
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

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
              max_sequence_length=self.model_config.max_sequence_length,
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
            reports_as_map = []
            total_precision = 0
            total_recall = 0
            for i in range(0, self.model_config.fold_number):
                print('\n------------------------ fold ' + str(i) + ' --------------------------------------')

                # Prepare test data(steps, generator)
                test_generator = DataGenerator(x_test, y_test, 
                  batch_size=self.training_config.batch_size, preprocessor=self.p, 
                  char_embed_size=self.model_config.char_embedding_size, 
                  max_sequence_length=self.model_config.max_sequence_length,
                  embeddings=self.embeddings, shuffle=False)

                # Build the evaluator and evaluate the model
                scorer = Scorer(test_generator, self.p, evaluation=True)
                scorer.model = self.models[i]
                scorer.on_epoch_end(epoch=-1) 
                f1 = scorer.f1
                precision = scorer.precision
                recall = scorer.recall
                reports.append(scorer.report)
                reports_as_map.append(scorer.report_as_map)

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

            print("----------------------------------------------------------------------")
            print("\naverage over", self.model_config.fold_number, "folds")

            name_width = 0
            for label in self.p.vocab_tag:
              name_width = max(name_width, len(label))

            width = max(name_width, 10)
            digits = 4
            headers = ["precision", "recall", "f1-score", "support"]
            head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers) + "\n"
            print(head_fmt.format(u'', *headers, width=width))
            #print(u'\n')

            row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}'

            # field-level average over th n folds
            labels = []
            for label in self.p.vocab_tag:
              if label == 'O' or label == '<PAD>':
                continue
              if label.startswith("B-") or label.startswith("S-") or label.startswith("I-") or label.startswith("E-"):
                label = label[2:]

              if label in labels:
                continue
              labels.append(label)

              sum_p = 0
              sum_r = 0
              sum_f1 = 0
              sum_support = 0
              for j in range(0, self.model_config.fold_number):
                if not label in reports_as_map[j]:
                  continue
                report_as_map = reports_as_map[j][label]
                sum_p += report_as_map["precision"]
                sum_r += report_as_map["recall"]
                sum_f1 += report_as_map["f1"]
                sum_support += report_as_map["support"]
              avg_p = sum_p / self.model_config.fold_number
              avg_r = sum_r / self.model_config.fold_number
              avg_f1 = sum_f1 / self.model_config.fold_number
              avg_support = sum_support / self.model_config.fold_number
              avg_support_dec = str(avg_support-int(avg_support))[1:]
              if avg_support_dec != '0':
                avg_support = math.floor(avg_support)
              print(row_fmt.format(*[label, avg_p, avg_r, avg_f1, avg_support], width=width, digits=digits))

            print("\n\tmacro f1 =", '{0:.4f}'.format(macro_f1))
            print("\tmacro precision =", '{0:.4f}'.format(macro_precision))
            print("\tmacro recall =", '{0:.4f}'.format(macro_recall), "\n")

            print("\n** Worst ** model scores -")
            print(reports[worst_index])

            self.model = self.models[best_index]
            print("\n** Best ** model scores -")
            print(reports[best_index])

    def tag(self, texts, output_format):
        # annotate a list of sentences, return the list of annotations in the 
        # specified output_format
        if self.model:
            tagger = Tagger(self.model, self.model_config, self.embeddings, preprocessor=self.p)
            start_time = time.time()
            annotations = tagger.tag(texts, output_format)
            runtime = round(time.time() - start_time, 3)
            if output_format is 'json':
                annotations["runtime"] = runtime
            #else:
            #    print("runtime: %s seconds " % (runtime))
            return annotations
        else:
                raise (OSError('Could not find a model.' + str(self.model)))

    def tag_file(self, file_in, output_format, file_out):
        # Annotate a text file containing one sentence per line, the annotations are
        # written in the output file if not None, in the standard output otherwise.
        # Processing is streamed by batches so that we can process huge files without
        # memory issues
        if self.model:
            tagger = Tagger(self.model, self.model_config, self.embeddings, preprocessor=self.p)
            start_time = time.time()
            if file_out is not None:
                out = open(file_out,'w')
            first = True
            with open(file_in, 'r') as f:
                texts = None
                while texts is None or len(texts) == self.model_config.batch_size * self.nb_workers:

                  texts = next_n_lines(f, self.model_config.batch_size * self.nb_workers)
                  annotations = tagger.tag(texts, output_format)
                  # if the following is true, we just output the JSON returned by the tagger without any modification
                  directDump = False
                  if first:
                      first = False
                      if len(texts) < self.model_config.batch_size * self.nb_workers:
                          runtime = round(time.time() - start_time, 3)
                          annotations['runtime'] = runtime
                          jsonString = json.dumps(annotations, sort_keys=False, indent=4, ensure_ascii=False)
                          if file_out is None:
                              print(jsonString)
                          else:
                              out.write(jsonString)
                          directDump = True
                      else:
                          # we need to modify a bit the JSON outputted by the tagger to glue the different batches
                          # output the general information attributes
                          jsonString = '{\n    "software": ' + json.dumps(annotations["software"], ensure_ascii=False) + ",\n"
                          jsonString += '    "date": ' + json.dumps(annotations["date"], ensure_ascii=False) + ",\n"
                          jsonString += '    "model": ' + json.dumps(annotations["model"], ensure_ascii=False) + ",\n"
                          jsonString += '    "texts": ['
                          if file_out is None:
                              print(jsonString, end='', flush=True)
                          else:
                              out.write(jsonString)
                          first = True
                          for jsonStr in annotations["texts"]:
                              jsonString = json.dumps(jsonStr, sort_keys=False, indent=4, ensure_ascii=False)
                              #jsonString = jsonString.replace('\n', '\n\t\t')
                              jsonString = re.sub('\n', '\n        ', jsonString)
                              if file_out is None:
                                  if not first:
                                      print(',\n        '+jsonString, end='', flush=True)
                                  else:
                                      first = False
                                      print('\n        '+jsonString, end='', flush=True)
                              else:
                                  if not first:
                                      out.write(',\n        ')
                                      out.write(jsonString)
                                  else:
                                      first = False
                                      out.write('\n        ')
                                      out.write(jsonString)
                  else:
                      for jsonStr in annotations["texts"]:
                          jsonString = json.dumps(jsonStr, sort_keys=False, indent=4, ensure_ascii=False)
                          jsonString = re.sub('\n', '\n        ', jsonString)
                          if file_out is None:
                              print(',\n        '+jsonString, end='', flush=True)
                          else:
                              out.write(',\n        ')
                              out.write(jsonString)

            runtime = round(time.time() - start_time, 3)
            if not directDump: 
                jsonString = "\n    ],\n"
                jsonString += '    "runtime": ' + str(runtime)
                jsonString += "\n}\n"
                if file_out is None:
                    print(jsonString)
                else:
                    out.write(jsonString) 

            if file_out is not None:
                out.close() 
            #print("runtime: %s seconds " % (runtime))
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
        self.embeddings = Embeddings(self.model_config.embeddings_name, use_ELMo=self.model_config.use_ELMo, use_BERT=self.model_config.use_BERT) 
        self.model_config.word_embedding_size = self.embeddings.embed_size

        self.model = get_model(self.model_config, self.p, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, self.weight_file))


def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]
