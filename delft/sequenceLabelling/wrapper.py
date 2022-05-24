import os

# ask tensorflow to be quiet and not print hundred lines of logs
from delft.utilities.Transformer import TRANSFORMER_CONFIG_FILE_NAME, DEFAULT_TRANSFORMER_TOKENIZER_DIR
from delft.utilities.misc import print_parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from itertools import islice
import time
import json
import re
import math

import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# unfortunately when running in graph mode, we cannot use BERT pre-trained, 
# see https://github.com/huggingface/transformers/issues/3086
# but this is apparently not useful anyway to disable eager mode here, because 
# the Keras API compiles models before running them  
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

from delft.sequenceLabelling.trainer import DEFAULT_WEIGHT_FILE_NAME
from delft.sequenceLabelling.trainer import CONFIG_FILE_NAME
from delft.sequenceLabelling.trainer import PROCESSOR_FILE_NAME

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import prepare_preprocessor, Preprocessor
from delft.sequenceLabelling.tagger import Tagger
from delft.sequenceLabelling.trainer import Trainer
from delft.sequenceLabelling.trainer import Scorer
from delft.sequenceLabelling.evaluation import get_report

from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.utilities.numpy import concatenate_or_none

from delft.sequenceLabelling.evaluation import classification_report

import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)

from tensorflow.keras.utils import plot_model

class Sequence(object):

    # number of parallel worker for the data generator
    nb_workers = 6

    def __init__(self,
                 model_name=None,
                 architecture=None,
                 embeddings_name=None,
                 char_emb_size=25,
                 max_char_length=30,
                 char_lstm_units=25,
                 word_lstm_units=100,
                 max_sequence_length=300,
                 dropout=0.5,
                 recurrent_dropout=0.25,
                 batch_size=20,
                 optimizer='adam',
                 learning_rate=0.001,
                 lr_decay=0.9,
                 clip_gradients=5.0,
                 max_epoch=50,
                 early_stop=True,
                 patience=5,
                 max_checkpoints_to_keep=0,
                 use_ELMo=False,
                 log_dir=None,
                 fold_number=1,
                 multiprocessing=True,
                 features_indices=None,
                 transformer_name: str = None):

        if model_name is None:
            # add a dummy name based on the architecture
            model_name = architecture
            if embeddings_name is not None:
                model_name += "_" + embeddings_name
            if transformer_name is not None:
                model_name += "_" + transformer_name

        self.model = None
        self.models = None
        self.p: Preprocessor = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name

        word_emb_size = 0
        self.embeddings = None
        self.model_local_path = None

        self.registry = load_resource_registry("delft/resources-registry.json")

        if self.embeddings_name is not None:
            self.embeddings = Embeddings(self.embeddings_name, resource_registry=self.registry, use_ELMo=use_ELMo)
            word_emb_size = self.embeddings.embed_size
        else:
            self.embeddings = None
            word_emb_size = 0

        self.model_config = ModelConfig(model_name=model_name,
                                        architecture=architecture,
                                        embeddings_name=embeddings_name,
                                        word_embedding_size=word_emb_size,
                                        char_emb_size=char_emb_size,
                                        char_lstm_units=char_lstm_units,
                                        max_char_length=max_char_length,
                                        word_lstm_units=word_lstm_units,
                                        max_sequence_length=max_sequence_length,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        fold_number=fold_number,
                                        batch_size=batch_size,
                                        use_ELMo=use_ELMo,
                                        features_indices=features_indices,
                                        transformer_name=transformer_name)

        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              early_stop, patience,
                                              max_checkpoints_to_keep, multiprocessing)

    def train(self, x_train, y_train, f_train=None, x_valid=None, y_valid=None, f_valid=None, callbacks=None):
        # TBD if valid is None, segment train to get one if early_stop is True

        # we concatenate all the training+validation data to create the model vocabulary
        if not x_valid is None:
            x_all = np.concatenate((x_train, x_valid), axis=0)
        else:
            x_all = x_train

        if not y_valid is None:
            y_all = np.concatenate((y_train, y_valid), axis=0)
        else:
            y_all = y_train

        features_all = concatenate_or_none((f_train, f_valid), axis=0)

        self.p = prepare_preprocessor(x_all, y_all, features=features_all, model_config=self.model_config)

        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

        self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag), load_pretrained_weights=True)
        print_parameters(self.model_config, self.training_config)
        self.model.print_summary()

        # uncomment to plot graph
        #plot_model(self.model, 
        #    to_file='data/models/textClassification/'+self.model_config.model_name+'_'+self.model_config.architecture+'.png')

        trainer = Trainer(self.model,
                          self.models,
                          self.embeddings,
                          self.model_config,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p,
                          transformer_preprocessor=self.model.transformer_preprocessor
                          )
        trainer.train(x_train, y_train, x_valid, y_valid, features_train=f_train, features_valid=f_valid, callbacks=callbacks)
        if self.embeddings and self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()

    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, f_train=None, f_valid=None, callbacks=None):
        x_all = np.concatenate((x_train, x_valid), axis=0) if x_valid is not None else x_train
        y_all = np.concatenate((y_train, y_valid), axis=0) if y_valid is not None else y_train
        features_all = concatenate_or_none((f_train, f_valid), axis=0)

        self.p = prepare_preprocessor(x_all, y_all, features=features_all, model_config=self.model_config)

        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

        self.models = []
        trainer = Trainer(self.model,
                          self.models,
                          self.embeddings,
                          self.model_config,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p)

        trainer.train_nfold(x_train, y_train, x_valid, y_valid, f_train=f_train, f_valid=f_valid, callbacks=callbacks)
        if self.embeddings and self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()

    def eval(self, x_test, y_test, features=None):
        if self.model_config.fold_number > 1:
            self.eval_nfold(x_test, y_test, features=features)
        else:
            self.eval_single(x_test, y_test, features=features)

    def eval_single(self, x_test, y_test, features=None):
        if self.model is None:
            raise (OSError('Could not find a model.'))
        print_parameters(self.model_config, self.training_config)
        self.model.print_summary()

        if self.model_config.transformer_name is None:
            # we can use a data generator for evaluation

            # Prepare test data(steps, generator)
            generator = self.model.get_generator()
            test_generator = generator(x_test, y_test,
                batch_size=self.model_config.batch_size, preprocessor=self.p,
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False, features=features,
                output_input_offsets=True, use_chain_crf=self.model_config.use_chain_crf)

            # Build the evaluator and evaluate the model
            scorer = Scorer(test_generator, self.p, evaluation=True, use_crf=self.model_config.use_crf,
                use_chain_crf=self.model_config.use_chain_crf)
            scorer.model = self.model
            scorer.on_epoch_end(epoch=-1)
        else:
            # the architecture model uses a transformer layer
            # note that we could also use the above test_generator, but as an alternative here we check the 
            # test/prediction alignment of tokens and the validity of the maximum sequence input length
            # wrt the length of the test sequences 


            tagger = Tagger(self.model,
                          self.model_config,
                          self.embeddings,
                          preprocessor=self.p,
                          transformer_preprocessor=self.model.transformer_preprocessor)
            y_pred_pairs = tagger.tag(x_test, output_format=None, features=features)

            # keep only labels
            y_pred = []
            for result in y_pred_pairs:
                result_labels = []
                for pair in result:
                    result_labels.append(pair[1])
                y_pred.append(result_labels)

            nb_alignment_issues = 0
            for i in range(len(y_test)):
                if len(y_test[i]) != len(y_pred[i]):
                    #print("y_test:", y_test[i])
                    #print("y_pred:", y_pred[i])

                    nb_alignment_issues += 1
                    # BERT tokenizer appears to introduce some additional tokens without ## prefix,
                    # but we normally handled that well when predicting.
                    # To be very conservative, the following ensure the number of tokens always
                    # match, but it should never be used in practice.
                    if len(y_test[i]) < len(y_pred[i]):
                        y_test[i] = y_test[i] + ["O"] * (len(y_pred[i]) - len(y_test[i]))
                    if len(y_test[i]) > len(y_pred[i]):
                        y_pred[i] = y_pred[i] + ["O"] * (len(y_test[i]) - len(y_pred[i]))

            if nb_alignment_issues > 0:
                print("number of alignment issues with test set:", nb_alignment_issues)
                print("to solve them consider increasing the maximum sequence input length of the model and retrain")

            report, report_as_map = classification_report(y_test, y_pred, digits=4)
            print(report)

    def eval_nfold(self, x_test, y_test, features=None):
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
            for i in range(self.model_config.fold_number):

                if self.model_config.transformer_name is None:
                    the_model = self.models[i]
                    bert_preprocessor = None
                else:
                    # the architecture model uses a transformer layer, it is large and needs to be loaded from disk
                    dir_path = 'data/models/sequenceLabelling/'
                    weight_file = DEFAULT_WEIGHT_FILE_NAME.replace(".hdf5", str(i)+".hdf5")
                    self.model = get_model(self.model_config,
                               self.p,
                               ntags=len(self.p.vocab_tag),
                               load_pretrained_weights=False,
                               local_path= os.path.join(dir_path, self.model_config.model_name))
                    self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, weight_file))
                    the_model = self.model
                    bert_preprocessor = self.model.transformer_preprocessor

                if i == 0:
                    the_model.print_summary()
                    print_parameters(self.model_config, self.training_config)

                print('\n------------------------ fold ' + str(i) + ' --------------------------------------')

                # we can use a data generator for evaluation
                # Prepare test data(steps, generator)
                generator = the_model.get_generator()
                test_generator = generator(x_test, y_test,
                    batch_size=self.model_config.batch_size, preprocessor=self.p,
                    bert_preprocessor=bert_preprocessor,
                    char_embed_size=self.model_config.char_embedding_size,
                    max_sequence_length=self.model_config.max_sequence_length,
                    embeddings=self.embeddings, shuffle=False, features=features,
                    output_input_offsets=True, use_chain_crf=self.model_config.use_chain_crf)

                # Build the evaluator and evaluate the model
                scorer = Scorer(test_generator,
                                self.p,
                                evaluation=True,
                                use_crf=self.model_config.use_crf,
                                use_chain_crf=self.model_config.use_chain_crf)
                scorer.model = the_model
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

            fold_average_evaluation = {'labels': {}, 'micro': {}, 'macro': {}}

            micro_f1 = total_f1 / self.model_config.fold_number
            micro_precision = total_precision / self.model_config.fold_number
            micro_recall = total_recall / self.model_config.fold_number

            micro_eval_block = {'f1': micro_f1, 'precision': micro_precision, 'recall': micro_recall}
            fold_average_evaluation['micro'] = micro_eval_block

            # field-level average over the n folds
            labels = []
            for label in sorted(self.p.vocab_tag):
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
                    if label not in reports_as_map[j]['labels']:
                        continue
                    report_as_map = reports_as_map[j]['labels'][label]
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

                block_label = {'precision': avg_p, 'recall': avg_r, 'support': avg_support, 'f1': avg_f1}
                fold_average_evaluation['labels'][label] = block_label

            print("----------------------------------------------------------------------")
            print("\n** Worst ** model scores - run", str(worst_index))
            print(reports[worst_index])

            print("\n** Best ** model scores - run", str(best_index))
            print(reports[best_index])

            fold_nb = self.model_config.fold_number
            self.model_config.fold_number = 1
            if self.model_config.transformer_name is None:
                self.model = self.models[best_index]
            else:
                dir_path = 'data/models/sequenceLabelling/'
                weight_file = DEFAULT_WEIGHT_FILE_NAME.replace(".hdf5", str(best_index)+".hdf5")
                # saved config file must be updated to single fold
                self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, weight_file))

            print("----------------------------------------------------------------------")
            print("\nAverage over", str(int(fold_nb)), "folds")
            print(get_report(fold_average_evaluation, digits=4, include_avgs=['micro']))


    def tag(self, texts, output_format, features=None, batch_size=None):
        # annotate a list of sentences, return the list of annotations in the 
        # specified output_format

        if batch_size != None:
            self.model_config.batch_size = batch_size
            print("---")
            print("batch_size (prediction):", self.model_config.batch_size)
            print("---")

        if self.model:
            tagger = Tagger(self.model,
                            self.model_config,
                            self.embeddings,
                            preprocessor=self.p,
                            transformer_preprocessor=self.model.transformer_preprocessor)
            start_time = time.time()
            annotations = tagger.tag(texts, output_format, features=features)
            runtime = round(time.time() - start_time, 3)
            if output_format == 'json':
                annotations["runtime"] = runtime
            #else:
            #    print("runtime: %s seconds " % (runtime))
            return annotations
        else:
            raise (OSError('Could not find a model.' + str(self.model)))

    def tag_file(self, file_in, output_format, file_out, batch_size=None):
        # Annotate a text file containing one sentence per line, the annotations are
        # written in the output file if not None, in the standard output otherwise.
        # Processing is streamed by batches so that we can process huge files without
        # memory issues

        if batch_size != None:
            self.model_config.batch_size = batch_size
            print("---")
            print("batch_size (prediction):", self.model_config.batch_size)
            print("---")

        if self.model:
            tagger = Tagger(self.model,
                            self.model_config,
                            self.embeddings,
                            preprocessor=self.p,
                            transformer_preprocessor=self.model.transformer_preprocessor)
            start_time = time.time()
            if file_out != None:
                out = open(file_out,'w')
            first = True
            with open(file_in, 'r') as f:
                texts = None
                while texts == None or len(texts) == self.model_config.batch_size * self.nb_workers:

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
                          if file_out == None:
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
                          if file_out == None:
                              print(jsonString, end='', flush=True)
                          else:
                              out.write(jsonString)
                          first = True
                          for jsonStr in annotations["texts"]:
                              jsonString = json.dumps(jsonStr, sort_keys=False, indent=4, ensure_ascii=False)
                              #jsonString = jsonString.replace('\n', '\n\t\t')
                              jsonString = re.sub('\n', '\n        ', jsonString)
                              if file_out == None:
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
                          if file_out == None:
                              print(',\n        '+jsonString, end='', flush=True)
                          else:
                              out.write(',\n        ')
                              out.write(jsonString)

            runtime = round(time.time() - start_time, 3)
            if not directDump:
                jsonString = "\n    ],\n"
                jsonString += '    "runtime": ' + str(runtime)
                jsonString += "\n}\n"
                if file_out == None:
                    print(jsonString)
                else:
                    out.write(jsonString)

            if file_out != None:
                out.close()
            #print("runtime: %s seconds " % (runtime))
        else:
            raise (OSError('Could not find a model.'))

    def save(self, dir_path='data/models/sequenceLabelling/', weight_file=DEFAULT_WEIGHT_FILE_NAME):
        # create subfolder for the model if not already exists
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, CONFIG_FILE_NAME))
        print('model config file saved')

        self.p.save(os.path.join(directory, PROCESSOR_FILE_NAME))
        print('preprocessor saved')

        if self.model is None and self.model_config.fold_number > 1:
            print('Error: model not saved. Evaluation need to be called first to select the best fold model to be saved')
        else:
            self.model.save(os.path.join(directory, weight_file))

            # save pretrained transformer config if used in the model
            if self.model.transformer_config is not None:
                self.model.transformer_config.to_json_file(os.path.join(directory, TRANSFORMER_CONFIG_FILE_NAME))
                print('transformer config saved')

            if self.model.transformer_preprocessor is not None:
                self.model.transformer_preprocessor.tokenizer.save_pretrained(os.path.join(directory, DEFAULT_TRANSFORMER_TOKENIZER_DIR))
                print('transformer tokenizer saved')

        print('model saved')

    def load(self, dir_path='data/models/sequenceLabelling/', weight_file=DEFAULT_WEIGHT_FILE_NAME):
        model_path = os.path.join(dir_path, self.model_config.model_name)
        self.model_config = ModelConfig.load(os.path.join(model_path, CONFIG_FILE_NAME))

        if self.model_config.embeddings_name is not None:
            # load embeddings
            # Do not use cache in 'prediction/production' mode
            self.embeddings = Embeddings(self.model_config.embeddings_name, resource_registry=self.registry, use_ELMo=self.model_config.use_ELMo, use_cache=False)
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.embeddings = None
            self.model_config.word_embedding_size = 0

        self.p = Preprocessor.load(os.path.join(dir_path, self.model_config.model_name, PROCESSOR_FILE_NAME))
        self.model = get_model(self.model_config,
                               self.p,
                               ntags=len(self.p.vocab_tag),
                               load_pretrained_weights=False,
                               local_path=os.path.join(dir_path, self.model_config.model_name))
        print("load weights from", os.path.join(dir_path, self.model_config.model_name, weight_file))
        self.model.load(filepath=os.path.join(dir_path, self.model_config.model_name, weight_file))
        self.model.print_summary()

def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]
