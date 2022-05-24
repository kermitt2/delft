import os

# ask tensorflow to be quiet and not print hundred lines of logs
from delft.utilities.misc import print_parameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# unfortunately when running in graph mode, we cannot use BERT pre-trained, 
# see https://github.com/huggingface/transformers/issues/3086
# but this is apparently not useful anyway to disable eager mode here, because 
# the Keras API compiles models before running them 
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import datetime

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import getModel
from delft.textClassification.models import train_folds
from delft.textClassification.models import predict_folds
from delft.textClassification.data_generator import DataGenerator

from delft.utilities.Transformer import Transformer, TRANSFORMER_CONFIG_FILE_NAME, DEFAULT_TRANSFORMER_TOKENIZER_DIR

from delft.utilities.Embeddings import Embeddings, load_resource_registry

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, r2_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR) 

from tensorflow.keras.utils import plot_model

class Classifier(object):

    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'

    def __init__(self, 
                 model_name=None,
                 architecture="gru",
                 embeddings_name=None,
                 list_classes=[],
                 char_emb_size=25, 
                 dropout=0.5, 
                 recurrent_dropout=0.25,
                 use_char_feature=False, 
                 batch_size=256, 
                 optimizer='adam', 
                 learning_rate=0.001, 
                 lr_decay=0.9,
                 clip_gradients=5.0, 
                 max_epoch=50, 
                 patience=5,
                 log_dir=None,
                 maxlen=300,
                 fold_number=1,
                 use_roc_auc=True,
                 early_stop=True,
                 class_weights=None,
                 multiprocessing=True,
                 transformer_name: str=None):

        if model_name is None:
            # add a dummy name based on the architecture
            model_name = architecture
            if embeddings_name is not None:
                model_name += "_" + embeddings_name
            if transformer_name is not None:
                model_name += "_" + transformer_name

        self.model = None
        self.models = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name
        self.embeddings = None

        # if transformer_name is None, no bert layer is present in the model
        self.transformer_name = None

        self.registry = load_resource_registry("delft/resources-registry.json")

        word_emb_size = 0
        if transformer_name is not None:
            self.transformer_name = transformer_name
            self.embeddings_name = None
            self.embeddings = None
        elif self.embeddings_name is not None:
            self.embeddings = Embeddings(self.embeddings_name, resource_registry=self.registry)
            word_emb_size = self.embeddings.embed_size
        
        self.model_config = ModelConfig(model_name=model_name, 
                                        architecture=architecture, 
                                        embeddings_name=embeddings_name, 
                                        list_classes=list_classes, 
                                        char_emb_size=char_emb_size, 
                                        word_emb_size=word_emb_size, 
                                        dropout=dropout, 
                                        recurrent_dropout=recurrent_dropout,
                                        use_char_feature=use_char_feature, 
                                        maxlen=maxlen, 
                                        fold_number=fold_number, 
                                        batch_size=batch_size,
                                        transformer_name=self.transformer_name)

        self.training_config = TrainingConfig(batch_size=batch_size, 
                                              optimizer=optimizer, 
                                              learning_rate=learning_rate,
                                              lr_decay=lr_decay, 
                                              clip_gradients=clip_gradients, 
                                              max_epoch=max_epoch,
                                              patience=patience, 
                                              use_roc_auc=use_roc_auc, 
                                              early_stop=early_stop,
                                              class_weights=class_weights, 
                                              multiprocessing=multiprocessing)

    def train(self, x_train, y_train, vocab_init=None, callbacks=None):
        self.model = getModel(self.model_config, self.training_config)

        print_parameters(self.model_config, self.training_config)
        self.model.print_summary()

        bert_data = False
        if self.transformer_name is not None:
            bert_data = True

        if self.training_config.early_stop:
            # create validation set 
            xtr, val_x, y, val_y = train_test_split(x_train, y_train, test_size=0.1)

            training_generator = DataGenerator(xtr, y, batch_size=self.training_config.batch_size, 
                maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                embeddings=self.embeddings, shuffle=True, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)
            validation_generator = DataGenerator(val_x, None, batch_size=self.training_config.batch_size, 
                maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                embeddings=self.embeddings, shuffle=False, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)
        else:
            val_y = y_train

            training_generator = DataGenerator(x_train, y_train, batch_size=self.training_config.batch_size, 
                maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                embeddings=self.embeddings, shuffle=True, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)
            validation_generator = None

        # uncomment to plot graph
        #plot_model(self.model, 
        #    to_file='data/models/textClassification/'+self.model_config.model_name+'_'+self.model_config.architecture+'.png')
        self.model.train_model(
            self.model_config.list_classes, 
            self.training_config.batch_size, 
            self.training_config.max_epoch, 
            self.training_config.use_roc_auc, 
            self.training_config.class_weights, 
            training_generator, 
            validation_generator, 
            val_y, 
            patience=self.training_config.patience, 
            multiprocessing=self.training_config.multiprocessing, 
            callbacks=callbacks)


    def train_nfold(self, x_train, y_train, vocab_init=None, callbacks=None):
        self.models = train_folds(x_train, y_train, self.model_config, self.training_config, self.embeddings, callbacks=callbacks)


    def predict(self, texts, output_format='json', use_main_thread_only=False, batch_size=None):
        bert_data = False
        if self.transformer_name != None:
            bert_data = True

        if batch_size != None:
            self.model_config.batch_size = batch_size
            print("---")
            print("batch_size (prediction):", self.model_config.batch_size)
            print("---")

        if self.model_config.fold_number == 1:
            if self.model != None: 
                
                predict_generator = DataGenerator(texts, None, batch_size=self.model_config.batch_size, 
                    maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                    embeddings=self.embeddings, shuffle=False, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)

                result = self.model.predict(predict_generator, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find a model.'))
        else:            
            if self.models != None: 

                # just a warning: n classifiers using BERT layer for prediction might be heavy in term of model sizes 
                predict_generator = DataGenerator(texts, None, batch_size=self.model_config.batch_size, 
                    maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                    embeddings=self.embeddings, shuffle=False, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)

                result = predict_folds(self.models, 
                                       predict_generator, 
                                       self.model_config, 
                                       self.training_config, 
                                       use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find nfolds models.'))
        if output_format == 'json':
            res = {
                "software": "DeLFT",
                "date": datetime.datetime.now().isoformat(),
                "model": self.model_config.model_name,
                "classifications": []
            }
            i = 0
            for text in texts:
                classification = {
                    "text": text
                }
                the_res = result[i]
                j = 0
                for cl in self.model_config.list_classes:
                    classification[cl] = float(the_res[j])
                    j += 1
                res["classifications"].append(classification)
                i += 1
            return res
        else:
            return result

    def eval(self, x_test, y_test, use_main_thread_only=False):
        print_parameters(self.model_config, self.training_config)

        bert_data = False
        if self.transformer_name is not None:
            bert_data = True

        if self.model_config.fold_number == 1:
            if self.model != None:
                self.model.print_summary()
                test_generator = DataGenerator(x_test, None, batch_size=self.model_config.batch_size,
                        maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                        embeddings=self.embeddings, shuffle=False, bert_data=bert_data, transformer_tokenizer=self.model.transformer_tokenizer)

                result = self.model.predict(test_generator, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find a model.'))
        else:
            if self.models is None:
                raise (OSError('Could not find nfolds models.'))

            self.models[0].print_summary()

            # just a warning: n classifiers using BERT layer for prediction might be heavy in term of model sizes
            test_generator = DataGenerator(x_test, None, batch_size=self.model_config.batch_size,
                maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes,
                embeddings=self.embeddings, shuffle=False, bert_data=bert_data, transformer_tokenizer=self.models[0].transformer_tokenizer)
            result = predict_folds(self.models, test_generator, self.model_config, self.training_config, use_main_thread_only=use_main_thread_only)

        print("-----------------------------------------------")
        print("\nEvaluation on", x_test.shape[0], "instances:")

        total_accuracy = 0.0
        total_f1 = 0.0
        total_loss = 0.0
        total_roc_auc = 0.0

        '''
        def normer(t):
            if t < 0.5: 
                return 0 
            else: 
                return 1
        vfunc = np.vectorize(normer)
        result_binary = vfunc(result)
        '''
        result_intermediate = np.asarray([np.argmax(line) for line in result])
        
        def vectorize(index, size):
            result = np.zeros(size)
            if index < size:
                result[index] = 1
            return result
        result_binary = np.array([vectorize(xi, len(self.model_config.list_classes)) for xi in result_intermediate])

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, result_binary, average=None)
        print('{:>14}  {:>12}  {:>12}  {:>12}  {:>12}'.format(" ", "precision", "recall", "f-score", "support"))
        p = 0
        for the_class in self.model_config.list_classes:
            the_class = the_class[:14]
            print('{:>14}  {:>12}  {:>12}  {:>12}  {:>12}'.format(the_class, "{:10.4f}"
                .format(precision[p]), "{:10.4f}".format(recall[p]), "{:10.4f}".format(fscore[p]), support[p]))
            p += 1

        # macro-average (average of class scores)
        # we distinguish 1-class and multiclass problems 
        if len(self.model_config.list_classes) == 1:
            total_accuracy = accuracy_score(y_test, result_binary)
            total_f1 = f1_score(y_test, result_binary)

            # sklearn will complain if log(0)
            total_loss = log_loss(y_test, result, labels=[0,1])
            if len(np.unique(y_test)) == 1:
                # roc_auc_score sklearn implementation is not working in this case, it needs more balanced batches
                # a simple fix is to return the r2_score instead in this case (which is a regression score and not a loss)
                total_roc_auc = r2_score(y_test, result)
                if total_roc_auc < 0:
                    total_roc_auc = 0 
            else:
                total_roc_auc = roc_auc_score(y_test, result)
        else:
            for j in range(0, len(self.model_config.list_classes)):
                accuracy = accuracy_score(y_test[:, j], result_binary[:, j])
                total_accuracy += accuracy
                f1 = f1_score(y_test[:, j], result_binary[:, j], average='micro')
                total_f1 += f1
                loss = log_loss(y_test[:, j], result[:, j], labels=[0,1])
                total_loss += loss
                if len(np.unique(y_test[:, j])) == 1:
                    # roc_auc_score sklearn implementation is not working in this case, it needs more balanced batches
                    # a simple fix is to return the r2_score instead in this case (which is a regression score and not a loss)
                    roc_auc = r2_score(y_test[:, j], result[:, j])
                    if roc_auc < 0:
                        roc_auc = 0 
                else:
                    roc_auc = roc_auc_score(y_test[:, j], result[:, j], labels=[0,1])
                total_roc_auc += roc_auc
                '''
                print("\nClass:", self.model_config.list_classes[j])
                print("\taccuracy at 0.5 =", accuracy)
                print("\tf-1 at 0.5 =", f1)
                print("\tlog-loss =", loss)
                print("\troc auc =", roc_auc)
                '''

        total_accuracy /= len(self.model_config.list_classes)
        total_f1 /= len(self.model_config.list_classes)
        total_loss /= len(self.model_config.list_classes)
        total_roc_auc /= len(self.model_config.list_classes)

        '''
        if len(self.model_config.list_classes) != 1:
            print("\nMacro-average:")
        print("\taverage accuracy at 0.5 =", "{:10.4f}".format(total_accuracy))
        print("\taverage f-1 at 0.5 =", "{:10.4f}".format(total_f1))
        print("\taverage log-loss =","{:10.4f}".format( total_loss))
        print("\taverage roc auc =", "{:10.4f}".format(total_roc_auc))
        '''
        
        # micro-average (average of scores for each instance)
        # make sense only if we have more than 1 class, otherwise same as 
        # macro-avergae
        if len(self.model_config.list_classes) != 1:
            total_accuracy = 0.0
            total_f1 = 0.0
            total_loss = 0.0
            total_roc_auc = 0.0

            for i in range(0, result.shape[0]):
                accuracy = accuracy_score(y_test[i,:], result_binary[i,:])
                total_accuracy += accuracy
                f1 = f1_score(y_test[i,:], result_binary[i,:], average='micro')
                total_f1 += f1
                loss = log_loss(y_test[i,:], result[i,:], labels=[0.0, 1.0])
                total_loss += loss
                if len(np.unique(y_test[i,:])) == 1:
                    # roc_auc_score sklearn implementation is not working in this case, it needs more balanced batches
                    # a simple fix is to return the r2_score instead in this case (which is a regression score and not a loss)
                    roc_auc = r2_score(y_test[i,:], result[i,:])
                    if roc_auc < 0:
                        roc_auc = 0 
                else:
                    roc_auc = roc_auc_score(y_test[i,:], result[i,:], labels=[0.0, 1.0])
                total_roc_auc += roc_auc

            total_accuracy /= result.shape[0]
            total_f1 /= result.shape[0]
            total_loss /= result.shape[0]
            total_roc_auc /= result.shape[0]
            
            '''
            print("\nMicro-average:")
            print("\taverage accuracy at 0.5 =", "{:10.4f}".format(total_accuracy))
            print("\taverage f-1 at 0.5 =", "{:10.4f}".format(total_f1))
            print("\taverage log-loss =", "{:10.4f}".format(total_loss))
            print("\taverage roc auc =", "{:10.4f}".format(total_roc_auc))
            '''
            
    def save(self, dir_path='data/models/textClassification/'):
        # create subfolder for the model if not already exists
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, self.config_file))
        print('model config file saved')

        if self.model_config.fold_number == 1:
            if self.model != None:
                self.model.save(os.path.join(directory, self.weight_file))
                print('model saved')
            else:
                print('Error: model has not been built')
        else:
            if self.models == None:
                print('Error: nfolds models have not been built')
            else:
                # fold models having a transformer layers are already saved
                if self.model_config.transformer_name is None:
                    for i in range(0, self.model_config.fold_number):
                        self.models[i].save(os.path.join(directory, "model{0}_weights.hdf5".format(i)))
                    print('nfolds model saved')

        # save pretrained transformer config and tokenizer if used in the model and if single fold (otherwise it is saved in the nfold process)
        if self.transformer_name is not None and self.model_config.fold_number == 1:
            if self.model.transformer_config is not None:
                self.model.transformer_config.to_json_file(os.path.join(directory, TRANSFORMER_CONFIG_FILE_NAME))
            if self.model.transformer_tokenizer is not None:
                self.model.transformer_tokenizer.save_pretrained(os.path.join(directory, DEFAULT_TRANSFORMER_TOKENIZER_DIR))


    def load(self, dir_path='data/models/textClassification/'):
        model_path = os.path.join(dir_path, self.model_config.model_name)
        self.model_config = ModelConfig.load(os.path.join(model_path, self.config_file))

        if self.model_config.transformer_name is None:
            # load embeddings
            # Do not use cache in 'production' mode
            self.embeddings = Embeddings(self.model_config.embeddings_name, resource_registry=self.registry, use_cache=False)
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.transformer_name = self.model_config.transformer_name
            self.embeddings = None

        self.model = getModel(self.model_config, 
                              self.training_config, 
                              load_pretrained_weights=False, 
                              local_path=model_path)
        print_parameters(self.model_config, self.training_config)
        self.model.print_summary()

        if self.model_config.fold_number == 1:
            print("load weights from", os.path.join(model_path, self.weight_file))
            self.model.load(os.path.join(model_path, self.weight_file))
        else:
            self.models = []
            if self.model_config.transformer_name is None:
                for i in range(0, self.model_config.fold_number):
                    local_model = getModel(self.model_config, 
                                        self.training_config, 
                                        load_pretrained_weights=False, 
                                        local_path=model_path)
                    local_model.load(os.path.join(model_path, "model{0}_weights.hdf5".format(i)))
                    self.models.append(local_model)
            else:
                # only init first fold one, the other will be init at prediction time, all weights will be loaded at prediction time
                local_model = getModel(self.model_config, 
                                    self.training_config, 
                                    load_pretrained_weights=False, 
                                    local_path=model_path)
                self.models.append(local_model)
