import os

import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)

import datetime

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import getModel
from delft.textClassification.models import train_model
from delft.textClassification.models import train_folds
from delft.textClassification.models import predict
from delft.textClassification.models import predict_folds
from delft.textClassification.models import BERT_classifier
from delft.textClassification.data_generator import DataGenerator
from delft.textClassification.preprocess import to_vector_single, BERT_classifier_processor

from delft.utilities.Embeddings import Embeddings

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, r2_score, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras.utils import plot_model


class Classifier(object):

    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'

    def __init__(self, 
                 model_name="",
                 model_type="gru",
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
                 use_ELMo=False,
                 use_BERT=False,
                 embeddings=(),
                 class_weights=None):
        self.model = None
        self.models = None
        self.log_dir = log_dir
        self.embeddings_name = embeddings_name

        word_emb_size = 0
        if embeddings_name is not None:
            self.embeddings = Embeddings(embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT) 
            word_emb_size = self.embeddings.embed_size

        self.model_config = ModelConfig(model_name=model_name, 
                                        model_type=model_type, 
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
                                        use_ELMo=use_ELMo, 
                                        use_BERT=use_BERT)

        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              patience, use_roc_auc,
                                              class_weights=class_weights)

    def train(self, x_train, y_train, vocab_init=None):
        self.model = getModel(self.model_config, self.training_config)

        # bert models
        if self.model_config.model_type.find("bert") != -1:     
            self.model.processor = BERT_classifier_processor(labels=self.model_config.list_classes, x_train=x_train, y_train=y_train)
            self.model.train()
            return

        # create validation set in case we don't use k-folds
        xtr, val_x, y, val_y = train_test_split(x_train, y_train, test_size=0.1)

        training_generator = DataGenerator(xtr, y, batch_size=self.training_config.batch_size, 
            maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
            embeddings=self.embeddings, shuffle=True)
        validation_generator = DataGenerator(val_x, None, batch_size=self.training_config.batch_size, 
            maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
            embeddings=self.embeddings, shuffle=False)
        
        # uncomment to plot graph
        #plot_model(self.model, 
        #    to_file='data/models/textClassification/'+self.model_config.model_name+'_'+self.model_config.model_type+'.png')
        self.model, best_roc_auc = train_model(self.model, self.model_config.list_classes, self.training_config.batch_size, 
            self.training_config.max_epoch, self.training_config.use_roc_auc, self.training_config.class_weights, 
            training_generator, validation_generator, val_y, use_ELMo=self.embeddings.use_ELMo, 
            use_BERT=self.embeddings.use_BERT)
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

    def train_nfold(self, x_train, y_train, vocab_init=None):
        self.models = train_folds(x_train, y_train, self.model_config, self.training_config, self.embeddings)
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()

    # classification
    def predict(self, texts, output_format='json', use_main_thread_only=False):
        if self.model_config.fold_number is 1:
            if self.model is not None:
                # bert model?
                if self.model_config.model_type.find("bert") != -1:
                    # be sure the input processor is instanciated
                    self.model.processor = BERT_classifier_processor(labels=self.model_config.list_classes)
                    result = self.model.predict(texts)
                else:
                    predict_generator = DataGenerator(texts, None, batch_size=self.model_config.batch_size, 
                        maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                        embeddings=self.embeddings, shuffle=False)

                    result = predict(self.model, predict_generator, use_ELMo=self.embeddings.use_ELMo, use_BERT=self.embeddings.use_BERT, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find a model.'))
        else:
            if self.models is not None:
                # bert model?
                if self.model_config.model_type.find("bert") != -1:
                    # we don't support n classifiers for BERT (would be too large)
                    # be sure the input processor is instanciated
                    self.model.processor = BERT_classifier_processor(labels=self.model_config.list_classes)
                    result = self.models[0].predict(texts)
                else:    
                    predict_generator = DataGenerator(texts, None, batch_size=self.model_config.batch_size, 
                        maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                        embeddings=self.embeddings, shuffle=False)

                    result = predict_folds(self.models, predict_generator, use_ELMo=self.embeddings.use_ELMo, use_BERT=self.embeddings.use_BERT, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find nfolds models.'))
        if output_format is 'json':
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
        if self.model_config.fold_number is 1:
            if self.model is not None:
                # bert model?
                if self.model_config.model_type.find("bert") != -1:
                    self.model.eval(x_test, y_test)
                    result = self.model.predict(x_test)
                else:
                    test_generator = DataGenerator(x_test, None, batch_size=self.model_config.batch_size, 
                        maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                        embeddings=self.embeddings, shuffle=False)

                    result = predict(self.model, test_generator, use_ELMo=self.embeddings.use_ELMo, use_BERT=self.embeddings.use_BERT, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find a model.'))
        else:
            if self.models is not None:
                # bert model?
                if self.model_config.model_type.find("bert") != -1:
                    result = self.models[0].eval(x_test, y_test)

                test_generator = DataGenerator(x_test, None, batch_size=self.model_config.batch_size, 
                    maxlen=self.model_config.maxlen, list_classes=self.model_config.list_classes, 
                    embeddings=self.embeddings, shuffle=False)

                result = predict_folds(self.models, test_generator, use_ELMo=self.embeddings.use_ELMo, use_BERT=self.embeddings.use_BERT, use_main_thread_only=use_main_thread_only)
            else:
                raise (OSError('Could not find nfolds models.'))
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
        if len(self.model_config.list_classes) is 1:
            total_accuracy = accuracy_score(y_test, result_binary)
            total_f1 = f1_score(y_test, result_binary)
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
                    roc_auc = roc_auc_score(y_test[:, j], result[:, j])
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
        if len(self.model_config.list_classes) is not 1:
            print("\nMacro-average:")
        print("\taverage accuracy at 0.5 =", "{:10.4f}".format(total_accuracy))
        print("\taverage f-1 at 0.5 =", "{:10.4f}".format(total_f1))
        print("\taverage log-loss =","{:10.4f}".format( total_loss))
        print("\taverage roc auc =", "{:10.4f}".format(total_roc_auc))
        '''
        
        # micro-average (average of scores for each instance)
        # make sense only if we have more than 1 class, otherwise same as 
        # macro-avergae
        if len(self.model_config.list_classes) is not 1:
            total_accuracy = 0.0
            total_f1 = 0.0
            total_loss = 0.0
            total_roc_auc = 0.0

            for i in range(0, result.shape[0]):
                accuracy = accuracy_score(y_test[i,:], result_binary[i,:])
                total_accuracy += accuracy
                f1 = f1_score(y_test[i,:], result_binary[i,:], average='micro')
                total_f1 += f1
                loss = log_loss(y_test[i,:], result[i,:])
                total_loss += loss
                roc_auc = roc_auc_score(y_test[i,:], result[i,:])
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

        # bert model are always saved via training process steps
        if self.model_config.model_type.find("bert") != -1:
            print('model saved')
            return

        if self.model_config.fold_number is 1:
            if self.model is not None:
                self.model.save(os.path.join(directory, self.model_config.model_type+"."+self.weight_file))
                print('model saved')
            else:
                print('Error: model has not been built')
        else:
            if self.models is None:
                print('Error: nfolds models have not been built')
            else:
                for i in range(0, self.model_config.fold_number):
                    self.models[i].save(os.path.join(directory, self.model_config.model_type+".model{0}_weights.hdf5".format(i)))
                print('nfolds model saved')

    def load(self, dir_path='data/models/textClassification/'):
        self.model_config = ModelConfig.load(os.path.join(dir_path, self.model_config.model_name, self.config_file))

        if self.model_config.model_type.find("bert") != -1:
             self.model = getModel(self.model_config, self.training_config)
             self.model.load()

        # load embeddings
        self.embeddings = Embeddings(self.model_config.embeddings_name, use_ELMo=self.model_config.use_ELMo, use_BERT=self.model_config.use_BERT) 
        self.model_config.word_embedding_size = self.embeddings.embed_size

        self.model = getModel(self.model_config, self.training_config)
        if self.model_config.fold_number is 1:
            self.model.load_weights(os.path.join(dir_path, self.model_config.model_name, self.model_config.model_type+"."+self.weight_file))
        else:
            self.models = []
            for i in range(0, self.model_config.fold_number):
                local_model = getModel(self.model_config, self.training_config)
                local_model.load_weights(os.path.join(dir_path, self.model_config.model_name, self.model_config.model_type+".model{0}_weights.hdf5".format(i)))
                self.models.append(local_model)
