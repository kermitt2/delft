import numpy as np
import sys, os
import argparse
import math
import json
import time
import shutil

from delft.textClassification.data_generator import DataGenerator

from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Input, InputLayer, concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, SpatialDropout1D, AveragePooling1D, GlobalAveragePooling1D, TimeDistributed, Masking, Lambda 
from tensorflow.keras.layers import GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation, Add, Flatten, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, schedules
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, precision_recall_fscore_support

from transformers import AutoConfig, TFAutoModel
from transformers import create_optimizer

TRANSFORMER_CONFIG_FILE_NAME = 'transformer-config.json'

architectures = [
    'lstm',
    'bidLstm_simple',
    'cnn',
    'cnn2',
    'cnn3',
    'gru_lstm',
    'dpcnn',
    'conv',
    "gru",
    "gru_simple",
    'lstm_cnn',
    'han',
    'bert'
]


def getModel(model_config, training_config, load_pretrained_weights=True, local_path=None):
    architecture = model_config.architecture

    # awww Python has no case/switch statement :D
    if (architecture == 'bidLstm_simple'):
        model = bidLstm_simple(model_config, training_config)
    elif (architecture == 'lstm'):
        model = lstm(model_config, training_config)
    elif (architecture == 'cnn'):
        model = cnn(model_config, training_config)
    elif (architecture == 'cnn2'):
        model = cnn2(model_config, training_config)
    elif (architecture == 'cnn3'):
        model = cnn3(model_config, training_config)
    elif (architecture == 'lstm_cnn'):
        model = lstm_cnn(model_config, training_config)
    elif (architecture == 'conv'):
        model = dpcnn(model_config, training_config)
    elif (architecture == 'gru_lstm'):
        model = gru_lstm(model_config, training_config)
    elif (architecture == 'dpcnn'):
        model = dpcnn(model_config, training_config)
    elif (architecture == 'gru'):
        model = gru(model_config, training_config)
    elif (architecture == 'gru_simple'):
        model = gru_simple(model_config, training_config)
    elif (architecture == 'bert'):
        print(model_config.transformer, "will be used")
        model = bert(model_config, training_config,
                    load_pretrained_weights=load_pretrained_weights,
                    local_path=local_path)
    else:
        raise (OSError('The model type '+architecture+' is unknown'))
    return model


class BaseModel(object):

    transformer_config = None
    parameters = {}

    def __init__(self, model_config, training_config, load_pretrained_weights=True, local_path=None):
        """
        Base class for DeLFT text classification models

        Args:
            config (ModelConfig): DeLFT model configuration object
            ntags (integer): number of classes of the model
            load_pretrained_weights (boolean): used only when the model contains a transformer layer - indicate whether 
                                               or not we load the pretrained weights of this transformer. For training
                                               a new model set it to True. When getting the full Keras model to load
                                               existing weights, set it False to avoid reloading the pretrained weights. 
            local_path (string): used only when the model contains a transformer layer - the path where to load locally the 
                                 pretrained transformer. If None, the transformer model will be fetched from HuggingFace 
                                 transformers hub.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model = None

    def update_parameters(self, model_config, training_config):
        for key in self.parameters:
            if hasattr(model_config, key):
                self.parameters[key] = getattr(model_config, key)
            elif hasattr(training_config, key):
                self.parameters[key] = getattr(training_config, key)

    def train_model(self, 
                list_classes, 
                batch_size, 
                max_epoch, 
                use_roc_auc, 
                class_weights, 
                training_generator, 
                validation_generator, 
                val_y, 
                multiprocessing=True, 
                patience=5,
                callbacks=None):

        nb_train_steps = (len(val_y) // batch_size) * max_epoch
        self.compile(nb_train_steps)

        best_loss = -1
        best_roc_auc = -1

        if validation_generator == None:
            # no early stop
            nb_workers = 6
            best_loss = self.model.fit(
                training_generator,
                use_multiprocessing=multiprocessing,
                workers=nb_workers,
                class_weight=class_weights,
                epochs=max_epoch, callbacks=callbacks)
        else:
            best_weights = None
            current_epoch = 1
            best_epoch = 0
            while current_epoch <= max_epoch:

                nb_workers = 6
                loss = self.model.fit(
                    training_generator,
                    use_multiprocessing=multiprocessing,
                    workers=nb_workers,
                    class_weight=class_weights,
                    epochs=1, callbacks=callbacks)

                y_pred = self.model.predict(
                    validation_generator, 
                    use_multiprocessing=multiprocessing,
                    workers=nb_workers)

                total_loss = 0.0
                total_roc_auc = 0.0

                # we distinguish 1-class and multiclass problems 
                if len(list_classes) == 1:
                    total_loss = log_loss(val_y, y_pred, labels=[0,1])
                    if len(np.unique(val_y)) == 1:
                        # roc_auc_score sklearn implementation is not working in this case, it needs more balanced batches
                        # a simple fix is to return the r2_score instead in this case (which is a regression score and not a loss)
                        roc_auc = r2_score(val_y, y_pred)
                        if roc_auc < 0:
                            roc_auc = 0 
                    else:
                        total_roc_auc = roc_auc_score(val_y, y_pred)
                else:
                    for j in range(0, len(list_classes)):
                        loss = log_loss(val_y[:, j], y_pred[:, j], labels=[0,1])
                        total_loss += loss
                        if len(np.unique(val_y[:, j])) == 1:
                            # roc_auc_score sklearn implementation is not working in this case, it needs more balanced batches
                            # a simple fix is to return the r2_score instead in this case (which is a regression score and not a loss)
                            roc_auc = r2_score(val_y[:, j], y_pred[:, j])
                            if roc_auc < 0:
                                roc_auc = 0 
                        else:
                            roc_auc = roc_auc_score(val_y[:, j], y_pred[:, j])
                        total_roc_auc += roc_auc

                total_loss /= len(list_classes)
                total_roc_auc /= len(list_classes)
                if use_roc_auc:
                    print("Epoch {0} loss {1} best_loss {2} (for info) ".format(current_epoch, total_loss, best_loss))
                    print("Epoch {0} roc_auc {1} best_roc_auc {2} (for early stop) ".format(current_epoch, total_roc_auc, best_roc_auc))
                else:
                    print("Epoch {0} loss {1} best_loss {2} (for early stop) ".format(current_epoch, total_loss, best_loss))
                    print("Epoch {0} roc_auc {1} best_roc_auc {2} (for info) ".format(current_epoch, total_roc_auc, best_roc_auc))

                current_epoch += 1
                if total_loss < best_loss or best_loss == -1 or math.isnan(best_loss) == True:
                    best_loss = total_loss
                    if use_roc_auc == False:
                        best_weights = self.model.get_weights()
                        best_epoch = current_epoch
                elif use_roc_auc == False:
                    if current_epoch - best_epoch == patience:
                        break

                if total_roc_auc > best_roc_auc or best_roc_auc == -1:
                    best_roc_auc = total_roc_auc
                    if use_roc_auc:
                        best_weights = self.model.get_weights()
                        best_epoch = current_epoch
                elif use_roc_auc:
                    if current_epoch - best_epoch == patience:
                        break

            self.model.set_weights(best_weights)


    def predict(self, predict_generator, use_main_thread_only=False):
        nb_workers = 6
        multiprocessing = True
        y = self.model.predict(
                predict_generator, 
                use_multiprocessing=multiprocessing,
                workers=nb_workers)
        return y

    def compile(self, train_size):
        # default compilation of the model. 
        # train_size gives the number of steps for the traning, to be used for learning rate scheduler/decay
        self.model.compile(loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])

    def instanciate_transformer_layer(self, transformer_model_name, load_pretrained_weights=True, local_path=None):
        if load_pretrained_weights:
            if local_path is None:
                transformer_model = TFAutoModel.from_pretrained(transformer_model_name, from_pt=True)
            else:
                transformer_model = TFAutoModel.from_pretrained(local_path, from_pt=True)
            self.bert_config = transformer_model.config
        else:
            # load config in JSON format
            if local_path is None:
                self.bert_config = AutoConfig.from_pretrained(transformer_model_name)
            else:
                config_path = os.path.join(".", local_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.bert_config = AutoConfig.from_pretrained(config_path)
            transformer_model = TFAutoModel.from_config(self.bert_config)
        return transformer_model

    def get_transformer_config(self):
        # transformer config (PretrainedConfig) if a pretrained transformer is used in the model
        return None

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        print('loading model weights', filepath)
        self.model.load_weights(filepath=filepath)


def train_folds(X, y, model_config, training_config, embeddings, tokenizer, callbacks=None):
    fold_count = model_config.fold_number
    max_epoch = training_config.max_epoch
    architecture = model_config.architecture
    use_roc_auc = training_config.use_roc_auc
    class_weights = training_config.class_weights

    fold_size = len(X) // fold_count
    models = []
    scores = []

    bert_data = False
    if model_config.transformer != None:
        bert_data = True

    for fold_id in range(0, fold_count):
        print('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        training_generator = DataGenerator(train_x, train_y, batch_size=training_config.batch_size, 
            maxlen=model_config.maxlen, list_classes=model_config.list_classes, 
            embeddings=embeddings, bert_data=bert_data, shuffle=True, tokenizer=tokenizer)

        validation_generator = None
        if training_config.early_stop:
            validation_generator = DataGenerator(val_x, val_y, batch_size=training_config.batch_size, 
                maxlen=model_config.maxlen, list_classes=model_config.list_classes, 
                embeddings=embeddings, bert_data=bert_data, shuffle=False, tokenizer=tokenizer)

        foldModel = getModel(model_config, training_config)

        foldModel.train_model(model_config.list_classes, training_config.batch_size, max_epoch, use_roc_auc, 
                class_weights, training_generator, validation_generator, val_y, multiprocessing=training_config.multiprocessing, 
                patience=training_config.patience, callbacks=callbacks)
        
        if model_config.transformer is None:
            models.append(foldModel)
        else:
            # if we are using a transformer layer in the architecture, we need to save the fold model on the disk
            directory = os.path.join("data/models/textClassification/", model_config.model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            if fold_id == 0:
                models.append(foldModel)
                # save transformer config
                if foldModel.get_transformer_config() is not None:
                    foldModel.get_transformer_config().to_json_file(os.path.join(directory, TRANSFORMER_CONFIG_FILE_NAME))

            model_path = os.path.join("data/models/textClassification/", model_config.model_name, model_config.architecture+".model{0}_weights.hdf5".format(fold_id))
            foldModel.save(model_path)
            if fold_id != 0:
                del foldModel

    return models


def predict_folds(models, predict_generator, model_config, training_config, use_main_thread_only=False):
    fold_count = model_config.fold_number
    y_predicts_list = []
    for fold_id in range(0, fold_count):

        if model_config.transformer is not None:
            model = models[0]
            #if fold_id != 0:
            # load new weight from disk
            model_path = os.path.join("data/models/textClassification/", model_config.model_name, model_config.architecture+".model{0}_weights.hdf5".format(fold_id))
            model.load(model_path)  
        else:
            model = models[fold_id]
        
        nb_workers = 6
        y_predicts = model.predict(predict_generator, use_main_thread_only=use_main_thread_only)
        y_predicts_list.append(y_predicts)

    y_predicts = np.ones(y_predicts_list[0].shape)
    for fold_predict in y_predicts_list:
        y_predicts *= fold_predict

    y_predicts **= (1. / len(y_predicts_list))

    return y_predicts    


class lstm(BaseModel):
    """
    A Keras implementation of a LSTM classifier
    """
    name = 'lstm'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 40,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        # basic LSTM
        #def lstm(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):

        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = LSTM(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"])(input_layer)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b])
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()
        

class bidLstm_simple(BaseModel):
    """
    A Keras implementation of a bidirectional LSTM  classifier
    """
    name = 'bidLstm_simple'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 25,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 300,
        'dense_size': 256
    }

    # bidirectional LSTM 
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def bidLstm_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):

        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Bidirectional(LSTM(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"]))(input_layer)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b])
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()


class cnn(BaseModel):
    """
    A Keras implementation of a CNN classifier
    """
    name = 'cnn'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 250,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # conv+GRU with embeddings
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)
        #def cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Dropout(self.parameters["dropout_rate"])(input_layer) 
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GRU(self.parameters["recurrent_units"])(x)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()  


class cnn2(BaseModel):
    """
    A Keras implementation of a CNN classifier (variant)
    """
    name = 'cnn2'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 250,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def cnn2(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Dropout(self.parameters["dropout_rate"])(input_layer) 
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = GRU(self.parameters["recurrent_units"], return_sequences=False, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"])(x)
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()  


class cnn3(BaseModel):
    """
    A Keras implementation of a CNN classifier (variant)
    """
    name = 'cnn3'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def cnn3(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = GRU(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"])(input_layer)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b])
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()  


class conv(BaseModel):
    """
    A Keras implementation of a multiple level Convolutional classifier (variant)
    """
    name = 'conv'

    # default parameters 
    parameters_conv = {
        'max_features': 200000,
        'maxlen': 250,
        'embed_size': 300,
        'epoch': 25,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 256,
        'dense_size': 64
    }

    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def conv(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        filter_kernels = [7, 7, 5, 5, 3, 3]
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        conv = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[0], border_mode='valid', activation='relu')(input_layer)
        conv = MaxPooling1D(pool_length=3)(conv)
        conv1 = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[1], border_mode='valid', activation='relu')(conv)
        conv1 = MaxPooling1D(pool_length=3)(conv1)
        conv2 = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[2], border_mode='valid', activation='relu')(conv1)
        conv3 = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[3], border_mode='valid', activation='relu')(conv2)
        conv4 = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[4], border_mode='valid', activation='relu')(conv3)
        conv5 = Conv1D(nb_filter=self.parameters["recurrent_units"], filter_length=filter_kernels[5], border_mode='valid', activation='relu')(conv4)
        conv5 = MaxPooling1D(pool_length=3)(conv5)
        conv5 = Flatten()(conv5)
        z = Dropout(0.5)(Dense(self.parameters["dense_size"], activation='relu')(conv5))
        x = Dense(nb_classes, activation="sigmoid")(z)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()  


class lstm_cnn(BaseModel):
    """
    A Keras implementation of a LSTM + CNN classifier
    """
    name = 'lstm_cnn'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 250,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # LSTM + conv
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def lstm_cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = LSTM(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"])(input_layer)
        x = Dropout(self.parameters["dropout_rate"])(x)

        x = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, padding='same', activation='relu')(x)
        x = Conv1D(filters=300,
                           kernel_size=5,
                           padding='valid',
                           activation='tanh',
                           strides=1)(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b])
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.summary()


class gru(BaseModel):
    """
    A Keras implementation of a Bidirectional GRU classifier
    """
    name = 'gru'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # 2 bid. GRU 
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def gru(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Bidirectional(GRU(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["recurrent_dropout_rate"]))(input_layer)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Bidirectional(GRU(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["recurrent_dropout_rate"]))(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b], axis=1)
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        output_layer = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        
    def compile(self, train_size):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(clipvalue=1, clipnorm=1),
                      metrics=['accuracy'])


class gru_simple(BaseModel):
    """
    A Keras implementation of a one layer Bidirectional GRU classifier
    """
    name = 'gru_simple'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 512,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # 1 layer bid GRU
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def gru_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Bidirectional(GRU(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["dropout_rate"]))(input_layer)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b], axis=1)
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        output_layer = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

    def compile(self, train_size):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(clipvalue=1, clipnorm=1),
                      metrics=['accuracy'])


class gru_lstm(BaseModel):
    """
    A Keras implementation of a mixed Bidirectional GRU and LSTM classifier
    """
    name = 'gru_lstm'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 30,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # bid GRU + bid LSTM
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def mix1(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        x = Bidirectional(GRU(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["recurrent_dropout_rate"]))(input_layer)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = Bidirectional(LSTM(self.parameters["recurrent_units"], return_sequences=True, dropout=self.parameters["dropout_rate"],
                               recurrent_dropout=self.parameters["recurrent_dropout_rate"]))(x)
        x_a = GlobalMaxPool1D()(x)
        x_b = GlobalAveragePooling1D()(x)
        x = concatenate([x_a,x_b])
        x = Dense(self.parameters["dense_size"], activation="relu")(x)
        output_layer = Dense(nb_classes, activation="sigmoid")(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()

    def compile(self, train_size):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(clipvalue=1, clipnorm=1),
                      metrics=['accuracy'])


class dpcnn(BaseModel):
    """
    A Keras implementation of a DPCNN classifier
    """
    name = 'dpcnn'

    # default parameters 
    parameters = {
        'max_features': 200000,
        'maxlen': 300,
        'embed_size': 300,
        'epoch': 25,
        'batch_size': 256,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.3,
        'recurrent_units': 64,
        'dense_size': 32
    }

    # DPCNN
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def dpcnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
        input_layer = Input(shape=(self.parameters["maxlen"], self.parameters["embed_size"]), )
        # first block
        X_shortcut1 = input_layer
        X = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, strides=3)(X_shortcut1)
        X = Activation('relu')(X)
        X = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, strides=3)(X)
        X = Activation('relu')(X)

        # connect shortcut to the main path
        X = Activation('relu')(X_shortcut1)  # pre activation
        X = Add()([X_shortcut1,X])
        X = MaxPooling1D(pool_size=3, strides=2, padding='valid')(X)

        # second block
        X_shortcut2 = X
        X = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, strides=3)(X)
        X = Activation('relu')(X)
        X = Conv1D(filters=self.parameters["recurrent_units"], kernel_size=2, strides=3)(X)
        X = Activation('relu')(X)

        # connect shortcut to the main path
        X = Activation('relu')(X_shortcut2)  # pre activation
        X = Add()([X_shortcut2,X])
        X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)

        # Output
        X = Flatten()(X)
        X = Dense(nb_classes, activation='sigmoid')(X)

        self.model = Model(inputs = input_layer, outputs = X, name='dpcnn')
        self.model.summary()


class bert(BaseModel):
    """
    A Keras implementation of a BERT classifier for fine-tuning, with BERT layer to be 
    instanciated with a pre-trained BERT model
    """
    name = 'bert'
    bert_config = None 

    # default parameters 
    parameters = {
        'dense_size': 512,
        'max_seq_len': 512,
        'dropout_rate': 0.1,
        'batch_size': 10,
        'transformer': 'bert-base-en'
    }

    # simple BERT classifier with TF transformers, architecture equivalent to the original BERT implementation
    def __init__(self, model_config, training_config, load_pretrained_weights=True, local_path=None):
        self.model_config = model_config
        self.training_config = training_config
        self.update_parameters(model_config, training_config)
        nb_classes = len(model_config.list_classes)

        #def bert(dense_size, nb_classes, max_seq_len=512, transformer="bert-base-en", load_pretrained_weights=True, local_path=None):
        transformer_model_name = self.parameters["transformer"]
        #print(transformer_model_name)
        transformer_model = self.instanciate_transformer_layer(transformer_model_name, 
                                                          load_pretrained_weights=load_pretrained_weights, 
                                                          local_path=local_path)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        #input_masks_in = Input(shape=(None,), name='masked_token', dtype='int32')
        #embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[1]

        # get the pooler_output as in the original BERT implementation
        embedding_layer = transformer_model(input_ids_in)[1]
        cls_out = Dropout(0.1)(embedding_layer)
        logits = Dense(units=nb_classes, activation="softmax")(cls_out)

        self.model = Model(inputs=[input_ids_in], outputs=logits)
        #self.model = Model(inputs=[input_ids_in, input_masks_in], outputs=logits)
        self.model.summary()

    def compile(self, train_size):
        #optimizer = Adam(learning_rate=2e-5, clipnorm=1)
        optimizer, lr_schedule = create_optimizer(
                init_lr=2e-5, 
                num_train_steps=train_size,
                weight_decay_rate=0.01,
                num_warmup_steps=0.1*train_size,
            )
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

    def get_transformer_config(self):
        '''
        return the PretrainedConfig object of the transformer layer used in the mode, if any
        '''
        return self.bert_config

def _get_description(name, path="delft/resources-registry.json"):
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["embeddings-contextualized"]:
        if emb["name"] == name:
            return emb
    for emb in registry["transformers"]:
            if emb["name"] == name:
                return emb
    return None
