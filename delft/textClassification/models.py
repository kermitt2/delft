# Class for experimenting various single input DL models at word level
#
#   model_type      type of the model to be used into ['lstm', 'bidLstm', 'cnn', 'cudnngru', 'cudnnlstm']
#   fold_count      number of folds for k-fold training (default is 1)
#

import pandas as pd
import numpy as np
import pandas as pd
import sys, os
import argparse
import math
import json
import time
import shutil

from delft.textClassification.data_generator import DataGenerator

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import LSTM, Bidirectional, Dropout, SpatialDropout1D, AveragePooling1D, GlobalAveragePooling1D, TimeDistributed, Masking, Lambda 
from keras.layers import GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation, Add, Flatten, BatchNormalization
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, precision_recall_fscore_support

#import utilities.Attention
from delft.utilities.Attention import Attention
#from ToxicAttentionAlternative import AttentionAlternative
#from ToxicAttentionWeightedAverage import AttentionWeightedAverage

from delft.textClassification.preprocess import BERT_classifier_processor

from delft.utilities.bert.run_classifier_delft import *
import delft.utilities.bert.modeling as modeling
import delft.utilities.bert.optimization as optimization
import delft.utilities.bert.tokenization as tokenization

# seed is fixed for reproducibility
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(8)

modelTypes = ['lstm', 'bidLstm_simple', 'bidLstm', 'cnn', 'cnn2', 'cnn3', 'mix1', 'dpcnn', 
            'conv', "gru", "gru_simple", 'lstm_cnn', 'han', 'bert-base-en', 'scibert', 'biobert']

# default parameters of the different DL models
parameters_lstm = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 40,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'LSTM.csv'
}

parameters_bidLstm_simple = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 25,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 300,
    'dense_size': 256,
    'resultFile': 'BidLSTM_simple.csv'
}

parameters_bidLstm = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 25,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 300,
    'dense_size': 256,
    'resultFile': 'BidLSTM_attention.csv'
}

parameters_cnn = {
    'max_features': 200000,
    'maxlen': 250,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'CNN.csv'
}

parameters_cnn2 = {
    'max_features': 200000,
    'maxlen': 250,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'CNN2.csv'
}

parameters_cnn3 = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'CNN3.csv'
}

parameters_lstm_cnn = {
    'max_features': 200000,
    'maxlen': 250,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'LSTM_CNN.csv'
}

parameters_conv = {
    'max_features': 200000,
    'maxlen': 250,
    'embed_size': 300,
    'epoch': 25,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 256,
    'dense_size': 64,
    'resultFile': 'CNN.csv'
}

parameters_gru = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'GRU.csv'
}

parameters_gru_old = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 512,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'GRU.csv'
}

parameters_gru_simple = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 512,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'GRU_simple.csv'
}

parameters_mix1 = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 30,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'mix1.csv'
}

parameters_dpcnn = {
    'max_features': 200000,
    'maxlen': 300,
    'embed_size': 300,
    'epoch': 25,
    'batch_size': 256,
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.3,
    'recurrent_units': 64,
    'dense_size': 32,
    'resultFile': 'dpcnn.csv'
}

parametersMap = { 'lstm' : parameters_lstm, 'bidLstm_simple' : parameters_bidLstm_simple, 'bidLstm': parameters_bidLstm, 
                  'cnn': parameters_cnn, 'cnn2': parameters_cnn2, 'cnn3': parameters_cnn3, 'lstm_cnn': parameters_lstm_cnn,
                  'mix1': parameters_mix1, 'gru': parameters_gru, 'gru_simple': parameters_gru_simple, 
                  'dpcnn': parameters_dpcnn, 'conv': parameters_conv }


# basic LSTM
def lstm(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(inp)
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(input_layer)
    #x = CuDNNLSTM(recurrent_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    return model


# bidirectional LSTM 
def bidLstm_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])
    return model


# bidirectional LSTM with attention layer
def bidLstm(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(input_layer)
    #x = Dropout(dropout_rate)(x)
    x = Attention(maxlen)(x)
    #x = AttentionWeightedAverage(maxlen)(x)
    #print('len(x):', len(x))
    #x = AttentionWeightedAverage(maxlen)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# conv+GRU with embeddings
def cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn2_best(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units, return_sequences=False, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn2(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units, return_sequences=False, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn3(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(input_layer)
    #x = Dropout(dropout_rate)(x) 

    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def conv(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    filter_kernels = [7, 7, 5, 5, 3, 3]
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    conv = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[0], border_mode='valid', activation='relu')(input_layer)
    conv = MaxPooling1D(pool_length=3)(conv)
    conv1 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[1], border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)
    conv2 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[2], border_mode='valid', activation='relu')(conv1)
    conv3 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[3], border_mode='valid', activation='relu')(conv2)
    conv4 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[4], border_mode='valid', activation='relu')(conv3)
    conv5 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[5], border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)
    z = Dropout(0.5)(Dense(dense_size, activation='relu')(conv5))
    #x = GlobalMaxPool1D()(x)
    x = Dense(nb_classes, activation="sigmoid")(z)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# LSTM + conv
def lstm_cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(inp)
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(input_layer)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = Conv1D(filters=300,
                       kernel_size=5,
                       padding='valid',
                       activation='tanh',
                       strides=1)(x)
    #x = MaxPooling1D(pool_size=2)(x)

    #x = Conv1D(filters=300,
    #                   kernel_size=5,
    #                   padding='valid',
    #                   activation='tanh',
    #                   strides=1)(x)
    #x = MaxPooling1D(pool_size=2)(x)

    #x = Conv1D(filters=300,
    #                   kernel_size=3,
    #                   padding='valid',
    #                   activation='tanh',
    #                   strides=1)(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    return model


# 2 bid. GRU 
def gru(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #input_layer = Input(shape=(maxlen,))
    input_layer = Input(shape=(maxlen, embed_size), )
    #embedding_layer = Embedding(max_features, embed_size, 
    #                            weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(x)
    #x = AttentionWeightedAverage(maxlen)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b], axis=1)
    #x = Dense(dense_size, activation="relu")(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model


def gru_best(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #input_layer = Input(shape=(maxlen,))
    input_layer = Input(shape=(maxlen, embed_size), )
    #embedding_layer = Embedding(max_features, embed_size,
    #                            weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(x)
    #x = AttentionWeightedAverage(maxlen)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b], axis=1)
    #x = Dense(dense_size, activation="relu")(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  #optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# 1 layer bid GRU
def gru_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #input_layer = Input(shape=(maxlen,))
    input_layer = Input(shape=(maxlen, embed_size), )
    #embedding_layer = Embedding(max_features, embed_size,
    #                            weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(input_layer)
    #x = AttentionWeightedAverage(maxlen)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b], axis=1)
    #x = Dense(dense_size, activation="relu")(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model


# bid GRU + bid LSTM
def mix1(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #input_layer = Input(shape=(maxlen,))
    input_layer = Input(shape=(maxlen, embed_size), )
    #embedding_layer = Embedding(max_features, embed_size,
    #                            weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model


# DPCNN
def dpcnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    #input_layer = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, embed_size), )
    #X = Embedding(max_features, embed_size, weights=[embedding_matrix], 
    #              trainable=False)(input_layer)
    # first block
    X_shortcut1 = input_layer
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X_shortcut1)
    X = Activation('relu')(X)
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut1)  # pre activation
    X = Add()([X_shortcut1,X])
    X = MaxPooling1D(pool_size=3, strides=2, padding='valid')(X)

    # second block
    X_shortcut2 = X
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut2)  # pre activation
    X = Add()([X_shortcut2,X])
    X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)

    # Output
    X = Flatten()(X)
    X = Dense(nb_classes, activation='sigmoid')(X)

    model = Model(inputs = input_layer, outputs = X, name='dpcnn')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def getModel(model_config, training_config):
    model_type = model_config.model_type
    fold_count = model_config.fold_number

    # for BERT models, parameters are set at class level
    if model_config.model_type.find('bert') != -1:
        print("model_config.maxlen: " + str(model_config.maxlen))
        print("model_config.batch_size: " + str(model_config.batch_size))
        model = BERT_classifier(model_config, 
                                fold_count=fold_count, 
                                labels=model_config.list_classes, 
                                class_weights=training_config.class_weights)
        # the following will ensuring that the model stays warm/available
        model.load()
        return model

    # default model parameters
    parameters = parametersMap[model_type]
    embed_size = parameters['embed_size']
    maxlen = parameters['maxlen']
    batch_size = parameters['batch_size']
    recurrent_units = parameters['recurrent_units']
    dropout_rate = parameters['dropout_rate']
    recurrent_dropout_rate = parameters['recurrent_dropout_rate']
    dense_size = parameters['dense_size']

    # overwrite with config paramters 
    embed_size = model_config.word_embedding_size
    maxlen = model_config.maxlen
    batch_size = training_config.batch_size
    max_epoch = training_config.max_epoch
    model_type = model_config.model_type
    use_roc_auc = training_config.use_roc_auc
    nb_classes = len(model_config.list_classes)    
    dropout_rate = model_config.dropout
    recurrent_dropout_rate = model_config.recurrent_dropout

    # awww Python has no case/switch statement :D
    if (model_type == 'bidLstm'):
        model = bidLstm(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'bidLstm_simple'):
        model = bidLstm_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'lstm'):
        model = lstm(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'cnn'):
        model = cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'cnn2'):
        model = cnn2(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'cnn3'):
        model = cnn3(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'lstm_cnn'):
        model = lstm_cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'conv'):
        model = dpcnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'mix1'):
        model = mix1(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'dpcnn'):
        model = dpcnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'gru'):
        model = gru(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    elif (model_type == 'gru_simple'):
        model = gru_simple(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    else:
        raise (OSError('The model type '+model_type+' is unknown'))
    return model


def train_model(model, list_classes, batch_size, max_epoch, use_roc_auc, class_weights, training_generator, validation_generator, val_y, use_ELMo=False, use_BERT=False):
    best_loss = -1
    best_roc_auc = -1
    best_weights = None
    best_epoch = 0
    current_epoch = 1

    while current_epoch <= max_epoch:

        #model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        nb_workers = 6
        multiprocessing = True
        if use_ELMo or use_BERT:
            # worker at 0 means the training will be executed in the main thread
            nb_workers = 0 
            multiprocessing = False
        model.fit_generator(
            generator=training_generator,
            use_multiprocessing=multiprocessing,
            workers=nb_workers,
            class_weight=class_weights,
            epochs=1)

        y_pred = model.predict_generator(
            generator=validation_generator, 
            use_multiprocessing=multiprocessing,
            workers=nb_workers)

        total_loss = 0.0
        total_roc_auc = 0.0

        # we distinguish 1-class and multiclass problems 
        if len(list_classes) is 1:
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
                #for n in range(0, len(val_y[:, j])):
                #    print(val_y[n, j])
                #print(val_y[:, j])
                #print(y_pred[:, j])
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
        if total_loss < best_loss or best_loss == -1 or math.isnan(best_loss) is True:
            best_loss = total_loss
            if use_roc_auc is False:
                best_weights = model.get_weights()
                best_epoch = current_epoch
        elif use_roc_auc is False:
            if current_epoch - best_epoch == 5:
                break

        if total_roc_auc > best_roc_auc or best_roc_auc == -1:
            best_roc_auc = total_roc_auc
            if use_roc_auc:
                best_weights = model.get_weights()
                best_epoch = current_epoch
        elif use_roc_auc:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)

    if use_roc_auc:
        return model, best_roc_auc
    else:
        return model, best_loss


def train_folds(X, y, model_config, training_config, embeddings):
    fold_count = model_config.fold_number
    max_epoch = training_config.max_epoch
    model_type = model_config.model_type
    use_roc_auc = training_config.use_roc_auc
    class_weights = training_config.class_weights

    fold_size = len(X) // fold_count
    models = []
    scores = []

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
            embeddings=embeddings, shuffle=True)
        validation_generator = DataGenerator(val_x, val_y, batch_size=training_config.batch_size, 
            maxlen=model_config.maxlen, list_classes=model_config.list_classes, 
            embeddings=embeddings, shuffle=False)

        foldModel, best_score = train_model(getModel(model_config, training_config), 
                model_config.list_classes, training_config.batch_size, max_epoch, use_roc_auc, class_weights, training_generator, validation_generator, val_y)
        models.append(foldModel)

        #model_path = os.path.join("../data/models/textClassification/",model_name, model_type+".model{0}_weights.hdf5".format(fold_id))
        #foldModel.save_weights(model_path, foldModel.get_weights())
        #foldModel.save(model_path)
        #del foldModel

        scores.append(best_score)

    all_scores = sum(scores)
    avg_score = all_scores/fold_count

    if (use_roc_auc):
        print("Average best roc_auc scores over the", fold_count, "fold: ", avg_score)
    else:
        print("Average best log loss scores over the", fold_count, "fold: ", avg_score)

    return models


def predict(model, predict_generator, use_ELMo=False, use_BERT=False, use_main_thread_only=False):
    nb_workers = 6
    multiprocessing = True
    if use_ELMo or use_BERT or use_main_thread_only:
        # worker at 0 means the training will be executed in the main thread
        nb_workers = 0 
        multiprocessing = False
    y = model.predict_generator(
            generator=predict_generator, 
            use_multiprocessing=multiprocessing,
            workers=nb_workers)
    return y


def predict_folds(models, predict_generator, use_ELMo=False, use_BERT=False, use_main_thread_only=False):
    fold_count = len(models)
    y_predicts_list = []
    for fold_id in range(0, fold_count):
        model = models[fold_id]
        #y_predicts = model.predict(xte)
        nb_workers = 6
        multiprocessing = True
        if use_ELMo or use_BERT or use_main_thread_only:
            # worker at 0 means the training will be executed in the main thread
            nb_workers = 0 
            multiprocessing = False
        y_predicts = model.predict_generator(
            generator=predict_generator, 
            use_multiprocessing=multiprocessing,
            workers=nb_workers)
        y_predicts_list.append(y_predicts)

    y_predicts = np.ones(y_predicts_list[0].shape)
    for fold_predict in y_predicts_list:
        y_predicts *= fold_predict

    y_predicts **= (1. / len(y_predicts_list))

    return y_predicts    


class BERT_classifier():
    """
    BERT classifier model with fine-tuning.

    Implementation is an adaptation of the official repository: 
    https://github.com/google-research/bert

    For reference:
    --
    @article{devlin2018bert,
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
      author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
      journal={arXiv preprint arXiv:1810.04805},
      year={2018}
    }
    """

    def __init__(self, config, model_name=None, fold_count=1, labels=None, class_weights=None):
        self.graph = tf.get_default_graph()

        print("config.maxlen: ", config.maxlen)
        print("config.batch_size: ", config.batch_size)

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = config.model_name
        self.model_type = config.model_type

        # we get the BERT pretrained files from the embeddings registry
        description = _get_description(self.model_type)
        self.class_weights = class_weights

        if description is None:
            raise Exception('no embeddings description found for ' + self.model_type)

        self.fold_count = fold_count

        self.config_file = description["path-config"]
        self.weight_file = description["path-weights"] # init_checkpoint
        self.vocab_file = description["path-vocab"]

        self.labels = labels

        self.do_lower_case = False
        self.max_seq_length= config.maxlen
        self.train_batch_size = config.batch_size
        self.predict_batch_size = config.batch_size
        self.learning_rate = 2e-5 
        self.num_train_epochs = 1.0
        self.warmup_proportion = 0.1
        self.master = None
        self.save_checkpoints_steps = 99999999 # <----- don't want to save any checkpoints
        self.iterations_per_loop = 1000

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        #self.processor = BERT_classifier_processor(labels=labels)

        self.bert_config = modeling.BertConfig.from_json_file(self.config_file)
        self.model_dir = 'data/models/textClassification/' + self.model_name
        
    def train(self, x_train=None, y_train=None):
        '''
        Train the classifier(s). We train fold_count classifiers if fold_count>1. 
        '''
        start = time.time()

        # remove possible previous model(s)
        for fold_number in range(0, self.fold_count):
            if os.path.exists(self.model_dir+str(fold_number)):
                shutil.rmtree(self.model_dir+str(fold_number))

        train_examples = self.processor.get_train_examples(x_train=x_train, y_train=y_train)

        if self.fold_count == 1:
            self.train_fold(0, train_examples)
        else:
            fold_size = len(train_examples) // self.fold_count

            for fold_id in range(0, self.fold_count):
                tf.logging.info('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')
                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size

                if fold_id == fold_size - 1:
                    fold_end = len(train_examples)

                fold_train_examples = train_examples[:fold_start] + train_examples[fold_end:]

                self.train_fold(fold_id, fold_train_examples)

        end = time.time()
        tf.logging.info("\nTotal training complete in " + str(end - start) + " seconds")


    def train_fold(self, fold_number, train_examples):
        '''
        Train the classifier
        '''
        start = time.time()

        print("len(train_examples): ", len(train_examples))
        print("self.train_batch_size: ", self.train_batch_size)
        print("self.num_train_epochs: ", self.num_train_epochs)

        num_train_steps = int(len(train_examples) / self.train_batch_size * self.num_train_epochs)

        print("num_train_steps: ", num_train_steps)
        print("self.warmup_proportion: ", self.warmup_proportion)

        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        print("num_warmup_steps: ", num_warmup_steps)

        model_fn = model_fn_builder(
              bert_config=self.bert_config,
              num_labels=len(self.labels),
              init_checkpoint=self.weight_file,
              learning_rate=self.learning_rate,
              num_train_steps=num_train_steps,
              num_warmup_steps=num_warmup_steps,
              use_one_hot_embeddings=True)

        run_config = self._get_run_config(fold_number)

        estimator = tf.contrib.tpu.TPUEstimator(
              use_tpu=False,
              model_fn=model_fn,
              config=run_config,
              train_batch_size=self.train_batch_size)
              
        # create dir if does not exist
        if not os.path.exists(self.model_dir+str(fold_number)):
            os.makedirs(self.model_dir+str(fold_number))
        
        train_file = os.path.join(self.model_dir+str(fold_number), "train.tf_record")

        file_based_convert_examples_to_features(train_examples, self.labels, 
            self.max_seq_length, self.tokenizer, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", self.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        print("self.max_seq_length: ", self.max_seq_length)
        print("self.train_batch_size: ", self.train_batch_size)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=False,
            batch_size=self.train_batch_size)
            
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        end = time.time()
        tf.logging.info("\nTraining complete in " + str(end - start) + " seconds")

        # cleaning the training garbages
        os.remove(train_file)

        # the initial check point has prefix model.ckpt-0* and can be removed
        # (given that it is 1.3GB file, it's preferable!) 
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.data-00000-of-00001")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.index")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.meta")
        if os.path.exists(garbage):
            os.remove(garbage)

    def eval(self, x_test=None, y_test=None, run_number=0):
        '''
        Train and eval the nb_runs classifier(s) against holdout set. If nb_runs>1, the final
        score are averaged over the nb_runs models. The best model against holdout is saved.
        '''
        start = time.time()
        predict_examples, y_test = self.processor.get_test_examples(x_test=x_test, y_test=y_test)
        #y_test_gold = np.asarray([np.argmax(line) for line in y_test])

        y_predicts = self.eval_fold(predict_examples)
        result_intermediate = np.asarray([np.argmax(line) for line in y_predicts])

        def vectorize(index, size):
            result = np.zeros(size)
            if index < size:
                result[index] = 1
            return result
        result_binary = np.array([vectorize(xi, len(self.labels)) for xi in result_intermediate])

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, result_binary, average=None)
        print('\n')
        print('{:>14}  {:>12}  {:>12}  {:>12}  {:>12}'.format(" ", "precision", "recall", "f-score", "support"))
        p = 0
        for the_class in self.labels:
            the_class = the_class[:14]
            print('{:>14}  {:>12}  {:>12}  {:>12}  {:>12}'.format(the_class, "{:10.4f}"
                .format(precision[p]), "{:10.4f}".format(recall[p]), "{:10.4f}".format(fscore[p]), support[p]))
            p += 1

        runtime = round(time.time() - start, 3)

        print("Total runtime for eval: " + str(runtime) + " seconds")

    def eval_fold(self, predict_examples, fold_number=0):
        
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(self.model_dir+str(fold_number), "predict.tf_record")

        file_based_convert_examples_to_features(predict_examples, self.labels,
                                                self.max_seq_length, self.tokenizer,
                                                predict_file)

        tf.logging.info("***** Running holdout prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=self.predict_batch_size)

        num_train_steps = int(31861 / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        model_fn = model_fn_builder(
              bert_config=self.bert_config,
              num_labels=len(self.labels),
              init_checkpoint=self.weight_file,
              learning_rate=self.learning_rate,
              num_train_steps=num_train_steps,
              num_warmup_steps=num_warmup_steps,
              #use_tpu=self.use_tpu,
              use_one_hot_embeddings=True)

        run_config = self._get_run_config(fold_number)

        estimator = tf.contrib.tpu.TPUEstimator(
              use_tpu=False,
              model_fn=model_fn,
              config=run_config,
              predict_batch_size=self.predict_batch_size)

        result = estimator.predict(input_fn=predict_input_fn)
        
        y_pred = np.zeros(shape=(len(predict_examples),len(self.labels)))

        p = 0
        for prediction in result:
            probabilities = prediction["probabilities"]
            q = 0
            for class_probability in probabilities:
                if self.class_weights is not None:
                    y_pred[p,q] = class_probability * self.class_weights[q]
                else:
                    y_pred[p,q] = class_probability
                q += 1
            p += 1
        
        # cleaning the garbages
        os.remove(predict_file)

        return y_pred


    def predict(self, texts, fold_number=0):
        if self.loaded_estimator is None:
            self.load_model(fold_number)        

        # create the DeLFT json result remplate
        '''
        res = {
            "software": "DeLFT",
            "date": datetime.datetime.now().isoformat(),
            "model": self.model_name,
            "classifications": []
        }
        '''
        if texts is None or len(texts) == 0:
            return res

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        y_pred = np.zeros(shape=(len(texts),len(self.labels)))
        y_pos = 0

        for text_batch in list(chunks(texts, self.predict_batch_size)):
            if type(text_batch) is np.ndarray:
                text_batch = text_batch.tolist()

            # if the size of the last batch is less than the batch size, we need to fill it with dummy input
            num_current_batch = len(text_batch)
            if num_current_batch < self.predict_batch_size:
                dummy_text = text_batch[-1]                
                for p in range(0, self.predict_batch_size-num_current_batch):
                    text_batch.append(dummy_text)

            # segment in batches corresponding to self.predict_batch_size
            input_examples = self.processor.create_inputs(text_batch, dummy_label=self.labels[0])
            input_features = convert_examples_to_features(input_examples, self.labels, self.max_seq_length, self.tokenizer)

            results = self.loaded_estimator.predict(input_features, self.max_seq_length, self.predict_batch_size)

            #y_pred = np.zeros(shape=(num_current_batch,len(self.labels)))
            p = 0
            for prediction in results:
                if p == num_current_batch:
                    break
                probabilities = prediction["probabilities"]
                q = 0
                for class_probability in probabilities:
                    if self.class_weights and len(self.class_weights) == len(probabilities):    
                        y_pred[y_pos+p,q] = class_probability * self.class_weights[q]
                    else:
                        y_pred[y_pos+p,q] = class_probability 
                    q += 1
                p += 1
            y_pos += num_current_batch
            '''
            y_pred_best = np.asarray([np.argmax(line) for line in y_pred])            
            
            i = 0
            for text in text_batch:
                if i == num_current_batch:
                    break
                classification = {
                    "text": text
                }
                j = 0
                for cl in self.labels:
                    classification[cl] = float(y_pred[i,j])
                    j += 1
                best = {
                    "class": self.labels[y_pred_best[i]],
                    "conf": float(y_pred[i][y_pred_best[i]])
                }
                classification['selection'] = best
                res["classifications"].append(classification)
                i += 1
            '''

        return y_pred

    def _get_run_config(self, fold_number=0):
        tpu_cluster_resolver = None
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.master,
            model_dir=self.model_dir+str(fold_number),
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                #num_shards=self.num_tpu_cores,
                per_host_input_for_training=is_per_host)
            )
        return run_config

    def load(self):
        # default
        num_train_steps = int(10000 / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        model_fn = model_fn_builder(
              bert_config=self.bert_config,
              num_labels=len(self.labels),
              init_checkpoint=self.weight_file,
              learning_rate=self.learning_rate,
              num_train_steps=num_train_steps,
              num_warmup_steps=num_warmup_steps,
              use_one_hot_embeddings=True)

        run_config = self._get_run_config(0)

        self.loaded_estimator = FastPredict(tf.contrib.tpu.TPUEstimator(
              use_tpu=False,
              model_fn=model_fn,
              config=run_config,
              predict_batch_size=self.predict_batch_size), input_fn_generator)   

def _get_description(name, path="./embedding-registry.json"):
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["embeddings-contextualized"]:
        if emb["name"] == name:
            return emb
    return None