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

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

#import utilities.Attention
from delft.utilities.Attention import Attention
#from ToxicAttentionAlternative import AttentionAlternative
#from ToxicAttentionWeightedAverage import AttentionWeightedAverage

# seed is fixed for reproducibility
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(8)

modelTypes = ['lstm', 'bidLstm_simple', 'bidLstm', 'cnn', 'cnn2', 'cnn3', 'mix1', 'dpcnn', 'conv', "gru", "gru_simple", 'lstm_cnn', 'han', 'bert']

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
    fold_count = model_config.fold_number
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
            total_loss = log_loss(val_y, y_pred)
            total_roc_auc = roc_auc_score(val_y, y_pred)
        else:
            for j in range(0, len(list_classes)):
                loss = log_loss(val_y[:, j], y_pred[:, j])
                total_loss += loss
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


def predict(model, predict_generator, use_ELMo=False, use_BERT=False):
    nb_workers = 6
    multiprocessing = True
    if use_ELMo or use_BERT:
        # worker at 0 means the training will be executed in the main thread
        nb_workers = 0 
        multiprocessing = False
    y = model.predict_generator(
            generator=predict_generator, 
            use_multiprocessing=multiprocessing,
            workers=nb_workers)
    return y


def predict_folds(models, predict_generator, use_ELMo=False, use_BERT=False):
    fold_count = len(models)
    y_predicts_list = []
    for fold_id in range(0, fold_count):
        model = models[fold_id]
        #y_predicts = model.predict(xte)
        nb_workers = 6
        multiprocessing = True
        if use_ELMo or use_BERT:
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
