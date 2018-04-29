# Class for experimenting various single input DL models at word level
#
# arguments:
#   modelName       name of the model to be used into ['lstm', 'bidLstm', 'cnn', 'cudnngru', 'cudnnlstm']
#   use-holdout     use the holdout to limit the training and generate prediction on holdout set
#   fold-count      number of folds for k-fold training (default is 1)
#

import pandas as pd
import numpy as np
import pandas as pd
import sys, os
import argparse
import math

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
from utilities.Attention import Attention
#from ToxicAttentionAlternative import AttentionAlternative
#from ToxicAttentionWeightedAverage import AttentionWeightedAverage

# seed is fixed for reproducibility
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(8)

modelNames = ['lstm', 'bidLstm_simple', 'bidLstm', 'cnn', 'cnn2', 'cnn3', 'mix1', 'dpcnn', 'conv', "gru", "gru_simple", 'lstm_cnn', 'han']

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
def lstm(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(x)
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
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    return model

# bidirectional LSTM 
def bidLstm_simple(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(x)
    x = Dropout(dropout_rate)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])
    return model

# bidirectional LSTM with attention layer
def bidLstm(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(x)
    #x = Dropout(dropout_rate)(x)
    x = Attention(maxlen)(x)
    #x = AttentionWeightedAverage(maxlen)(x)
    #print('len(x):', len(x))
    #x = AttentionWeightedAverage(maxlen)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# conv+GRU with embeddings
def cnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(x) 
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn2_best(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(x) 
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
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn2(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(x) 
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
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn3(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(x)
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
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def conv(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    filter_kernels = [7, 7, 5, 5, 3, 3]
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    conv = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[0], border_mode='valid', activation='relu')(x)
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
    x = Dense(6, activation="sigmoid")(z)
    model = Model(inputs=inp, outputs=x)
    model.summary()  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# LSTM + conv
def lstm_cnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(x)
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
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    return model

# 2 bid. GRU 
def gru(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    input_layer = Input(shape=(maxlen,))

    #print(max_features)
    #print(embed_size)
    #print(embedding_matrix)

    embedding_layer = Embedding(max_features, embed_size, 
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(embedding_layer)
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
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  #optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def gru_best(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features, embed_size,
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(embedding_layer)
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
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model

# 1 layer bid GRU
def gru_simple(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size,embedding_matrix):
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features, embed_size,
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(embedding_layer)
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
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model

# bid GRU + bid LSTM
def mix1(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features, embed_size,
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  #optimizer='adam',
                  metrics=['accuracy'])
    return model

# DPCNN
def dpcnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_matrix):
    input_layer = Input(shape=(maxlen, ))
    X = Embedding(max_features, embed_size, weights=[embedding_matrix], 
                  trainable=False)(input_layer)
    # first block
    X_shortcut1 = X
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
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
    X = Dense(6, activation='sigmoid')(X)

    model = Model(inputs = input_layer, outputs = X, name='dpcnn')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def getModel(modelName, embedding_vector):
    parameters = parametersMap[modelName]

    max_features = embedding_vector.shape[0]
    embed_size = embedding_vector.shape[1]
    maxlen = parameters['maxlen']
    epoch = parameters['epoch']
    batch_size = parameters['batch_size']
    recurrent_units = parameters['recurrent_units']
    dropout_rate = parameters['dropout_rate']
    recurrent_dropout_rate = parameters['recurrent_dropout_rate']
    dense_size = parameters['dense_size']

    if (modelName == 'bidLstm'):
        model = bidLstm(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'bidLstm_simple'):
        model = bidLstm_simple(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'lstm'):
        model = lstm(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'cnn'):
        model = cnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'cnn2'):
        model = cnn2(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'cnn3'):
        model = cnn3(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'lstm_cnn'):
        model = lstm_cnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'conv'):
        model = dpcnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'mix1'):
        model = mix1(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'dpcnn'):
        model = dpcnn(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'gru'):
        model = gru(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    if (modelName == 'gru_simple'):
        model = gru_simple(maxlen, max_features, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, embedding_vector)
    
    return model


def train_model(model, list_classes, batch_size, max_epoch, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_roc_auc = -1
    best_weights = None
    best_epoch = 0
    current_epoch = 1

    while current_epoch <= max_epoch:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0.0
        total_roc_auc = 0.0
        for j in range(len(list_classes)):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss
            roc_auc = roc_auc_score(val_y[:, j], y_pred[:, j])
            total_roc_auc += roc_auc

        total_loss /= len(list_classes)
        total_roc_auc /= len(list_classes)
        print("Epoch {0} loss {1} best_loss {2} (for info) ".format(current_epoch, total_loss, best_loss))
        print("Epoch {0} roc_auc {1} best_roc_auc {2} (for early stop) ".format(current_epoch, total_roc_auc, best_roc_auc))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1 or math.isnan(best_loss) is True:
            best_loss = total_loss

        if total_roc_auc > best_roc_auc or best_roc_auc == -1:
            best_roc_auc = total_roc_auc
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model, best_roc_auc


def train_folds(X, y, fold_count, list_classes, batch_size, max_epoch, model_name, model_type, embeddings):
    fold_size = len(X) // fold_count
    models = []
    roc_scores = []

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

        foldModel, best_roc_auc = train_model(getModel(model_type, embeddings), list_classes, batch_size, max_epoch, train_x, train_y, val_x, val_y)
        models.append(foldModel)
        
        #model_path = os.path.join("../data/models/textClassification/",model_name, model_type+".model{0}_weights.hdf5".format(fold_id))
        #foldModel.save_weights(model_path, foldModel.get_weights())
        #foldModel.save(model_path)
        #del foldModel

        roc_scores.append(best_roc_auc)
    all_roc_scores = sum(roc_scores)
    print("Average best roc_auc scores over the", fold_count, "fold: ", all_roc_scores/fold_count)

    return models

def predict(model, xte):
    y = model.predict(xte)
    return y

def predict_folds(models, xte):
    fold_count = len(models)
    y_predicts_list = []
    for fold_id in range(0, fold_count):
        model = models[fold_id]
        y_predicts = model.predict(xte)
        y_predicts_list.append(y_predicts)

    y_predicts = np.ones(y_predicts_list[0].shape)
    for fold_predict in y_predicts_list:
        y_predicts *= fold_predict

    y_predicts **= (1. / len(y_predicts_list))

    return y_predicts    

