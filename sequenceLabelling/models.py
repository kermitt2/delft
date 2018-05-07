import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Flatten, GlobalMaxPool2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import clone_model
from utilities.layers import ChainCRF
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)

class BaseModel(object):

    def __init__(self, config, ntags):
        self.config = config
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        print('loading model weights', filepath)
        self.model.load_weights(filepath=filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def clone_model(self):
        model_copy = clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return model_copy


class SeqLabelling_BidLSTM_CRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), )

        # build character based embedding
        char_input = Input(shape=(None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True
                                    )(char_input)
        s = K.shape(char_embeddings)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], config.char_embedding_size)))(char_embeddings)

        fwd_state = LSTM(config.num_char_lstm_units, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(config.num_char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * config.num_char_lstm_units]))(char_embeddings)

        # combine characters and word embeddings
        x = Concatenate(axis=-1)([word_input, char_embeddings])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True, recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32')
        self.model = Model(inputs=[word_input, char_input, length_input], outputs=[pred])
        self.config = config


class SeqLabelling_BidLSTM_CNN(BaseModel):
    """
    A Keras implementation of BidLSTM-CNN for sequence labelling.

    References
    --
    Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. 
    https://arxiv.org/abs/1511.08308
    """

    def __init__(self, config, embeddings=None, ntags=None):
        
        # :: Hard coded case lookup ::
        case_index = {'<PAD>': 0, 'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7}
        caseEmbeddings = np.identity(len(case_index), dtype='float32')

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), )

        # build character based embedding        
        character_input = Input(shape=(None,config.char_vocab_size,),name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), 
                                    name='char_embedding')(character_input)
        dropout = Dropout(config.dropout)(char_embeddings)
        
        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(config.char_vocab_size))(conv1d_out)
        
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(config.dropout)(char)
        
        # custom features input and embeddings
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], 
                            trainable=False)(casing_input)

        # combine words, custom features and characters
        x = concatenate([word_embeddings, casing, char])
        x = Bidirectional(LSTM(200, return_sequences=True, dropout=config.dropout, recurrent_dropout=config.recurrent_dropout))(x)
        x = TimeDistributed(Dense(ntags, activation='softmax'))(x)
        
        model = Model(inputs=[words_input, character_input, casing_input], outputs=[x])
        self.config = config
        