import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import clone_model
from sequenceLabelling.layers import ChainCRF

class BaseModel(object):

    def __init__(self, config, embeddings, ntags):
        self.config = config
        self.embeddings = embeddings
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
    """A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, embeddings=None, ntags=None):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        if embeddings is None:
            word_embeddings = Embedding(input_dim=config.vocab_size,
                                        output_dim=config.word_embedding_size, trainable=False,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=embeddings.shape[0],
                                        output_dim=embeddings.shape[1], trainable=False,
                                        mask_zero=True, 
                                        weights=[embeddings])(word_ids)

        # build character based embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True
                                    )(char_ids)
        s = K.shape(char_embeddings)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], config.char_embedding_size)))(char_embeddings)

        fwd_state = LSTM(config.num_char_lstm_units, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(config.num_char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * config.num_char_lstm_units]))(char_embeddings)

        # combine characters and word
        x = Concatenate(axis=-1)([word_embeddings, char_embeddings])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        self.model = Model(inputs=[word_ids, char_ids, sequence_lengths], outputs=[pred])
        self.config = config


class SeqLabelling_BidLSTM_CNN(BaseModel):
    """A Keras implementation of BidLSTM-CNN for sequence labelling.

    References
    --
    Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. 
    https://arxiv.org/abs/1511.08308
    """

    def __init__(self, config, embeddings=None, ntags=None):

        # :: Create a mapping for the labels ::
        """
        label2Idx = {}
        for label in labelSet:
            label2Idx[label] = len(label2Idx)

        # :: Hard coded case lookup ::
        case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
        caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

        char2Idx = {"PADDING":0, "UNKNOWN":1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            char2Idx[c] = len(char2Idx)

        # build word embedding
        word_ids = Input(shape=(None,),dtype='int32', name='words_input')
        if embeddings is None:
            word_embeddings = Embedding(input_dim=config.vocab_size,
                                        output_dim=config.word_embedding_size, trainable=False,
                                        mask_zero=True)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=embeddings.shape[0],
                                        output_dim=embeddings.shape[1], trainable=False,
                                        mask_zero=True, 
                                        weights=[embeddings])(word_ids)

        # build character based embedding        
        character_input = Input(shape=(None,52,),name='char_input')
        embed_char_out = TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), 
                            name='char_embedding')(character_input)
        dropout = Dropout(0.5)(embed_char_out)
        
        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
        
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(0.5)(char)
        
        # custom features input and embeddings
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], 
                            trainable=False)(casing_input)

        # combine words, custom features and characters
        x = concatenate([word_embeddings, casing, char])
        x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(x)
        x = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(x)
        
        model = Model(inputs=[words_input, character_input, casing_input], outputs=[x])
        self.config = config
        """