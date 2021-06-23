from keras.layers import Dense, LSTM, GRU, Bidirectional, Embedding, Input, Dropout
from keras.layers import GlobalMaxPooling1D, TimeDistributed, Conv1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import clone_model

from delft.utilities.bert import modeling
from delft.utilities.bert import optimization
from delft.utilities.bert import tokenization

from delft.utilities.layers import ChainCRF

from delft.sequenceLabelling.preprocess import NERProcessor, convert_single_example, input_fn_generator, convert_examples_to_features
from delft.sequenceLabelling.preprocess import file_based_input_fn_builder, file_based_convert_examples_to_features

import json
import time
import os
import shutil
import numpy as np
np.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)


def get_model(config, preprocessor, ntags=None, dir_path=None):
    if config.model_type == BidLSTM_CRF.name:
        preprocessor.return_casing = False
        config.use_crf = True
        return BidLSTM_CRF(config, ntags)
    elif config.model_type == BidLSTM_CNN.name:
        preprocessor.return_casing = True
        config.use_crf = False
        return BidLSTM_CNN(config, ntags)
    elif config.model_type == BidLSTM_CNN_CRF.name:
        preprocessor.return_casing = True
        config.use_crf = True
        return BidLSTM_CNN_CRF(config, ntags)
    elif config.model_type == BidGRU_CRF.name:
        preprocessor.return_casing = False
        config.use_crf = True
        return BidGRU_CRF(config, ntags)
    elif config.model_type == BidLSTM_CRF_FEATURES.name:
        preprocessor.return_casing = False
        preprocessor.return_features = True
        config.use_crf = True
        return BidLSTM_CRF_FEATURES(config, ntags)
    elif config.model_type == BidLSTM_CRF_CASING.name:
        preprocessor.return_casing = True
        config.use_crf = True
        return BidLSTM_CRF_CASING(config, ntags)
    elif 'bert' in config.model_type.lower():
        preprocessor.return_casing = False
        # note that we could consider optionally in the future using a CRF
        # as activation layer for BERT e
        config.use_crf = False
        config.labels = preprocessor.vocab_tag
        return BERT_Sequence(config, ntags, dir_path)
    else:
        raise (OSError('Model name does exist: ' + config.model_type))


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


class BidLSTM_CRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """
    name = 'BidLSTM_CRF'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    #mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False)))(char_embeddings)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, length_input], outputs=[pred])
        self.config = config


class BidLSTM_CNN(BaseModel):
    """
    A Keras implementation of BidLSTM-CNN for sequence labelling.

    References
    --
    Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. 
    https://arxiv.org/abs/1511.08308
    """

    name = 'BidLSTM_CNN'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    name='char_embeddings'
                                    ))(char_input)

        dropout = Dropout(config.dropout)(char_embeddings)

        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
        chars = Dropout(config.dropout)(maxpool_out)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')
        casing_embedding = Embedding(input_dim=config.case_vocab_size,
                           output_dim=config.case_embedding_size,
                           #mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32')

        # combine words, custom features and characters
        x = Concatenate(axis=-1)([word_input, casing_embedding, chars])
        x = Dropout(config.dropout)(x)
        x = Bidirectional(LSTM(units=config.num_word_lstm_units, 
                               return_sequences=True, 
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        #pred = TimeDistributed(Dense(ntags, activation='softmax'))(x)
        pred = Dense(ntags, activation='softmax')(x)

        self.model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[pred])
        self.config = config


class BidLSTM_CNN_CRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CNN-CRF for sequence labelling.

    References
    --
    Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF". 2016. 
    https://arxiv.org/abs/1603.01354
    """

    name = 'BidLSTM_CNN_CRF'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    name='char_embeddings'
                                    ))(char_input)

        dropout = Dropout(config.dropout)(char_embeddings)

        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
        chars = Dropout(config.dropout)(maxpool_out)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')

        """
        casing_embedding = Embedding(input_dim=config.case_vocab_size, 
                           output_dim=config.case_embedding_size,
                           mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)
        """

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32')

        # combine words, custom features and characters
        x = Concatenate(axis=-1)([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units, 
                               return_sequences=True, 
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[pred])
        self.config = config


class BidGRU_CRF(BaseModel):
    """
    A Keras implementation of BidGRU-CRF for sequence labelling.
    """

    name = 'BidGRU_CRF'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False)))(char_embeddings)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(GRU(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Bidirectional(GRU(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, length_input], outputs=[pred])
        self.config = config


class BidLSTM_CRF_CASING(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling with additinal features related to casing
    (inferred from word forms).
    """

    name = 'BidLSTM_CRF_CASING'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False)))(char_embeddings)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')

        casing_embedding = Embedding(input_dim=config.case_vocab_size, 
                           output_dim=config.case_embedding_size,
                           mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, casing_embedding, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True, 
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[pred])
        self.config = config

class BidLSTM_CRF_FEATURES(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information.
    """

    name = 'BidLSTM_CRF_FEATURES'

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units,
                                                   return_sequences=False)))(char_embeddings)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=False,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)
        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, features_input, length_input], outputs=[pred])
        self.config = config


class BERT_Sequence(BaseModel):
    """
    This class allows to use a BERT TensorFlow architecture for sequence labelling. The architecture is
    build on the official Google TensorFlow implementation and cannot be mixed with Keras layers for 
    retraining. Training corresponds to a fine tuning only of a provided pre-trained model.

    BERT sequence labelling model with fine-tuning, using a CRF as activation layer. Replacing the usual 
    softmax activation layer by a CRF activation always improves performance for sequence labelling.

    The implementation is an adaptation of the official repository: 
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

    def __init__(self, config, ntags=None, dir_path=None):
        self.graph = tf.get_default_graph()

        self.model_name = config.model_name
        self.model_type = config.model_type

        # we get the BERT pretrained files from the embeddings registry
        description = _get_description(self.model_type)

        if description is None:
            raise Exception('no embeddings description found for ' + self.model_type)

        self.fold_count = config.fold_number

        self.config_file = description["path-config"]
        self.weight_file = description["path-weights"] # init_checkpoint
        self.vocab_file = description["path-vocab"]

        self.labels = []

        # by bert convention, PAD is zero
        self.labels.append("[PAD]")
        # adding other default bert convention labels 
        self.labels.append("[CLS]")
        self.labels.append("[SEP]")
        # the following label is added for added bert tokens introduced by ## 
        self.labels.append("X")

        for label in config.labels:
            if label == '<PAD>' or label == '<UNK>':
                continue
            if label not in self.labels:
                self.labels.append(label)

        self.do_lower_case = False
        self.max_seq_length = config.max_sequence_length
        self.train_batch_size = config.batch_size
        self.predict_batch_size = config.batch_size
        self.learning_rate = 2e-5 
        self.num_train_epochs = 5.0
        self.warmup_proportion = 0.1
        self.master = None
        self.save_checkpoints_steps = 99999999 # <----- don't want to save any checkpoints
        self.iterations_per_loop = 1000

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.processor = NERProcessor(self.labels)

        self.bert_config = modeling.BertConfig.from_json_file(self.config_file)
        if dir_path is None:
            self.model_dir = 'data/models/sequenceLabelling/' + self.model_name    
        else:
            if not dir_path.endswith("/"):
                dir_path += "/"
            self.model_dir = dir_path + self.model_name 
        self.loaded_estimator = None

    def train(self, x_train=None, y_train=None):
        '''
        Train the sequence labelling model. We train fold_count classifiers if fold_count>1. 
        '''
        start = time.time()

        # remove possible previous model(s)
        for fold_number in range(0, self.fold_count):
            if os.path.exists(self.model_dir+str(fold_number)):
                shutil.rmtree(self.model_dir+str(fold_number))

        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        train_examples = self.processor.get_train_examples(x_train, y_train)

        if self.fold_count == 1:
            self.train_fold(-1, train_examples)
        else:
            for fold_id in range(0, self.fold_count):
                print('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')
                # no validation set used during training (it's just used for setting the hyper-parameters)
                # so it's simply repeating n times a tranining with the test set
                self.train_fold(fold_id, train_examples)

        end = time.time()
        tf.logging.info("\nTotal training complete in " + str(end - start) + " seconds")


    def train_fold(self, fold_number, train_examples):
        '''
        Train the sequence labelling model
        '''
        start = time.time()

        num_train_steps = int(len(train_examples) / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        model_fn = self.model_fn_builder(
              bert_config=self.bert_config,
              label_list=self.labels,
              init_checkpoint=self.weight_file,
              learning_rate=self.learning_rate,
              num_train_steps=num_train_steps,
              num_warmup_steps=num_warmup_steps,
              use_tpu=False,
              use_one_hot_embeddings=True)

        run_config = self._get_run_config(fold_number)

        estimator = tf.contrib.tpu.TPUEstimator(
              use_tpu=False,
              model_fn=model_fn,
              config=run_config,
              train_batch_size=self.train_batch_size)
              
        # create dir if does not exist
        if fold_number == -1:
            suffix = ""
        else:
            suffix = str(fold_number)
        if not os.path.exists(self.model_dir+suffix):
            os.makedirs(self.model_dir+suffix)
        
        train_file = os.path.join(self.model_dir+suffix, "train.tf_record")

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
        # (given that there is a 1.3 GB file, it's better!) 
        garbage = os.path.join(self.model_dir+suffix, "model.ckpt-0.data-00000-of-00001")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+suffix, "model.ckpt-0.index")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+suffix, "model.ckpt-0.meta")
        if os.path.exists(garbage):
            os.remove(garbage)


    def predict(self, texts, fold_id=-1):
        if self.loaded_estimator is None:
            self.load_model(fold_id)        

        y_pred = []

        if texts is None or len(texts) == 0:
            return y_pred

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
        
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
            input_examples = self.processor.create_inputs(text_batch)
            input_features, input_tokens = convert_examples_to_features(input_examples, self.labels, self.max_seq_length, self.tokenizer)

            results = self.loaded_estimator.predict(input_features, self.max_seq_length, self.predict_batch_size)
            p = 0
            for i, prediction in enumerate(results):
                if p == num_current_batch:
                    break
                predicted_labels = prediction["predicts"]
                y_pred_result = []
                for q in range(len(predicted_labels)):
                    if input_tokens[i][q] == '[SEP]':
                        break
                    if self.labels[predicted_labels[q]] in ['[PAD]', '[CLS]', '[SEP]']:
                        continue
                    if input_tokens[i][q].startswith("##"): 
                        continue
                    y_pred_result.append(self.labels[predicted_labels[q]]) 
                y_pred.append(y_pred_result)
                p += 1

        return y_pred

    def model_fn_builder(self,
                         bert_config,
                         label_list,
                         init_checkpoint,
                         learning_rate=2e-5,
                         num_train_steps=0,
                         num_warmup_steps=0,
                         use_tpu=False,
                         use_one_hot_embeddings=True):
        """
        Returns `model_fn` closure for TPUEstimator.
        """
        def model_fn(features,
                     labels,
                     mode,
                     params):
            """
            The `model_fn` for TPUEstimator
            """
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"] if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] else None
            
            is_training = False
            if mode == tf.estimator.ModeKeys.TRAIN: 
                is_training = True
            (loss, logits, predicts) = self.create_model(bert_config, 
                                                        is_training, 
                                                        input_ids, 
                                                        input_mask, 
                                                        segment_ids, 
                                                        label_ids, 
                                                        len(label_list), 
                                                        use_one_hot_embeddings, 
                                                        mode)
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            
            if init_checkpoint:
                assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
            
            output_spec = None        
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(labels,
                              predicts,
                              label_list):
                    label_map = {}
                    for (i, label) in enumerate(label_list):
                        label_map[label] = i
                    
                    pad_id = tf.constant(label_map["[PAD]"], shape=[], dtype=tf.int32)
                    out_id = tf.constant(label_map["O"], shape=[], dtype=tf.int32)
                    x_id = tf.constant(label_map["X"], shape=[], dtype=tf.int32)
                    cls_id = tf.constant(label_map["[CLS]"], shape=[], dtype=tf.int32)
                    sep_id = tf.constant(label_map["[SEP]"], shape=[], dtype=tf.int32)
                    
                    masked_labels = (tf.cast(tf.not_equal(labels, pad_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(labels, out_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(labels, x_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(labels, cls_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(labels, sep_id), dtype=tf.int32))
                    
                    masked_predicts = (tf.cast(tf.not_equal(predicts, pad_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(predicts, out_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(predicts, x_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(predicts, cls_id), dtype=tf.int32) *
                        tf.cast(tf.not_equal(predicts, sep_id), dtype=tf.int32))
                    
                    precision = tf.metrics.precision(labels=masked_labels, predictions=masked_predicts)
                    recall = tf.metrics.recall(labels=masked_labels, predictions=masked_predicts)
                    
                    metric = {
                        "precision": precision,
                        "recall": recall
                    }
                    
                    return metric
                
                eval_metrics = (metric_fn, [label_ids, predicts, label_list])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={ "predicts": predicts },
                    scaffold_fn=scaffold_fn)
            
            return output_spec
        
        return model_fn

    def _get_run_config(self, fold_id=-1):
        tpu_cluster_resolver = None
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        if fold_id == -1:
            suffix = ""
        else:
            suffix = str(fold_id)
        
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.master,
            model_dir=self.model_dir+suffix,
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                #num_shards=self.num_tpu_cores,
                per_host_input_for_training=is_per_host)
            )
        return run_config

    def create_model(self, bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, mode, 
                 use_crf=True):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )

        with tf.variable_scope("loss"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, num_labels])
            if use_crf:
                mask2len = tf.reduce_sum(input_mask, axis=1)
                loss, trans = _crf_loss(logits, labels, input_mask, num_labels, mask2len, mode)
                predicts, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
                return (loss, logits, predicts)
            else:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = 0.0
                if mode != tf.estimator.ModeKeys.PREDICT:
                    # loss to be ignored for prediction
                    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
                    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                    loss = tf.reduce_sum(per_example_loss)
                probabilities = tf.nn.softmax(logits, axis=-1)
                predicts = tf.argmax(probabilities, axis=-1)
                return (loss, logits, predicts)


    def load_model(self, fold_id=-1):
        # default
        num_train_steps = int(10000 / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        model_fn = self.model_fn_builder(
              bert_config=self.bert_config,
              label_list=self.labels,
              init_checkpoint=self.weight_file,
              learning_rate=self.learning_rate,
              num_train_steps=num_train_steps,
              num_warmup_steps=num_warmup_steps,
              use_one_hot_embeddings=True)

        run_config = self._get_run_config(fold_id)

        self.loaded_estimator = FastPredict(tf.contrib.tpu.TPUEstimator(
              use_tpu=False,
              model_fn=model_fn,
              config=run_config,
              predict_batch_size=self.predict_batch_size), input_fn_generator)   


# note: use same method in Embeddings class utilities and remove this one
def _get_description(name, path="./embedding-registry.json"):
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["transformers"]:
        if emb["name"] == name:
            return emb
    return None

def _crf_loss(logits, labels, mask, num_labels, mask2len, mode):

    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels,num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        if mode == tf.estimator.ModeKeys.PREDICT:   
            return None, trans
        else:
            log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans, sequence_lengths=mask2len)
            loss = tf.math.reduce_mean(-log_likelihood)
            return loss, transition

def _gen_builder(features):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    return {"input_ids":all_input_ids, "input_mask":all_input_mask, "segment_ids":all_segment_ids, "label_ids": all_label_ids}

class FastPredict:
    '''
    Modified from https://github.com/marcsto/rl/blob/master/src/fast_predict2.py

    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.
    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir), my_input_fn)
    This version supports tf 1.4 and above and can be used by pre-made Estimators like tf.estimator.DNNClassifier. 
    
    Original author: Marc Stogaitis
    '''
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn
        self.next_features = None
        self.seq_length = None
        self.batch_size = None

    def _create_generator(self):
        while not self.closed:
            local_gen = _gen_builder(self.next_features)
            yield local_gen

    def predict(self, feature_batch, seq_length, batch_size):
        """ 
        Runs a prediction on a set of features. Calling multiple times does *not* regenerate the graph 
        which makes predict much faster.
        feature_batch is a list of list of features. IMPORTANT: If you're only classifying 1 thing, 
        you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), 
        not predict(my_feature) 
        """
        self.next_features = feature_batch
        self.seq_length = seq_length
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator, seq_length, batch_size))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. But this is probably OK.")

