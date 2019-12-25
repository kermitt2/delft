import keras.backend as K
from keras.layers import Dense, LSTM, GRU, Bidirectional, Embedding, Input, Dropout, Lambda, Flatten
from keras.layers import GlobalMaxPooling1D, TimeDistributed, Conv1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.initializers import RandomUniform
from keras.models import Model
from keras.models import clone_model

from delft.utilities.bert import modeling
from delft.utilities.bert import optimization
from delft.utilities.bert import tokenization

from delft.utilities.layers import ChainCRF

from delft.sequenceLabelling.preprocess import NERProcessor, convert_single_example

import json
import time
import os
import shutil
import collections
import numpy as np
np.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)


def get_model(config, preprocessor, ntags=None):
    if config.model_type == 'BidLSTM_CRF':
        preprocessor.return_casing = False
        config.use_crf = True
        return BidLSTM_CRF(config, ntags)
    elif config.model_type == 'BidLSTM_CNN':
        preprocessor.return_casing = True
        config.use_crf = False
        return BidLSTM_CNN(config, ntags)
    elif config.model_type == 'BidLSTM_CNN_CRF':
        preprocessor.return_casing = True
        config.use_crf = True
        return BidLSTM_CNN_CRF(config, ntags)
    elif config.model_type == 'BidGRU_CRF':
        preprocessor.return_casing = False
        config.use_crf = True
        return BidGRU_CRF(config, ntags)
    elif config.model_type == 'BidLSTM_CRF_CASING':
        preprocessor.return_casing = True
        config.use_crf = True
        return BidLSTM_CRF_CASING(config, ntags)
    elif 'bert' in config.model_type.lower():
        preprocessor.return_casing = False
        # note that we could consider optionnaly in the future using a CRF 
        # as activation layer for BERT e
        config.use_crf = False
        config.labels = preprocessor.vocab_tag
        return BERT_Sequence(config, ntags)
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

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    #mask_zero=True,
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

    def __init__(self, config, ntags=None):

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
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
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360

    In this architecture some casing features are added, just to see...
    """

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


class BERT_Sequence(BaseModel):
    """
    This class allows to use a BERT TensorFlow architecture for sequence labelling. The architecture is
    limited to the official Google TensorFlow implementation and cannot be mixed with Keras layers for 
    retraining. Training corresponds to a fine tuning only of a provided pre-trained model.

    BERT sequence labelling model with fine-tuning.

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

    def __init__(self, config, ntags=None):
        self.graph = tf.get_default_graph()

        print("config.max_sequence_length: ", config.max_sequence_length)
        print("config.batch_size: ", config.batch_size)

        self.model_name = config.model_name
        self.model_type = config.model_type

        print(self.model_name)
        print(self.model_type)

        # we get the BERT pretrained files from the embeddings registry
        description = _get_description(self.model_type)

        if description is None:
            raise Exception('no embeddings description found for ' + self.model_type)

        self.fold_count = config.fold_number

        self.config_file = description["path-config"]
        self.weight_file = description["path-weights"] # init_checkpoint
        self.vocab_file = description["path-vocab"]

        self.labels = []
        # by convention, PAD is zero
        self.labels.append("[PAD]")
        for key in config.labels:
            self.labels.append(key)

        # adding other default convention labels if necessary
        # adding other default convention labels if necessary
        if "X" not in self.labels:
            self.labels.append("X")
        if "[CLS]" not in self.labels:
            self.labels.append("[CLS]")
        if "[SEP]" not in self.labels:
            self.labels.append("[SEP]")

        self.do_lower_case = False
        self.max_seq_length = config.max_sequence_length
        self.train_batch_size = config.batch_size
        self.predict_batch_size = config.batch_size
        self.learning_rate = 2e-5 
        self.num_train_epochs = 1.0
        self.warmup_proportion = 0.1
        self.master = None
        self.save_checkpoints_steps = 99999999 # <----- don't want to save any checkpoints
        self.iterations_per_loop = 1000

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.processor = NERProcessor(self.labels)

        self.bert_config = modeling.BertConfig.from_json_file(self.config_file)
        self.model_dir = 'data/models/textClassification/' + self.model_name    

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

        train_examples = self.processor.get_train_examples(x_train, y_train)

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
        Train the seuqnce labelling model
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
        # (given that there is a 1.3 GB file, it's better!) 
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.data-00000-of-00001")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.index")
        if os.path.exists(garbage):
            os.remove(garbage)
        garbage = os.path.join(self.model_dir+str(fold_number), "model.ckpt-0.meta")
        if os.path.exists(garbage):
            os.remove(garbage)

    def evaluate(self, x_test=None, y_test=None, run_number=0):
        '''
        Train and eval the nb_runs model(s) against holdout set. If nb_runs>1, the final
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

    def evaluate_fold(self, predict_examples, fold_number=0):
        
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

        model_fn = self.model_fn_builder(
              self.bert_config,
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
              predict_batch_size=self.predict_batch_size)

        result = estimator.predict(input_fn=predict_input_fn)
        
        y_pred = np.zeros(shape=(len(predict_examples),len(self.labels)))

        p = 0
        for prediction in result:
            probabilities = prediction["probabilities"]
            q = 0
            for class_probability in probabilities:
                y_pred[p,q] = class_probability
                q += 1
            p += 1
        
        # cleaning the garbages
        os.remove(predict_file)

        return y_pred

    def predict_on_batch(self, texts, fold_number=0):
        if self.loaded_estimator is None:
            self.load_model(fold_number)        

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
                    y_pred[y_pos+p,q] = class_probability 
                    q += 1
                p += 1
            y_pos += num_current_batch

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
                     params):  # pylint: disable=unused-argument
            """
            The `model_fn` for TPUEstimator.
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
            (loss, logits, predicts) = self.create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, len(label_list), use_one_hot_embeddings)
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

    def create_model(self, bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
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
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, num_labels])
            # mask = tf.cast(input_mask,tf.float32)
            # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
            # return (loss, logits, predict)
            ##########################################################################
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predicts = tf.argmax(probabilities, axis=-1)
            return (loss, logits, predicts)
            ##########################################################################

    def load_model(self):
        # default
        num_train_steps = int(10000 / self.train_batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        model_fn = model_fn_builder(
              bert_config=self.bert_config,
              label_list=self.labels,
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


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # int32 cast 
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        
        return example

    def input_fn(params):
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
          tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

def file_based_convert_examples_to_features_old(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    print(output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_ids])
        #features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()

def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels


def _get_description(name, path="./embedding-registry.json"):
    print(name)
    print(path)
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["embeddings-contextualized"]:
        if emb["name"] == name:
            return emb
    return None


