import os

import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

#from delft.sequenceLabelling.data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# seqeval
from delft.sequenceLabelling.evaluation import accuracy_score, get_report, compute_metrics
from delft.sequenceLabelling.evaluation import classification_report
from delft.sequenceLabelling.evaluation import f1_score, accuracy_score, precision_score, recall_score

import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
import tensorflow as tf


def sparse_crossentropy_masked(y_true, y_pred):
    mask_value = 0
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, 
y_pred_masked))

def sparse_categorical_accuracy_masked(y_true, y_pred):
    mask_value = 0
    active_loss = tf.reshape(y_true, (-1,)) != mask_value
    reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, shape_list(y_pred)[2])), active_loss)
    y_true = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
    reduced_logits = tf.cast(tf.argmax(reduced_logits, axis=-1), tf.keras.backend.floatx())
    equality = tf.equal(y_true, reduced_logits)
    return tf.reduce_mean(tf.cast(equality, tf.keras.backend.floatx()))

def crossentropy_masked(y_true, y_pred):
    mask_value = 0
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_masked, 
y_pred_masked))

def categorical_accuracy_masked(y_true, y_pred):
    mask_value = 0
    active_loss = tf.reshape(y_true, (-1,)) != mask_value
    reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, shape_list(y_pred)[2])), active_loss)
    y_true = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
    reduced_logits = tf.cast(tf.argmax(reduced_logits, axis=-1), tf.keras.backend.floatx())
    equality = tf.equal(y_true, reduced_logits)
    return tf.reduce_mean(tf.cast(equality, tf.keras.backend.floatx()))

class Trainer(object):

    def __init__(self,
                 model,
                 models,
                 embeddings,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 preprocessor=None, 
                 bert_preprocessor=None
                 ):

        # for single model training
        self.model = model

        # for n-folds training
        self.models = models

        self.embeddings = embeddings
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.preprocessor = preprocessor
        self.bert_preprocessor = bert_preprocessor

    def train(self, x_train, y_train, x_valid, y_valid, features_train: np.array = None, features_valid: np.array = None, callbacks=None):
        """
        Train the instance self.model
        """      
        self.model.summary()
        if self.model_config.use_crf:
            self.model.compile(loss=self.model.crf.loss,
                           optimizer='adam')
        elif self.bert_preprocessor != None:
            self.model.compile(optimizer=Adam(learning_rate=5e-5), loss=sparse_crossentropy_masked)
        else:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            #self.model.compile(loss='categorical_crossentropy', optimizer='adam')
                           #optimizer=Adam(lr=self.training_config.learning_rate))
        # uncomment to plot graph
        #plot_model(self.model,
        #    to_file='data/models/sequenceLabelling/'+self.model_config.model_name+'_'+self.model_config.architecture+'.png')
        self.model = self.train_model(self.model, x_train, y_train, x_valid=x_valid, y_valid=y_valid,
                                  f_train=features_train, f_valid=features_valid,
                                  max_epoch=self.training_config.max_epoch, callbacks=callbacks)

    def train_model(self, local_model, x_train, y_train, f_train=None,
                    x_valid=None, y_valid=None, f_valid=None, max_epoch=50, callbacks=None):
        """
        The parameter model local_model must be compiled before calling this method.
        This model will be returned with trained weights
        """
        # todo: if valid set is None, create it as random segment of the shuffled train set

        generator = local_model.get_generator()
        if self.training_config.early_stop:
            training_generator = generator(x_train, y_train, 
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                bert_preprocessor=self.bert_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True, features=f_train)

            validation_generator = generator(x_valid, y_valid,  
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                bert_preprocessor=self.bert_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False, features=f_valid)

            _callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=True,
                                      patience=self.training_config.patience,
                                      valid=(validation_generator, self.preprocessor))
        else:
            x_train = np.concatenate((x_train, x_valid), axis=0)
            y_train = np.concatenate((y_train, y_valid), axis=0)
            feature_all = None
            if f_train is not None:
                feature_all = np.concatenate((f_train, f_valid), axis=0)

            training_generator = generator(x_train, y_train,
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                bert_preprocessor=self.bert_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True, features=feature_all)

            _callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=False)
        _callbacks += (callbacks or [])
        nb_workers = 6
        multiprocessing = self.training_config.multiprocessing
        # multiple workers will not work with ELMo due to GPU memory limit (with GTX 1080Ti 11GB)
        
        local_model.fit(training_generator,
                                epochs=max_epoch,
                                use_multiprocessing=multiprocessing,
                                workers=nb_workers,
                                callbacks=_callbacks)

        return local_model

    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, f_train=None, f_valid=None, callbacks=None):
        """
        n-fold training for the instance model
        the n models are stored in self.models, and self.model left unset at this stage
        """

        fold_count = len(self.models)
        fold_size = len(x_train) // fold_count
        #roc_scores = []

        for fold_id in range(0, fold_count):
            print('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')

            if x_valid is None:
                # segment train and valid
                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size

                if fold_id == fold_size - 1:
                    fold_end = len(x_train)

                train_x = np.concatenate([x_train[:fold_start], x_train[fold_end:]])
                train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])
                train_f = np.concatenate([f_train[:fold_start], f_train[fold_end:]])

                val_x = x_train[fold_start:fold_end]
                val_y = y_train[fold_start:fold_end]
                val_f = f_train[fold_start:fold_end]
            else:
                # reuse given segmentation
                train_x = x_train
                train_y = y_train
                train_f = f_train

                val_x = x_valid
                val_y = y_valid
                val_f = f_valid

            foldModel = self.models[fold_id]
            foldModel.summary()

            if self.model_config.use_crf:
                foldModel.compile(loss=foldModel.crf.loss,
                               optimizer='adam')
            else:
                foldModel.compile(loss='categorical_crossentropy',
                               optimizer='adam')

            foldModel = self.train_model(foldModel, 
                                    train_x,
                                    train_y,
                                    x_valid=val_x,
                                    y_valid=val_y,
                                    f_train=train_f,
                                    f_valid=val_f,
                                    max_epoch=self.training_config.max_epoch,
                                    callbacks=callbacks)
            self.models[fold_id] = foldModel


def get_callbacks(log_dir=None, valid=(), eary_stopping=True, patience=5):
    """
    Get callbacks.

    Args:
        log_dir (str): the destination to save logs
        valid (tuple): data for validation.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if valid:
        callbacks.append(Scorer(*valid))

    if log_dir:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)

        file_name = '_'.join(['model_weights', '{epoch:02d}']) + '.h5'
        save_callback = ModelCheckpoint(os.path.join(log_dir, file_name),
                                        monitor='f1',
                                        save_weights_only=True)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=patience, mode='max'))

    return callbacks


class Scorer(Callback):

    def __init__(self, validation_generator, preprocessor=None, evaluation=False):
        """
        If evaluation is True, we produce a full evaluation with complete report, otherwise it is a
        validation step and we will simply produce f1 score
        """
        super(Scorer, self).__init__()
        self.valid_steps = len(validation_generator)
        self.valid_batches = validation_generator
        self.p = preprocessor

        self.f1 = -1.0
        self.accuracy = -1.0
        self.precision = -1.0
        self.recall = -1.0
        self.report = None
        self.report_as_map = None
        self.evaluation = evaluation

    def on_epoch_end(self, epoch, logs={}):
        y_pred = None
        y_true = None
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true_batch = label
            y_true_batch = np.argmax(y_true_batch, -1)
            sequence_lengths = data[-1] # this is the vectors "length_input" of the models input
            # shape of (batch_size, 1), we want (batch_size)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))

            y_pred_batch = self.model.predict_on_batch(data)
            y_pred_batch = np.argmax(y_pred_batch, -1)

            y_pred_batch = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred_batch, sequence_lengths)]
            y_true_batch = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true_batch, sequence_lengths)]

            if i == 0:
                y_pred = y_pred_batch
                y_true = y_true_batch
            else:    
                y_pred = y_pred + y_pred_batch
                y_true = y_true + y_true_batch 


        #for i in range(0,len(y_pred)):
        #    print("pred", y_pred[i])
        #    print("true", y_true[i])
        has_data = y_true is not None and y_pred is not None
        f1 = f1_score(y_true, y_pred) if has_data else 0.0
        print("\tf1 (micro): {:04.2f}".format(f1 * 100))

        if self.evaluation:
            self.accuracy = accuracy_score(y_true, y_pred) if has_data else 0.0
            self.precision = precision_score(y_true, y_pred) if has_data else 0.0
            self.recall = recall_score(y_true, y_pred) if has_data else 0.0
            self.report_as_map = compute_metrics(y_true, y_pred) if has_data else compute_metrics([], [])
            self.report = get_report(self.report_as_map, digits=4)
            print(self.report)

        # save eval
        logs['f1'] = f1
        self.f1 = f1
