import os
from delft.sequenceLabelling.data_generator import DataGenerator
from keras.optimizers import Adam
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# seqeval
from delft.sequenceLabelling.evaluation import accuracy_score
from delft.sequenceLabelling.evaluation import classification_report
from delft.sequenceLabelling.evaluation import f1_score, accuracy_score, precision_score, recall_score

import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)


class Trainer(object):

    def __init__(self,
                 model,
                 models,
                 embeddings,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 preprocessor=None
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

    """ train the instance self.model """
    def train(self, x_train, y_train, x_valid, y_valid):
        self.model.summary()
        #print("self.model_config.use_crf:", self.model_config.use_crf)

        if self.model_config.use_crf:
            self.model.compile(loss=self.model.crf.loss,
                           optimizer='adam')
        else:
            self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam')
                           #optimizer=Adam(lr=self.training_config.learning_rate))
        # uncomment to plot graph
        #plot_model(self.model, 
        #    to_file='data/models/sequenceLabelling/'+self.model_config.model_name+'_'+self.model_config.model_type+'.png')
        self.model = self.train_model(self.model, x_train, y_train, x_valid, y_valid, 
                                                  self.training_config.max_epoch)

    """ parameter model local_model must be compiled before calling this method 
        this model will be returned with trained weights """
    def train_model(self, local_model, x_train, y_train, x_valid=None, y_valid=None, max_epoch=50):
        # todo: if valid set if None, create it as random segment of the shuffled train set 

        if self.training_config.early_stop:
            training_generator = DataGenerator(x_train, y_train, 
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True)

            validation_generator = DataGenerator(x_valid, y_valid,  
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False)

            callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=True,
                                      patience=self.training_config.patience,
                                      valid=(validation_generator, self.preprocessor))
        else:
            x_train = np.concatenate((x_train, x_valid), axis=0)
            y_train = np.concatenate((y_train, y_valid), axis=0)
            training_generator = DataGenerator(x_train, y_train, 
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True)

            callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=False)
        nb_workers = 6
        multiprocessing = True
        # multiple workers will not work with ELMo due to GPU memory limit (with GTX 1080Ti 11GB)
        if self.embeddings.use_ELMo or self.embeddings.use_BERT:
            # worker at 0 means the training will be executed in the main thread
            nb_workers = 0 
            multiprocessing = False
            # dump token context independent data for train set, done once for the training

        local_model.fit_generator(generator=training_generator,
                                epochs=max_epoch,
                                use_multiprocessing=multiprocessing,
                                workers=nb_workers,
                                callbacks=callbacks)

        return local_model

    """ n-fold training for the instance model 
        the n models are stored in self.models, and self.model left unset at this stage """
    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None):
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

                val_x = x_train[fold_start:fold_end]
                val_y = y_train[fold_start:fold_end]
            else:
                # reuse given segmentation
                train_x = x_train
                train_y = y_train

                val_x = x_valid
                val_y = y_valid

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
                                    val_x, 
                                    val_y,
                                    max_epoch=self.training_config.max_epoch)
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

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.2f}']) + '.h5'
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
        self.evaluation = evaluation

    def on_epoch_end(self, epoch, logs={}):
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true_batch = label
            y_true_batch = np.argmax(y_true_batch, -1)
            sequence_lengths = data[-1] # shape of (batch_size, 1)
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
        f1 = f1_score(y_true, y_pred)
        print("\tf1 (micro): {:04.2f}".format(f1 * 100))

        if self.evaluation:
            self.accuracy = accuracy_score(y_true, y_pred)
            self.precision = precision_score(y_true, y_pred)
            self.recall = recall_score(y_true, y_pred)
            self.report = classification_report(y_true, y_pred, digits=4)
            print(self.report)

        # save eval
        logs['f1'] = f1
        self.f1 = f1
