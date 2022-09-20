import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from transformers import create_optimizer

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.data_generator import DataGeneratorTransformers
from delft.sequenceLabelling.evaluation import f1_score, accuracy_score, precision_score, recall_score
from delft.sequenceLabelling.evaluation import get_report, compute_metrics
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import Preprocessor
from delft.utilities.Transformer import TRANSFORMER_CONFIG_FILE_NAME, DEFAULT_TRANSFORMER_TOKENIZER_DIR
from delft.utilities.misc import print_parameters

DEFAULT_WEIGHT_FILE_NAME = 'model_weights.hdf5'
CONFIG_FILE_NAME = 'config.json'
PROCESSOR_FILE_NAME = 'preprocessor.json'

class Trainer(object):

    def __init__(self,
                 model,
                 models,
                 embeddings,
                 model_config: ModelConfig,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 preprocessor: Preprocessor=None,
                 transformer_preprocessor=None
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
        self.transformer_preprocessor = transformer_preprocessor

    def train(self, x_train, y_train, x_valid, y_valid, features_train: np.array = None, features_valid: np.array = None, callbacks=None):
        """
        Train the instance self.model
        """      
        self.model = self.compile_model(self.model, len(x_train))

        # uncomment to plot graph
        #plot_model(self.model,
        #    to_file='data/models/sequenceLabelling/'+self.model_config.model_name+'_'+self.model_config.architecture+'.png')

        self.model = self.train_model(self.model, x_train, y_train, x_valid=x_valid, y_valid=y_valid,
                                  f_train=features_train, f_valid=features_valid,
                                  max_epoch=self.training_config.max_epoch, callbacks=callbacks)

    def compile_model(self, local_model, train_size):

        nb_train_steps = (train_size // self.training_config.batch_size) * self.training_config.max_epoch
        
        if self.model_config.transformer_name is not None:
            # we use a transformer layer in the architecture
            optimizer, lr_schedule = create_optimizer(
                init_lr=2e-5, 
                num_train_steps=nb_train_steps,
                weight_decay_rate=0.01,
                num_warmup_steps=0.1*nb_train_steps,
            )

            if local_model.config.use_chain_crf:
                local_model.compile(optimizer=optimizer, loss=local_model.crf.sparse_crf_loss_bert_masked)
            elif local_model.config.use_crf:
                # loss is calculated by the custom CRF wrapper
                local_model.compile(optimizer=optimizer)
            else:
                # we apply a mask on the predicted labels so that the weights 
                # corresponding to special symbols are neutralized
                local_model.compile(optimizer=optimizer, loss=sparse_crossentropy_masked)
        else:
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.training_config.learning_rate,
                decay_steps=nb_train_steps,
                decay_rate=0.1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            #optimizer = tf.keras.optimizers.Adam(self.training_config.learning_rate)
            if local_model.config.use_chain_crf:
                local_model.compile(optimizer=optimizer, loss=local_model.crf.loss)
            elif local_model.config.use_crf:
                if tf.executing_eagerly():
                    # loss is calculated by the custom CRF wrapper, no need to specify a loss function here
                    local_model.compile(optimizer=optimizer)
                else:
                    print("compile model, graph mode")
                    # always expecting a loss function here, but it is calculated internally by the CRF wapper
                    # the following will fail in graph mode because 
                    # '<tf.Variable 'chain_kernel:0' shape=(10, 10) dtype=float32> has `None` for gradient.'
                    # however this variable cannot be accessed, so no soluton for the moment 
                    # (probably need not using keras fit and creating a custom training loop to get the gradient)
                    local_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
                    #local_model.compile(optimizer=optimizer, loss=InnerLossPusher(local_model))
            else:
                # only sparse label encoding is used (no one-hot encoded labels as it was the case in DeLFT < 0.3)
                local_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        return local_model

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
                bert_preprocessor=self.transformer_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, 
                shuffle=True, features=f_train, use_chain_crf=self.model_config.use_chain_crf)

            validation_generator = generator(x_valid, y_valid,  
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                bert_preprocessor=self.transformer_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False, features=f_valid, 
                output_input_offsets=True, use_chain_crf=self.model_config.use_chain_crf)

            _callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=True,
                                      patience=self.training_config.patience,
                                      valid=(validation_generator, self.preprocessor), use_crf=self.model_config.use_crf,
                                      use_chain_crf=self.model_config.use_chain_crf)
        else:
            x_train = np.concatenate((x_train, x_valid), axis=0)
            y_train = np.concatenate((y_train, y_valid), axis=0)
            feature_all = None
            if f_train is not None:
                feature_all = np.concatenate((f_train, f_valid), axis=0)

            training_generator = generator(x_train, y_train,
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor, 
                bert_preprocessor=self.transformer_preprocessor,
                char_embed_size=self.model_config.char_embedding_size, 
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True, 
                features=feature_all, use_chain_crf=self.model_config.use_chain_crf)

            _callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=False,
                                      use_crf=self.model_config.use_crf,
                                      use_chain_crf=self.model_config.use_chain_crf)
        _callbacks += (callbacks or [])
        nb_workers = 6
        multiprocessing = self.training_config.multiprocessing

        # multiple workers should work with transformer layers, but not with ELMo due to GPU memory limit (with GTX 1080Ti 11GB)
        if self.model_config.transformer_name is not None or (self.embeddings and self.embeddings.use_ELMo):
            # worker at 0 means the training will be executed in the main thread
            nb_workers = 0 
            multiprocessing = False

        local_model.fit(training_generator,
                                epochs=max_epoch,
                                use_multiprocessing=multiprocessing,
                                workers=nb_workers,
                                callbacks=_callbacks)

        return local_model

    def train_nfold(self, x_train, y_train, x_valid=None, y_valid=None, f_train=None, f_valid=None, callbacks=None):
        """
        n-fold training for the instance model

        for RNN models:
        -> the n models are stored in self.models, and self.model left unset at this stage
        fold number is available with self.model_config.fold_number 

        for models with transformer layer:
        -> fold models are saved on disk (because too large) and self.models is not used, we identify the usage
        of folds with self.model_config.fold_number     
        """

        fold_count = self.model_config.fold_number
        fold_size = len(x_train) // fold_count

        dir_path = 'data/models/sequenceLabelling/'
        output_directory = os.path.join(dir_path, self.model_config.model_name)
        print("Output directory:", output_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if self.model_config.transformer_name is not None:
            # save the config, preprocessor and transformer layer config on disk
            self.model_config.save(os.path.join(output_directory, CONFIG_FILE_NAME))
            self.preprocessor.save(os.path.join(output_directory, PROCESSOR_FILE_NAME))

        for fold_id in range(0, fold_count):
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

            foldModel = get_model(self.model_config, 
                               self.preprocessor, 
                               ntags=len(self.preprocessor.vocab_tag), 
                               load_pretrained_weights=True)

            if fold_id == 0:
                print_parameters(self.model_config, self.training_config)
                foldModel.print_summary()

            print('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')
            self.transformer_preprocessor = foldModel.transformer_preprocessor
            foldModel = self.compile_model(foldModel, len(train_x))
            foldModel = self.train_model(foldModel, 
                                    train_x,
                                    train_y,
                                    x_valid=val_x,
                                    y_valid=val_y,
                                    f_train=train_f,
                                    f_valid=val_f,
                                    max_epoch=self.training_config.max_epoch,
                                    callbacks=callbacks)

            if self.model_config.transformer_name is None:
                self.models.append(foldModel)
            else:
                # save the model with transformer layer on disk
                weight_file = DEFAULT_WEIGHT_FILE_NAME.replace(".hdf5", str(fold_id)+".hdf5")
                foldModel.save(os.path.join(output_directory, weight_file))
                if fold_id == 0:
                    foldModel.transformer_config.to_json_file(os.path.join(output_directory, TRANSFORMER_CONFIG_FILE_NAME))
                    if self.model_config.transformer_name is not None:
                        transformer_preprocessor = foldModel.transformer_preprocessor
                        transformer_preprocessor.tokenizer.save_pretrained(os.path.join(output_directory, DEFAULT_TRANSFORMER_TOKENIZER_DIR))


def get_callbacks(log_dir=None, valid=(), eary_stopping=True, patience=5, use_crf=True, use_chain_crf=False):
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
        callbacks.append(Scorer(*valid, use_crf=use_crf, use_chain_crf=use_chain_crf))

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

    def __init__(self, validation_generator, preprocessor=None, evaluation=False, use_crf=False, use_chain_crf=False):
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
        self.use_crf = use_crf
        self.use_chain_crf = use_chain_crf

    def on_epoch_end(self, epoch, logs={}):
        y_pred = None
        y_true = None
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true_batch = label       

            if isinstance(self.valid_batches, DataGeneratorTransformers):
                y_true_batch = np.asarray(y_true_batch, dtype=object)

                # we need to remove one vector of the data corresponding to the token offsets, this vector is not 
                # expected by the model, but we need it to restore correctly the labels (which are produced
                # according to the sub-segmentation of wordpiece, not the expected segmentation)
                input_offsets = data[-1]
                data = data[:-1]

                y_pred_batch = self.model.predict_on_batch(data)

                if not self.use_crf:
                    y_pred_batch = np.argmax(y_pred_batch, -1)

                if self.use_chain_crf:
                    y_pred_batch = np.argmax(y_pred_batch, -1)

                # results have been produced by a model using a transformer layer, so a few things to do
                # the labels are sparse, so integers and not one hot encoded
                # we need to restore back the labels for wordpiece to the labels for normal tokens
                # for this we can use the marked tokens provided by the generator 
                new_y_pred_batch = []
                new_y_true_batch = []
                for y_pred_text, y_true_text, offsets_text in zip(y_pred_batch, y_true_batch, input_offsets):
                    new_y_pred_text = []
                    new_y_true_text = []
                    # this is the result per sequence, realign labels:
                    for q in range(len(offsets_text)):
                        if offsets_text[q][0] == 0 and offsets_text[q][1] == 0:
                            # special token
                            continue
                        if offsets_text[q][0] != 0: 
                            # added sub-token
                            continue
                        new_y_pred_text.append(y_pred_text[q]) 
                        new_y_true_text.append(y_true_text[q])
                    new_y_pred_batch.append(new_y_pred_text)
                    new_y_true_batch.append(new_y_true_text)
                y_pred_batch = new_y_pred_batch
                y_true_batch = new_y_true_batch

                y_true_batch = [self.p.inverse_transform(y) for y in y_true_batch]
                y_pred_batch = [self.p.inverse_transform(y) for y in y_pred_batch]
            else:
                # no transformer layer around, no mess to manage with the sub-tokenization...
                y_pred_batch = self.model.predict_on_batch(data)

                if not self.use_crf:
                    # one hot encoded predictions
                    y_pred_batch = np.argmax(y_pred_batch, -1)

                if self.use_chain_crf:
                    # one hot encoded predictions and labels
                    y_pred_batch = np.argmax(y_pred_batch, -1)
                    y_true_batch = np.argmax(y_true_batch, -1)

                # we also have the input length available 
                sequence_lengths = data[-1] # this is the vectors "length_input" of the models input, always last 
                # shape of (batch_size, 1), we want (batch_size)
                sequence_lengths = np.reshape(sequence_lengths, (-1,))

                y_pred_batch = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred_batch, sequence_lengths)]
                y_true_batch = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true_batch, sequence_lengths)]

            if i == 0:
                y_pred = y_pred_batch
                y_true = y_true_batch
            else:
                y_pred.extend(y_pred_batch)
                y_true.extend(y_true_batch)

        '''
        for i in range(0,len(y_pred)):
            print("pred", y_pred[i])
            print("true", y_true[i])
        '''
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


def sparse_crossentropy_masked(y_true, y_pred):
    mask_value = 0
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))

