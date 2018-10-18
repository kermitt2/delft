# -*- coding: utf-8 -*-
import os
import codecs

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

import cairocffi as cairo

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import pylab
import itertools
import editdistance
import numpy as np
import re

from scipy import ndimage

from traindatagen import TrainingDataGenerator

alphabet = u'□◆⚬♡♢abcdefghijklmnopqrstuvwxyz '

TRAINING_DATA_DIR = 'training_data'

VALIDATION_OUTPUT_DIR = 'validation'

WEIGHTS_OUTPUT_DIR = 'weights'


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func

        self.validation_output_dir = os.path.join(
            VALIDATION_OUTPUT_DIR, run_name)
        if not os.path.exists(self.validation_output_dir):
            os.makedirs(self.validation_output_dir)

        self.weights_output_dir = os.path.join(
            WEIGHTS_OUTPUT_DIR, run_name)
        if not os.path.exists(self.weights_output_dir):
            os.makedirs(self.weights_output_dir)

        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        print("on_epoch_end")
        self.model.save_weights(os.path.join(self.weights_output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        print(res)
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.validation_output_dir, 'e%02d.png' % (epoch)))
        pylab.close()

        #Add here callback for saving traing data.


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    print("shuffle_mats_or_lists")
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret

# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck

class TraningDataLoader(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    @staticmethod
    def get_output_size():
        return len(alphabet) + 1

    def get_expected_text(self, i):
        filename = 'e%02d' % (i)
        path_monogram = os.path.join(self.monogram_file, filename + '.txt')
        path_bigram = os.path.join(self.bigram_file, filename + '.txt')

        if os.path.isfile(path_monogram):
            file = codecs.open(path_monogram, mode='r', encoding='utf-8')
            for line in file:
                word = line.rstrip()
        elif os.path.isfile(path_bigram):
            file = codecs.open(path_bigram, mode='r', encoding='utf-8')
            for line in file:
                word = line.rstrip()
        return word


    def get_image_input(self, i, h, w):
        filename = 'e%02d' % (i)
        path_monogram = os.path.join(self.monogram_file, filename + '.png')
        path_bigram = os.path.join(self.bigram_file, filename + '.png')
        if os.path.isfile(path_monogram):
            image_surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h).create_from_png(path_monogram)
        elif os.path.isfile(path_bigram):
            image_surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h).create_from_png(path_bigram)

        buf = image_surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (h, w, 4)
        a = a[:, :, 0]  # grab single channel
        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)
        # if rotate:
        #     a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
        a = speckle(a)

        return a



    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words):
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words

        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        self.input_image = np.ones([self.num_words, self.img_w, self.img_h, 1])

        for i in range(self.num_words):
            expected_text = self.get_expected_text(i)
            self.Y_len[i] = len(expected_text)
            self.Y_data[i, 0:len(expected_text)] = text_to_labels(expected_text)
            self.X_text.append(expected_text)
            self.input_image[i, 0:self.img_w, :, 0] = self.get_image_input(i, self.img_h, self.img_w)[0, :, :].T
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
        self.cur_val_index = self.val_split
        self.cur_train_index = 0


    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            # if train and i > size - 4:
            #     if K.image_data_format() == 'channels_first':
            #         X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
            #     else:
            #         X_data[i, 0:self.img_w, :, 0] = self.paint_func('', )[0, :, :].T
            #     labels[i, 0] = self.blank_label
            #     input_length[i] = self.img_w // self.downsample_factor - 2
            #     label_length[i] = 1
            #     source_str.append('')
            # else:
            # if K.image_data_format() == 'channels_first':
            #     X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
            # else:
            labels[i, :] = self.Y_data[index + i]
            X_data[i, 0:self.img_w, :, 0] = self.input_image[index + i, 0:self.img_w, :, 0]
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = self.Y_len[index + i]
            source_str.append(self.X_text[index + i])

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        print("next_train")
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len, self.input_image) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len, self.input_image], self.val_split)
            yield ret

    def next_val(self):
        print("next_val")

        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        print("on_train_begin")
        self.build_word_list(16000)

    def on_epoch_begin(self, epoch, logs={}):
        print("on_epoch_begin")
        # rebind the paint function to implement curriculum learning
        # if 3 <= epoch < 6:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=False, ud=True, multi_fonts=False)
        # elif 6 <= epoch < 9:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=False, ud=True, multi_fonts=True)
        # elif epoch >= 9:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21:
            self.build_word_list(32000)


# Optical Character Recognition Engine in Keras, Unicode OCR
class UOCR:

    def __init__(self, img_w=512):
        self.img_h = 64
        self.img_w = img_w

        # Network parameters
        self.conv_num_filters = 16
        self.kernel_size = (3, 3)
        self.pool_size = 2
        self.time_dense_size = 32
        self.rnn_size = 512
        self.absolute_max_string_len = 16

        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.img_w, self.img_h)
        else:
            input_shape = (self.img_w, self.img_h, 1)

        act = 'relu'
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(self.conv_num_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max1')(inner)
        inner = Conv2D(self.conv_num_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max2')(inner)

        conv_to_rnn_dims = (self.img_w // (self.pool_size ** 2), (self.img_h // (self.pool_size ** 2)) * self.conv_num_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)

        # transforms RNN output to character activations:
        # this is the output layer
        inner = Dense(TraningDataLoader.get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        self.model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        # captures output of softmax so we can decode the output during visualization
        self.test_func = K.function([input_data], [y_pred])


    def loadweights(self, weightsfile='densified_labeltype_best.h5'):
        if weightsfile:
            self.model.load_weights(weightsfile)

    def train(self, run_name, start_epoch, stop_epoch, words_per_epoch=16000, val_split = 0.2, minibatch_size = 32, verbose=1):

        fdir = os.path.dirname(__file__)
        training_data = os.path.join(fdir, TRAINING_DATA_DIR)

        val_words = int(words_per_epoch * (val_split))

        train_gen = TraningDataLoader(monogram_file=training_data,
                                      bigram_file=training_data,
                                      minibatch_size=minibatch_size,
                                      img_w=self.img_w,
                                      img_h=self.img_h,
                                      downsample_factor=(self.pool_size ** 2),
                                      val_split=words_per_epoch - val_words
                                      )

        weight_file = os.path.join(WEIGHTS_OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        if start_epoch > 0 and os.path.exists(weight_file):
            self.model.load_weights(weight_file)

        # This callback is used to save validation data files.
        viz_cb = VizCallback(run_name, self.test_func, train_gen.next_val())

        self.model.fit_generator(generator=train_gen.next_train(),
                            steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
                            epochs=stop_epoch,
                            validation_data=train_gen.next_val(),
                            validation_steps=val_words // minibatch_size,
                            callbacks=[viz_cb, train_gen],
                            initial_epoch=start_epoch)
    #
    def ocr_frompic(self, image, w = 128, h = 64):
        if os.path.isfile(image):
            image_surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h).create_from_png(image)
        buf = image_surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (h, w, 4)
        a = a[:, :, 0]  # grab single channel

        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)

        Ximage = np.ones([1, w, h, 1])
        Ximage[0, 0:w, :, 0] = a[0,:,:].T

        # filename = 'e%02d' % (000)
        # scipy.misc.imsave(os.path.join(VALIDATION_OUTPUT_DIR, filename + '.png'), Ximage[0, 0:w, :, 0])
        return decode_batch(self.test_func, Ximage[0:1])

    def test_batch(self, words = ["test", "eds", "azerty", "and", "sDzeq", "qpsqd"], rotate=True, ud=True, multi_fonts=True):
        print("test_batch")
        num_words = len(words)
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([num_words, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([num_words, self.img_w, self.img_h, 1])

        datagenerator = TrainingDataGenerator()
        for i in range(num_words):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if K.image_data_format() == 'channels_first':
                X_data[i, 0, 0:self.img_w, :] = datagenerator.paint_text(words[i], self.img_w, self.img_h,
                                                  rotate=rotate, ud=ud, multi_fonts=multi_fonts)[0, :, :].T
            else:
                X_data[i, 0:self.img_w, :, 0] = datagenerator.paint_text(words[i], self.img_w, self.img_h,
                                                  rotate=rotate, ud=ud, multi_fonts=multi_fonts)[0, :, :].T

        res = decode_batch(self.test_func, X_data[0:num_words])
        print(res)
        if X_data[0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(num_words):
            pylab.subplot(num_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = X_data[i, 0, :, :]
            else:
                the_input = X_data[i, :, :, 0]
            # filename = 'e%02d' % i
            # scipy.misc.imsave(os.path.join(VALIDATION_OUTPUT_DIR, filename + '.png'), the_input)
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (words[i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(VALIDATION_OUTPUT_DIR, 'batch_text.png'))
        pylab.close()