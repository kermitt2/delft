# -*- coding: utf-8 -*-
import os
import codecs
import re
import numpy as np
import cairocffi as cairo
from keras.preprocessing import image
from scipy import ndimage
import scipy.misc

regex = r'^[a-z ]+$'

def is_valid_str(in_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(in_str))

fdir = os.path.dirname(__file__);

class TrainingDataGenerator:

    # this creates larger "blotches" of noise which look
    # more realistic than just adding gaussian noise
    # assumes greyscale with pixels ranging from 0 to 1
    def speckle(self, img):
        severity = np.random.uniform(0, 0.6)
        blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
        img_speck = (img + blur)
        img_speck[img_speck > 1] = 1
        img_speck[img_speck <= 0] = 0
        return img_speck

    # paints the text using cairo, applies random font and introduces some noise
    def paint_text(self, text, w, h, rotate=False, ud=False, multi_fonts=False):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
        with cairo.Context(surface) as context:
            context.set_source_rgb(1, 1, 1)  # White
            context.paint()
            # this font list works in CentOS 7
            if multi_fonts:
                fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
                context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                         np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
            else:
                context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            context.set_font_size(25)
            box = context.text_extents(text)
            border_w_h = (4, 4)
            if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                raise IOError('Could not fit string into image. Max char count is too large for given image width.')

            # teach the RNN translational invariance by
            # fitting text box randomly on canvas, with some room to rotate
            max_shift_x = w - box[2] - border_w_h[0]
            max_shift_y = h - box[3] - border_w_h[1]
            top_left_x = np.random.randint(0, int(max_shift_x))
            if ud:
                top_left_y = np.random.randint(0, int(max_shift_y))
            else:
                top_left_y = h // 2
            context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
            context.set_source_rgb(0, 0, 0)
            context.show_text(text)

        buf = surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (h, w, 4)
        a = a[:, :, 0]  # grab single channel
        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)
        if rotate:
            a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
        a = self.speckle(a)

        return a

    def generateTrainingData(self, num_words = 16000, mono_fraction = 1, img_h = 64, img_w = 128, max_string_len = 4,
                             monogram_file = os.path.join(fdir, 'wordlists/wordlist_mono_clean.txt'), bigram_file = os.path.join(fdir, 'wordlists/wordlist_bi_clean.txt'),
                             output_dir = os.path.join('training_data')):

        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        if(not os.path.exists(monogram_file) or not os.path.exists(bigram_file)):
            raise IOError('Could not find paths for monogram and bigram files. ')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

        tmp_string_list = [];
        print("Reading monogram and bigram text files.")
        # monogram file is sorted by frequency in english speech
        with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
            for line in f:
                if len(tmp_string_list) == int(num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')

        string_list = [''] * num_words

        # interlace to mix up the easy and hard words
        # this division by 2 should be done using the mono_fraction
        string_list[::2] = tmp_string_list[:num_words // 2]
        string_list[1::2] = tmp_string_list[num_words // 2:]

        output_dir = os.path.join('training_data')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving training data")
        for i, word in enumerate(string_list):
            filename = 'e%02d' % (i)
            file = codecs.open(os.path.join(output_dir, filename + '.txt'), "w", "utf-8")
            file.write(word)
            file.close()

            # here change the paint_text parameter
            scipy.misc.imsave(os.path.join(output_dir, filename + '.png'),
                              self.paint_text(word, img_w, img_h, rotate=False, ud=False, multi_fonts=False)[0, :, :])
            print(".")

if __name__ == '__main__':
    generator = TrainingDataGenerator()
    generator.generateTrainingData();