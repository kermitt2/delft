import os
import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint

# based on https://github.com/Hironsan/anago/blob/master/anago/metrics.py

def get_callbacks(log_dir=None, valid=(), tensorboard=True, eary_stopping=True):
    """
    Get callbacks.

    Args:
        log_dir (str): the destination to save logs(for TensorBoard).
        valid (tuple): data for validation.
        tensorboard (bool): Whether to use tensorboard.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if log_dir and tensorboard:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)
        callbacks.append(TensorBoard(log_dir))

    if valid:
        callbacks.append(F1score(*valid))

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
        callbacks.append(EarlyStopping(monitor='f1', patience=5, mode='max'))

    return callbacks


def get_entities_with_offsets(seq, offsets):
    """
    Gets entities from sequence

    Args:
        seq (list): sequence of labels.
        offsets (list of integer pair): sequence of offset position

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> offsets = [(0,10), (11, 15), (16, 29), (30, 41)]
        >>> print(get_entities(seq))
        [('PER', 0, 2, 0, 15), ('LOC', 3, 4, 30, 41)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            start_pos = offsets[i][0]
            end_pos = offsets[j][1]
            chunks.append((types[i], i, j, start_pos, end_pos))
            i = j
        else:
            i += 1
    return chunks

def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks

class F1score(Callback):

    def __init__(self, validation_generator, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = len(validation_generator)
        self.valid_batches = validation_generator
        self.p = preprocessor

        self.f1 = -1.0
        self.correct_preds = 0
        self.total_correct = 0
        self.total_preds = 0

    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            sequence_lengths = data[-1] # shape of (batch_size, 1)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))
            #y_pred = np.asarray(self.model_.predict(data, sequence_lengths))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]

            a, b, c = self.count_correct_and_pred(y_true, y_pred, sequence_lengths)
            correct_preds += a
            total_preds += b
            total_correct += c

        f1 = self.calc_f1(correct_preds, total_correct, total_preds)
        print(' - f1: {:04.2f}'.format(f1 * 100))
        
        # save eval
        logs['f1'] = f1
        self.last_f1 = f1
        self.correct_preds = correct_preds
        self.total_correct = total_correct
        self.total_preds = total_preds

    def calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return f1

    def count_correct_and_pred(self, y_true, y_pred, sequence_lengths):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]

            lab_chunks = set(get_entities(lab))
            lab_pred_chunks = set(get_entities(lab_pred))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
        return correct_preds, total_correct, total_preds
