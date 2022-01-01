# -*- coding: utf-8 -*-
from __future__ import absolute_import
"""
Originally from Philipp Gross, https://github.com/phipleg/keras/blob/crf/keras/layers/crf.py

Tentatively migrated to Keras/tensorflow 2 by your DeLFT servitor.

Note: in this version, zero masking is not working with TF2, so do not use mask_zero=True in
your archiecture when using this CRF layer

Note: there are still a few tensorflow.keras.backend function usages, but it's probably ok.
If tensorflow drops the support of this keras.backend API, we would need to move some of the 
compatibility API behavior here for the few functions still using it. 
"""

from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
import sys

def path_energy(y, x, U, b_start=None, b_end=None, mask=None):
    """
    Calculates the energy of a tag path y for a given input x (with mask),
    transition energies U and boundary energies b_start, b_end.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)
    return path_energy0(y, x, U, mask)


def path_energy0(y, x, U, mask=None):
    """
    Path energy without boundary potential handling.
    """
    n_classes = tf.shape(x)[2]
    y_one_hot = tf.one_hot(y, n_classes)

    # Tag path energy
    energy = tf.math.reduce_sum(x * y_one_hot, 2)
    energy = tf.math.reduce_sum(energy, 1)

    # Transition energy
    y_t = y[:, :-1]
    y_tp1 = y[:, 1:]
    U_flat = tf.reshape(U, [-1])
    # Convert 2-dim indices (y_t, y_tp1) of U to 1-dim indices of U_flat:
    flat_indices = y_t * n_classes + y_tp1
    U_y_t_tp1 = tf.gather(U_flat, flat_indices)

    if mask is not None:
        mask = tf.cast(mask, tf.keras.backend.floatx())
        y_t_mask = mask[:, :-1]
        y_tp1_mask = mask[:, 1:]
        U_y_t_tp1 *= y_t_mask * y_tp1_mask

    energy += tf.math.reduce_sum(U_y_t_tp1, axis=1)

    return energy


def sparse_chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """
    Given the true sparsely encoded tag sequence y, input x (with mask),
    transition energies U, boundary energies b_start and b_end, it computes
    the loss function of a Linear Chain Conditional Random Field:
    loss(y, x) = NNL(P(y|x)), where P(y|x) = exp(E(y, x)) / Z.
    So, loss(y, x) = - E(y, x) + log(Z)
    Here, E(y, x) is the tag path energy, and Z is the normalization constant.
    The values log(Z) is also called free energy.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)
    energy = path_energy0(y, x, U, mask)
    energy -= free_energy0(x, U, mask)
    return tf.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """
    Variant of sparse_chain_crf_loss but with one-hot encoded tags y.
    """
    y_sparse = tf.math.argmax(y, -1)
    y_sparse = tf.cast(y_sparse, 'int32')
    return sparse_chain_crf_loss(y_sparse, x, U, b_start, b_end, mask)


def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
    """
    Given the observations x, it adds the start boundary energy b_start (resp.
    end boundary energy b_end on the start (resp. end) elements and multiplies
    the mask.
    """
    if mask is None:
        if b_start is not None:
            x = tf.keras.backend.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)
        if b_end is not None:
            x = tf.keras.backend.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
    else:
        mask = tf.cast(mask, tf.keras.backend.floatx())
        mask = tf.expand_dims(mask, 2)
        x *= mask
        if b_start is not None:
            mask_r = tf.keras.backend.concatenate([tf.zeros_like(mask[:, :1]), mask[:, :-1]], axis=1)
            start_mask = tf.cast(tf.math.greater(mask, mask_r), tf.keras.backend.floatx())
            x = x + start_mask * b_start
        if b_end is not None:
            mask_l = tf.keras.backend.concatenate([mask[:, 1:], tf.zeros_like(mask[:, -1:])], axis=1)
            end_mask = tf.cast(tf.math.greater(mask, mask_l), tf.keras.backend.floatx())
            x = x + end_mask * b_end
    return x


def viterbi_decode(x, U, b_start=None, b_end=None, mask=None):
    """
    Computes the best tag sequence y for a given input x, i.e. the one that
    maximizes the value of path_energy.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)

    alpha_0 = x[:, 0, :]
    gamma_0 = tf.zeros_like(alpha_0)
    initial_states = [gamma_0, alpha_0]
    _, gamma = _forward(x,
                        lambda B: [tf.cast(tf.math.argmax(B, axis=1), tf.keras.backend.floatx()), tf.math.reduce_max(B, axis=1)],
                        initial_states,
                        U,
                        mask)
    y = _backward(gamma, mask)
    return y


def free_energy(x, U, b_start=None, b_end=None, mask=None):
    """
    Computes efficiently the sum of all path energies for input x, when
    runs over all possible tag sequences.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)
    return free_energy0(x, U, mask)


def free_energy0(x, U, mask=None):
    """
    Free energy without boundary potential handling.
    """
    initial_states = [x[:, 0, :]]
    last_alpha, _ = _forward(x,
                             lambda B: [tf.math.reduce_logsumexp(B, axis=1)],
                             initial_states,
                             U,
                             mask)
    return last_alpha[:, 0]


def _forward(x, reduce_step, initial_states, U, mask=None):
    """
    Forward recurrence of the linear chain crf.
    """

    def _forward_step(energy_matrix_t, states):
        alpha_tm1 = states[-1]
        new_states = reduce_step(tf.expand_dims(alpha_tm1, 2) + energy_matrix_t)
        return new_states[0], new_states

    U_shared = tf.expand_dims(tf.expand_dims(U, 0), 0)

    if mask is not None:
        mask = tf.cast(mask, tf.keras.backend.floatx())
        mask_U = tf.expand_dims(tf.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
        U_shared = U_shared * mask_U

    inputs = tf.expand_dims(x[:, 1:, :], 2) + U_shared
    inputs = tf.keras.backend.concatenate([inputs, tf.zeros_like(inputs[:, -1:, :, :])], axis=1)

    last, values, _ = tf.keras.backend.rnn(_forward_step, inputs, initial_states)
    return last, values


def batch_gather(reference, indices):
    ref_shape = tf.shape(reference)
    batch_size = ref_shape[0]
    n_classes = ref_shape[1]
    flat_indices = tf.keras.backend.arange(0, batch_size) * n_classes + tf.keras.backend.flatten(indices)
    return tf.gather(tf.keras.backend.flatten(reference), flat_indices)


def _backward(gamma, mask):
    """
    Backward recurrence of the linear chain crf.
    """
    gamma = tf.cast(gamma, 'int32')

    def _backward_step(gamma_t, states):
        y_tm1 = tf.squeeze(states[0], [0])
        y_t = batch_gather(gamma_t, y_tm1)
        return y_t, [tf.expand_dims(y_t, 0)]

    initial_states = [tf.expand_dims(tf.zeros_like(gamma[:, 0, 0]), 0)]
    _, y_rev, _ = tf.keras.backend.rnn(_backward_step,
                        gamma,
                        initial_states,
                        go_backwards=True)
    # note the axis integer changed to [axis], see tf.keras.backend
    y = tf.reverse(y_rev, [1])

    if mask is not None:
        mask = tf.cast(mask, dtype='int32')
        # mask output
        y *= mask
        # set masked values to -1
        y += -(1 - mask)
    return y


class ChainCRF(Layer):
    """
    A Linear Chain Conditional Random Field output layer.
    It carries the loss function and its weights for computing
    the global tag sequence scores. While training it acts as
    the identity function that passes the inputs to the subsequently
    used loss function. While testing it applies Viterbi decoding
    and returns the best scoring tag sequence as one-hot encoded vectors.
    # Arguments
        init: weight initialization function for chain energies U.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializers](../initializers.md)).
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the transition weight matrix.
        b_start_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the start bias b.
        b_end_regularizer: instance of [WeightRegularizer](../regularizers.md)
            module, applied to the end bias b.
        b_start_constraint: instance of the [constraints](../constraints.md)
            module, applied to the start bias b.
        b_end_constraint: instance of the [constraints](../constraints.md)
            module, applied to the end bias b.
        weights: list of Numpy arrays for initializing [U, b_start, b_end].
            Thus it should be a list of 3 elements of shape
            [(n_classes, n_classes), (n_classes, ), (n_classes, )]
    # Input shape
        3D tensor with shape `(nb_samples, timesteps, nb_classes)`, where
        ´timesteps >= 2`and `nb_classes >= 2`.
    # Output shape
        Same shape as input.
    # Masking
        This layer supports masking for input sequences of variable length.
    # Example
    ```python
    # As the last layer of sequential layer with
    # model.output_shape == (None, timesteps, nb_classes)
    crf = ChainCRF()
    model.add(crf)
    # now: model.output_shape == (None, timesteps, nb_classes)
    # Compile model with chain crf loss (and one-hot encoded labels) and accuracy
    model.compile(loss=crf.loss, optimizer='sgd', metrics=['accuracy'])
    # Alternatively, compile model with sparsely encoded labels and sparse accuracy:
    model.compile(loss=crf.sparse_loss, optimizer='sgd', metrics=['sparse_categorical_accuracy'])
    ```
    # Gotchas
    ## Model loading
    When you want to load a saved model that has a crf output, then loading
    the model with 'keras.models.load_model' won't work properly because
    the reference of the loss function to the transition parameters is lost. To
    fix this, you need to use the parameter 'custom_objects' as follows:
    ```python
    from keras.layer.crf import create_custom_objects:
    model = keras.models.load_model(filename, custom_objects=create_custom_objects())
    ```
    ## Temporal sample weights
    Given a ChainCRF instance crf both loss functions, crf.loss and crf.sparse_loss
    return a tensor of shape (batch_size, 1) and not (batch_size, maxlen).
    that sample weighting in temporal mode.
    """
    def __init__(self, init='glorot_uniform',
                 U_regularizer=None,
                 b_start_regularizer=None,
                 b_end_regularizer=None,
                 U_constraint=None,
                 b_start_constraint=None,
                 b_end_constraint=None,
                 weights=None,
                 **kwargs):
        super(ChainCRF, self).__init__(**kwargs)
        self.init = initializers.get(init)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_start_regularizer = regularizers.get(b_start_regularizer)
        self.b_end_regularizer = regularizers.get(b_end_regularizer)
        self.U_constraint = constraints.get(U_constraint)
        self.b_start_constraint = constraints.get(b_start_constraint)
        self.b_end_constraint = constraints.get(b_end_constraint)

        self.initial_weights = weights

        self.supports_masking = True
        self.uses_learning_phase = True
        self.input_spec = [InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], input_shape[2])

    def compute_mask(self, input, mask=None):
        if mask is not None:
            return tf.keras.backend.any(mask, axis=1)
        return mask

    def _fetch_mask(self):
        mask = None
        if self._inbound_nodes:
            #mask = self._inbound_nodes[0].input_masks[0]
            mask = self.get_input_mask_at(0)
        return mask

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_steps is None or n_steps >= 2
        self.input_spec = [InputSpec(dtype=tf.keras.backend.floatx(),
                                     shape=(None, n_steps, n_classes))]

        self.U = self.add_weight(shape=(n_classes, n_classes),
                                 initializer=self.init,
                                 name='U',
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)

        self.b_start = self.add_weight(shape=(n_classes, ),
                                       initializer='zero',
                                       name='b_start',
                                       regularizer=self.b_start_regularizer,
                                       constraint=self.b_start_constraint)

        self.b_end = self.add_weight(shape=(n_classes, ),
                                     initializer='zero',
                                     name='b_end',
                                     regularizer=self.b_end_regularizer,
                                     constraint=self.b_end_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        y_pred = viterbi_decode(x, self.U, self.b_start, self.b_end, mask)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = tf.one_hot(y_pred, nb_classes)
        return tf.keras.backend.in_train_phase(x, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        """
        Linear Chain Conditional Random Field loss function.
        """
        mask = self._fetch_mask()
        return chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def sparse_crf_loss_masked(self, y_true, y_pred):
        mask_value = 0
        y_true_masked = tf.ragged.boolean_mask(y_true, tf.not_equal(y_true, mask_value)).to_tensor()
        y_pred_masked = tf.ragged.boolean_mask(y_pred, tf.not_equal(y_true, mask_value)).to_tensor()

        # note: we could experiment some more aggressive stripping of the input for padding and
        # if special wordpiece symbol were used, to better ignore them during loss estimation

        return self.sparse_loss(y_true_masked, y_pred_masked)

    def sparse_loss(self, y_true, y_pred):
        """
        Linear Chain Conditional Random Field loss function with sparse
        tag sequences.
        """
        y_true = tf.cast(y_true, 'int32')

        # note: nothing to squeeze here with sparse tags which are 2D
        #y_true = tf.squeeze(y_true, [2])
        mask = self._fetch_mask()
        return sparse_chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def get_config(self):
        config = {
            'init': initializers.serialize(self.init),
            'U_regularizer': regularizers.serialize(self.U_regularizer),
            'b_start_regularizer': regularizers.serialize(self.b_start_regularizer),
            'b_end_regularizer': regularizers.serialize(self.b_end_regularizer),
            'U_constraint': constraints.serialize(self.U_constraint),
            'b_start_constraint': constraints.serialize(self.b_start_constraint),
            'b_end_constraint': constraints.serialize(self.b_end_constraint)
        }
        base_config = super(ChainCRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    """
    Returns the custom objects, needed for loading a persisted model.
    """
    instanceHolder = {'instance': None}

    class ClassWrapper(ChainCRF):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    def sparse_loss(*args):
        method = getattr(instanceHolder['instance'], 'sparse_loss')
        return method(*args)

    return {'ChainCRF': ClassWrapper, 'loss': loss, 'sparse_loss': sparse_loss}
