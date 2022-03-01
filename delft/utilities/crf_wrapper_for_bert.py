import tensorflow as tf

from tensorflow_addons.text import crf_log_likelihood
from tensorflow_addons.utils import types

from delft.utilities.crf_wrapper_default import CRFModelWrapperDefault

import numpy as np

'''
Alternative CRF model wrapper for models having a BERT/transformer layer. 
Loss is modified to ignore labels corresponding to tokens being special transformer symbols (e.g. SEP, 
CL, PAD, ...), but also those introduced sub-tokens.
The goal is to have similar effect as in pytorch when using the -100 to ignore some labels when calculating 
the loss, but less hacky.
'''

@tf.keras.utils.register_keras_serializable(package="Addons")
class CRFModelWrapperForBERT(CRFModelWrapperDefault):

    def train_step(self, data):
        x, y, sample_weight = self.unpack_training_data(data)

        with tf.GradientTape() as tape:
            (potentials, sequence_length, kernel), decoded_sequence, *_ = self(
                x, training=True, return_crf_internal=True
            )

            mask_value = 0
            # create a boolean mask with False for labels to be ignored
            special_mask = tf.not_equal(y, mask_value)
            special_mask = tf.cast(special_mask, tf.float32)
            #tf.print(special_mask)
            #tf.print(potentials)
            # apply the mask to prediction vectors, it will put to 0
            # weights for label to ignore, normally neutralizing them for 
            # loss calculation
            potentials = tf.multiply(potentials, tf.expand_dims(special_mask, -1))
            #tf.print(potentials)

            # experiment: replace 0 by -100 because it's log-based potential
            #the_minus = tf.fill(tf.shape(potentials), -100.0)
            #potentials = tf.where(tf.equal(potentials, 0.0), the_minus, potentials)

            crf_loss = self.compute_crf_loss(
                potentials, sequence_length, kernel, y, sample_weight
            )
            loss = crf_loss + tf.reduce_sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, decoded_sequence)
        # Return a dict mapping metric names to current value
        orig_results = {m.name: m.result() for m in self.metrics}
        crf_results = {"loss": loss, "crf_loss": crf_loss}
        #crf_results = {"crf_loss": crf_loss}
        return {**orig_results, **crf_results}

    def test_step(self, data):
        x, y, sample_weight = self.unpack_training_data(data)
        (potentials, sequence_length, kernel), decode_sequence, *_ = self(
            x, training=False, return_crf_internal=True
        )

        mask_value = 0
        special_mask = tf.not_equal(y, mask_value)
        special_mask = tf.cast(special_mask, tf.float32)
        potentials = tf.multiply(potentials, tf.expand_dims(special_mask, -1))

        crf_loss = self.compute_crf_loss(
            potentials, sequence_length, kernel, y, sample_weight
        )
        loss = crf_loss + tf.reduce_sum(self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, decode_sequence)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        #results.update({"loss": loss, "crf_loss": crf_loss})  # append loss
        results.update({"crf_loss": crf_loss})  # append loss
        return results

