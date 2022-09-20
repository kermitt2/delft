import tensorflow as tf

from tensorflow_addons.text import crf_log_norm
from tensorflow_addons.utils import types

from tensorflow_addons.text.crf_wrapper import CRFModelWrapper

'''
A slightly modified TensorFlow addons CRF wrapper trying to return by default usable scores 
with decode_sequence 

-> it turns out not working as expected when exploring https://stats.stackexchange.com/a/517325
we don't get probabilities and crf_log_norm normalization is far too low 
-> issue on tensorflow addons github.com/tensorflow/addons/issues/2088 
-> we will probably need to wait for PR https://github.com/tensorflow/addons/pull/1935 to be merged

'''

@tf.keras.utils.register_keras_serializable(package="Addons")
class CRFModelWrapperDefault(CRFModelWrapper):

    def call(self, inputs, training=None, mask=None, return_crf_internal=False):        
        base_model_outputs = self.base_model(inputs, training, mask)

        # change next line, if your model has more outputs
        crf_input = base_model_outputs

        decode_sequence, potentials, sequence_length, kernel = self.crf_layer(crf_input)

        # change next line, if your base model has more outputs
        # Aways keep `(potentials, sequence_length, kernel), decode_sequence, `
        # as first two outputs of model.
        # current `self.train_step()` expected such settings
        if not tf.executing_eagerly():
            decode_sequence = tf.cast(decode_sequence, tf.float32)
        outputs = (potentials, sequence_length, kernel), decode_sequence

        '''
        tf.print(potentials)
        tf.print(sequence_length)
        tf.print(kernel)

        # See https://stats.stackexchange.com/a/517325

        # compute the normalization log Z(x) of the CRF from input, lenghts and 
        # transition matrix (chain kernel)
        normalization = crf_log_norm(potentials, sequence_length, kernel)
        #sequence_length_float32 = tf.cast(sequence_length, tf.float32)
        #normalization = tf.divide(normalization, sequence_length_float32)
        #tf.print(normalization)
        
        normalization = tf.expand_dims(normalization, axis=1)
        normalization = tf.expand_dims(normalization, axis=1)

        # then compute the difference
        probabilities = tf.math.subtract(potentials, normalization)

        # and finally exponentiate to get probabilities
        probabilities = tf.math.exp(potentials)
        #tf.print(probabilities)
        '''

        if return_crf_internal:
            return outputs
        else:
            # outputs[0] is the crf internal, skip it
            output_without_crf_internal = outputs[1:]

            # it is nicer to return a tensor instead of an one tensor list
            if len(output_without_crf_internal) == 1:
                return output_without_crf_internal[0]
            else:
                return output_without_crf_internal

class InnerLossPusher(tf.keras.losses.Loss):
    '''
    Experimental... 
    When earger mode is disabled, Keras model.compile() requires a loss function.
    The following custom loss function is simply retrieving the inner model loss calculation
    returning it as explicit loss function for Keras fit() 
    '''
    def __init__(self, model, name="custom_inner_loss_pusher"):
        super().__init__(name=name)
        self.model = model

    def call(self, y_true, y_pred):
        print(self.model.losses)
        return self.model.losses
