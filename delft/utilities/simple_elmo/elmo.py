import tensorflow as tf


def weight_layers(name, bilm_ops, l2_coef=0.0, use_layers="average", do_layer_norm=False):
    """
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES

    Input:
        name = a string prefix used for the trainable variable names
        bilm_ops = the tensorflow ops returned to compute internal
            representations from a biLM.  This is the return value
            from BidirectionalLanguageModel(...)(ids_placeholder)
        l2_coef: the l2 regularization coefficient lambda.
            Pass None or 0.0 for no regularization.
        use_layers: if "top", only use the top layer; if "average", yield the average of all layers;
        if "all", yield all layers representations for each word.
        do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing

    Output:
        {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
        }
    """

    def _l2_regularizer(weights):
        return l2_coef * tf.reduce_sum(tf.square(weights))

    # Get ops for computing LM embeddings and mask
    lm_embeddings = bilm_ops['lm_embeddings']
    mask = bilm_ops['mask']

    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            n = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / n
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                     ) / n
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )
        # no regularization
        reg = 0.0

        if use_layers == "all":
            if do_layer_norm:
                sum_pieces = _do_ln(lm_embeddings)
            else:
                sum_pieces = lm_embeddings
        elif use_layers == "top":
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
            # just the top layer
            sum_pieces = tf.squeeze(layers[-1], axis=1)
        elif use_layers == "average":
            with tf.compat.v1.variable_scope("bilm", reuse=tf.compat.v1.AUTO_REUSE):
                elmo_weights = tf.compat.v1.get_variable(
                    '{}_ELMo_W'.format(name),
                    shape=(n_lm_layers,),
                    initializer=tf.zeros_initializer,
                    regularizer=_l2_regularizer,
                    trainable=True,
                )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(elmo_weights + 1.0 / n_lm_layers), n_lm_layers
            )
            # split LM layers
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

            # compute the weighted, normalized LM activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if do_layer_norm:
                    pieces.append(w * _do_ln(tf.squeeze(t, axis=1)))
                else:
                    pieces.append(w * tf.squeeze(t, axis=1))
            sum_pieces = tf.add_n(pieces)

            # get the regularizer 
            reg = [
                r for r in tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_ELMo_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by gamma
        with tf.compat.v1.variable_scope("bilm", reuse=tf.compat.v1.AUTO_REUSE):
            gamma = tf.compat.v1.get_variable(
                '{}_ELMo_gamma'.format(name),
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
            )

        weighted_lm_layers = sum_pieces * gamma

        ret = {'weighted_op': weighted_lm_layers, 'regularization_op': reg}

    return ret
