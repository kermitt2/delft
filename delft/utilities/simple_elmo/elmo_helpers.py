# /bin/env python3
# coding: utf-8

import json
import logging
import os
import sys
import zipfile
import numpy as np
import tensorflow as tf
from .data import Batcher
from .elmo import weight_layers
from .model import BidirectionalLanguageModel
from .training import (
    load_options_latest_checkpoint,
    load_vocab,
    LanguageModel,
    pack_encoded,
    _get_feed_dict_from_x,
)


class ElmoModel:
    """
    Embeddings from Language Models (ELMo)
    """

    def __init__(self):
        self.batcher = None
        self.sentence_character_ids = None
        self.elmo_sentence_input = None
        self.sentence_embeddings_op = None
        self.batch_size = None
        self.max_chars = None
        self.vector_size = None
        self.n_layers = None
        self.vocab = None
        self.session = None
        self.model = None
        self.init_state_tensors = None
        self.final_state_tensors = None
        self.init_state_values = None

        # We do not use eager execution from TF 2.0
        tf.compat.v1.disable_eager_execution()

        # Do not emit deprecation warnings:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def load(self, directory=None, vocab_file=None, options_file=None, weight_file=None, max_batch_size=32, limit=100, full=False):
        # Loading a pre-trained ELMo model:
        # You can call load with top=True to use only the top ELMo layer
        """
        :param directory: directory or a ZIP archive with an ELMo model
        ('*.hdf5' and 'options.json' files must be present)
        :param max_batch_size: the maximum allowable batch size during inference
        :param limit: cache only the first <limit> words from the vocabulary file
        :param full: set to True if loading from full checkpoints (for example, for LM)
        :return: nothing
        """
        self.batch_size = max_batch_size
        self.logger.info(f"Loading model from {directory}...")

        if directory is not None and os.path.isfile(directory) and directory.endswith(".zip"):
            message = """
            Assuming the model is a ZIP archive downloaded from the NLPL vector repository.
            Loading a model from a ZIP archive directly is slower than from the extracted files,
            but does not require additional disk space
            and allows to load from directories without write permissions.
            """
            self.logger.info(message)
            if sys.version_info.major < 3 or sys.version_info.minor < 7:
                raise SystemExit(
                    "Error: loading models from ZIP archives requires Python >= 3.7."
                )
            zf = zipfile.ZipFile(directory)
            vocab_file = zf.open("vocab.txt")
            options_file = zf.open("options.json")
            weight_file = zf.open("model.hdf5")
            m_options = json.load(options_file)
            options_file.seek(0)
        elif directory is not None and os.path.isdir(directory):
            # We have all the files already extracted in a separate directory
            options_file = os.path.join(directory, "options.json")
            with open(options_file, "r") as of:
                m_options = json.load(of)

            if os.path.isfile(os.path.join(directory, "vocab.txt.gz")):
                vocab_file = os.path.join(directory, "vocab.txt.gz")
            elif os.path.isfile(os.path.join(directory, "vocab.txt")):
                vocab_file = os.path.join(directory, "vocab.txt")
            else:
                self.logger.info("No vocabulary file found for the ELMo model.")
                vocab_file = None
            if full:
                _, weight_file = load_options_latest_checkpoint(directory)
                self.logger.info(f"Loading from {weight_file}...")

            else:
                if os.path.exists(os.path.join(directory, "model.hdf5")):
                    weight_file = os.path.join(directory, "model.hdf5")
                else:
                    weight_files = [
                        fl for fl in os.listdir(directory) if fl.endswith(".hdf5")
                    ]
                    if not weight_files:
                        raise SystemExit(
                            f"Error: no HDF5 model files found in the {directory} directory!"
                        )
                    weight_file = os.path.join(directory, weight_files[0])
                    self.logger.info(
                        f"No model.hdf5 file found. Using {weight_file} as a model file."
                    )
        elif options_file is not None and os.path.isfile(options_file):
            with open(options_file, "r") as of:
                m_options = json.load(of)

            if vocab_file is None or not os.path.isfile(vocab_file):
                self.logger.info("No vocabulary file found for the ELMo model.")
                vocab_file = None

            if weight_file is None or not os.path.isfile(weight_file):
                self.logger.info("ELMo weight file not found.")
                weight_file = None
        else:
            raise SystemExit(
                "Error: either provide a path to a directory with the ELMo model "
                "individual options and weight file or to the model in a ZIP archive."
            )

        max_chars = m_options["char_cnn"]["max_characters_per_token"]
        self.max_chars = max_chars
        if full:
            if m_options["char_cnn"]["n_characters"] == 262:
                self.logger.info(
                    "Invalid number of characters in the options.json file: 262."
                )
                self.logger.info("Setting it to 261 for using the model as LM")
                m_options["char_cnn"]["n_characters"] = 261
            self.vocab = load_vocab(vocab_file, self.max_chars)
            unroll_steps = 1
            lm_batch_size = self.batch_size

            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            self.session = tf.compat.v1.Session(config=config)

            with tf.compat.v1.variable_scope("lm"):
                test_options = dict(m_options)
                test_options["batch_size"] = lm_batch_size
                test_options["unroll_steps"] = 1
                self.model = LanguageModel(test_options, False)
                # we use the "Saver" class to load the variables
                loader = tf.compat.v1.train.Saver()
                loader.restore(self.session, weight_file)

            self.init_state_tensors = self.model.init_lstm_state
            self.final_state_tensors = self.model.final_lstm_state

            feed_dict = {
                self.model.tokens_characters: np.zeros(
                    [lm_batch_size, unroll_steps, max_chars], dtype=np.int32
                )
            }
            # Bidirectionality:
            feed_dict.update(
                {
                    self.model.tokens_characters_reverse: np.zeros(
                        [lm_batch_size, unroll_steps, max_chars], dtype=np.int32
                    )
                }
            )

            self.init_state_values = self.session.run(
                self.init_state_tensors, feed_dict=feed_dict
            )

        else:
            if m_options["char_cnn"]["n_characters"] == 261:
                raise SystemExit(
                    "Error: invalid number of characters in the options.json file: 261. "
                    "Set n_characters to 262 for inference."
                )

            # Create a Batcher to map text to character ids.
            self.batcher = Batcher(vocab_file, max_chars, limit=limit)

            # Input placeholders to the biLM.
            self.sentence_character_ids = tf.compat.v1.placeholder(
                "int32", shape=(None, None, max_chars)
            )

            # Build the biLM graph.
            bilm = BidirectionalLanguageModel(
                options_file, weight_file, max_batch_size=max_batch_size
            )

            # Get ops to compute the LM embeddings.
            self.sentence_embeddings_op = bilm(self.sentence_character_ids)

        self.vector_size = int(m_options["lstm"]["projection_dim"] * 2)
        self.n_layers = m_options["lstm"]["n_layers"] + 1

    def get_elmo_vectors(self, texts, warmup=True, layers="average", session=None):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :param session: external TensorFlow session to use
        :return: embedding tensor for all sentences
        (number of used layers by max word count by vector size)
        """
        max_text_length = max([len(t) for t in texts])

        # Creating the matrix which will eventually contain all embeddings from all batches:
        if layers == "all":
            final_vectors = np.zeros(
                (len(texts), self.n_layers, max_text_length, self.vector_size)
            )
        else:
            final_vectors = np.zeros((len(texts), max_text_length, self.vector_size))
        existing_session = True
        if not session:
            existing_session = False
            session = tf.compat.v1.Session()
        with session.as_default() as sess:
            if not existing_session:
                # Get an op to compute ELMo vectors (a function of the internal biLM layers)
                self.elmo_sentence_input = weight_layers(
                    "input", self.sentence_embeddings_op, use_layers=layers
                )

                # It is necessary to initialize variables once before running inference.
                sess.run(tf.compat.v1.global_variables_initializer())

            '''
            if warmup:
                self.warmup(sess, texts)
            '''

            # Running batches:
            chunk_counter = 0
            for chunk in divide_chunks(texts, self.batch_size):
                # Converting sentences to character ids:
                sentence_ids = self.batcher.batch_sentences(chunk)
                self.logger.info(f"Texts in the current batch: {len(chunk)}")

                # Compute ELMo representations.
                if warmup:
                    _ = sess.run(
                        self.elmo_sentence_input["weighted_op"],
                        feed_dict={self.sentence_character_ids: sentence_ids},
                    )
                elmo_vectors = sess.run(
                    self.elmo_sentence_input["weighted_op"],
                    feed_dict={self.sentence_character_ids: sentence_ids},
                )
                # Updating the full matrix:
                first_row = self.batch_size * chunk_counter
                last_row = first_row + elmo_vectors.shape[0]
                if layers == "all":
                    final_vectors[
                    first_row:last_row, :, : elmo_vectors.shape[2], :
                    ] = elmo_vectors
                else:
                    final_vectors[
                    first_row:last_row, : elmo_vectors.shape[1], :
                    ] = elmo_vectors
                chunk_counter += 1

            return final_vectors

    def get_elmo_vector_average(self, texts, warmup=True, layers="average", session=None):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :param session: external TensorFlow session to use
        :return: matrix of averaged embeddings for all sentences
        """

        if layers == "all":
            average_vectors = np.zeros((len(texts), self.n_layers, self.vector_size))
        else:
            average_vectors = np.zeros((len(texts), self.vector_size))

        counter = 0

        existing_session = True
        if not session:
            existing_session = False
            session = tf.compat.v1.Session()

        with session.as_default() as sess:
            if not existing_session:
                # Get an op to compute ELMo vectors (a function of the internal biLM layers)
                self.elmo_sentence_input = weight_layers(
                    "input", self.sentence_embeddings_op, use_layers=layers
                )

                # It is necessary to initialize variables once before running inference.
                sess.run(tf.compat.v1.global_variables_initializer())

            if warmup:
                self.warmup(sess, texts)

            # Running batches:
            for chunk in divide_chunks(texts, self.batch_size):
                # Converting sentences to character ids:
                sentence_ids = self.batcher.batch_sentences(chunk)
                self.logger.info(f"Texts in the current batch: {len(chunk)}")

                # Compute ELMo representations.
                elmo_vectors = sess.run(
                    self.elmo_sentence_input["weighted_op"],
                    feed_dict={self.sentence_character_ids: sentence_ids},
                )

                self.logger.debug(f"ELMo sentence input shape: {elmo_vectors.shape}")

                if layers == "all":
                    elmo_vectors = elmo_vectors.reshape(
                        (
                            len(chunk),
                            elmo_vectors.shape[2],
                            self.n_layers,
                            self.vector_size,
                        )
                    )
                for sentence in range(len(chunk)):
                    if layers == "all":
                        sent_vec = np.zeros(
                            (elmo_vectors.shape[1], self.n_layers, self.vector_size)
                        )
                    else:
                        sent_vec = np.zeros((elmo_vectors.shape[1], self.vector_size))
                    for nr, word_vec in enumerate(elmo_vectors[sentence]):
                        sent_vec[nr] = word_vec
                    semantic_fingerprint = np.sum(sent_vec, axis=0)
                    semantic_fingerprint = np.divide(
                        semantic_fingerprint, sent_vec.shape[0]
                    )
                    query_vec = semantic_fingerprint / np.linalg.norm(
                        semantic_fingerprint
                    )

                    average_vectors[counter] = query_vec
                    counter += 1

        return average_vectors

    def get_elmo_substitutes(self, data, topn=3, nodelimiters=True):
        """
        :param data: list of sentences
        :param topn: how many top probable substitutes to return
        :param nodelimiters: whether to filter out sentence delimiters (<s> and </S>)
        :return: a list of forward and backward LM predictions for each word in each input sentence
        """

        batch_size = self.batch_size
        if len(data) < batch_size:
            raise SystemError(
                "Batch size must be less than the number of input sentences!"
            )
        word_predictions = []

        # For loops and the multiplication operator are not the same in list comprehension:
        storage = [([], []) for _ in range(batch_size)]

        self.logger.info("Calculating language model predictions...")

        for batch_no, batch in enumerate(pack_encoded(data, self.vocab, batch_size)):
            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)

            feed_dict = {
                t: v for t, v in zip(self.init_state_tensors, self.init_state_values)
            }

            feed_dict.update(
                _get_feed_dict_from_x(
                    batch, 0, batch["token_ids"].shape[0], self.model, True, True
                )
            )
            ret = self.session.run(
                [self.final_state_tensors, self.model.output_scores, ],
                feed_dict=feed_dict,
            )

            init_state_values, predictions = ret
            forward_preds, backward_preds = predictions

            if nodelimiters:
                for val in [0, 1]:
                    for f_el, b_el in zip(forward_preds, backward_preds):
                        f_el[val] = 0
                        b_el[val] = 0

            forward_ind = np.argpartition(forward_preds, -topn)[:, -topn:]
            backward_ind = np.argpartition(backward_preds, -topn)[:, -topn:]

            # End of sentence:
            def merge_substitutes(sentence):
                forward_substitutes, backward_substitutes = sentence
                sentence_substitutes = []
                backward_substitutes.reverse()
                for f_position, b_position in zip(
                        forward_substitutes, backward_substitutes
                ):
                    cur_substitute = {
                        "word": f_position["word"],
                        "forward": {
                            el: f_position[el] for el in f_position if el != "word"
                        },
                        "backward": {
                            el: b_position[el] for el in b_position if el != "word"
                        },
                    }
                    sentence_substitutes.append(cur_substitute)
                self.logger.debug(" ".join([d["word"] for d in sentence_substitutes]))
                return sentence_substitutes

            # Next forward tokens:
            next_words = [self.vocab.id_to_word(t[0]) for t in batch["next_token_id"]]
            # Top forward predictions:
            forward_words = [
                [self.vocab.id_to_word(word) for word in row] for row in forward_ind
            ]
            # Top forward prediction probabilities:
            forward_probs = np.take_along_axis(forward_preds, forward_ind, 1)
            forward_probs = np.round(forward_probs.astype(float), 4)

            # Next backward tokens:
            next_back_words = [
                self.vocab.id_to_word(t[0]) for t in batch["next_token_id_reverse"]
            ]
            # Top backward predictions:
            backward_words = [
                [self.vocab.id_to_word(word) for word in row] for row in backward_ind
            ]
            # Top backward prediction probabilities:
            backward_probs = np.take_along_axis(backward_preds, backward_ind, 1)
            backward_probs = np.round(backward_probs.astype(float), 4)

            for nr in range(batch_size):
                if next_words[nr] == "</S>" and next_back_words[nr] == "<S>":
                    self.logger.debug(f"End of sentence found for {nr}!")
                    full_sentence = merge_substitutes(storage[nr])
                    word_predictions.append(full_sentence)
                    storage[nr] = ([], [])
                    continue
                # forward substitutes for the current sentence:
                storage[nr][0].append(
                    {
                        "word": next_words[nr],
                        "candidates": forward_ind[nr].tolist(),
                        "candidate_words": forward_words[nr],
                        "logp": forward_probs[nr].tolist(),
                    }
                )
                # backward_substitutes for the current sentence:
                storage[nr][1].append(
                    {
                        "word": next_back_words[nr],
                        "candidates": backward_ind[nr].tolist(),
                        "candidate_words": backward_words[nr],
                        "logp": backward_probs[nr].tolist(),
                    }
                )
        return word_predictions

    def warmup(self, sess, texts):
        for chunk0 in divide_chunks(texts, self.batch_size):
            self.logger.info(f"Warming up ELMo on {len(chunk0)} sentences...")
            sentence_ids = self.batcher.batch_sentences(chunk0)
            _ = sess.run(
                self.elmo_sentence_input["weighted_op"],
                feed_dict={self.sentence_character_ids: sentence_ids},
            )
            #break
        self.logger.info("Warming up finished.")


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i: i + n]
