# -*- coding: utf-8 -*-
import argparse
import os
import random
from ctypes import Union

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file


def random_sampling(example_pool, max):
    """
    The sampling is performed randomly for a max number of examples
    """
    if max > len(example_pool):
        print("The requested number is too high, there are not enough examples in the pool. ")
        exit(-1)
    else:
        print("Shuffling...")
        # shuffled = shuffle_triple_with_view(np.asarray(example_pool))[0]
        shuffled = random.sample(example_pool, len(example_pool))

        return shuffled[0:max]


def active_sampling(example_pool, max, delft_model: Sequence):
    sentences = [example[0] for example in example_pool]
    features = [example[2] for example in example_pool]
    labels = delft_model.tag(sentences, None, features=features)

    samples = []
    ignored = []

    for idx in range(0, max):
        labels_str = [label[1] for label in labels[idx]]

        if labels[idx].count("O") < len(labels[idx]):
            samples.append((sentences[idx], labels_str, features[idx]))
        else:
            ignored.append((sentences[idx], labels_str, features[idx]))

    if len(samples) < max:
        print("Injected", len(samples), " but not sufficient. We fill the missing with random samples.")
        for idx in range(0, max - len(samples)):
            samples.append(ignored[idx])

    return samples


RANDOM_SAMPLING = "random_sampling"
ACTIVE_SAMPLING = "active_sampling"


def load_examples(input_path, input_negative_path=None):
    sents, labels, features = load_data_and_labels_crf_file(input_path)

    positives = []
    negatives = []

    if input_negative_path:
        n_sents, n_labels, n_features = load_data_and_labels_crf_file(input_path)

        for idx in range(0, len(n_labels)):
            negatives.append((n_sents[idx], n_labels[idx], n_features[idx]))

        for idx in range(0, len(labels)):
            positives.append((sents[idx], labels[idx], features[idx]))

    else:
        for idx in range(0, len(labels)):
            if labels[idx].count("O") == len(labels[idx]):
                negatives.append((sents[idx], labels[idx], features[idx]))
            else:
                positives.append((sents[idx], labels[idx], features[idx]))

    return positives, negatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for applying over and under sampling from a CRF training file")

    actions = [RANDOM_SAMPLING, ACTIVE_SAMPLING]
    parser.add_argument("--input", help="Input file containing examples. ", required=True)
    parser.add_argument("--input-negatives",
                        help="Optional input file containing only negative examples. With this option it is assumed "
                             "that --input provides only positive examples",
                        required=False, default=None)
    parser.add_argument("--model-config", help="DeLFT configuration file model to be used for active sampling. ",
                        required=False)
    parser.add_argument("--output", help="Directory where to save the sampled output.", required=True)
    parser.add_argument("--output-only-negatives",
                        help="Indicate whether to write only negative examples in the output file.", required=False,
                        default=False)
    parser.add_argument("--action", choices=actions, required=True)
    parser.add_argument("--ratio", help="Sampling ratio", type=float, required=True)

    args = parser.parse_args()

    action = args.action
    input_path = args.input
    input_negative_path = args.input_negatives
    output_dir = args.output
    ratio = args.ratio
    delft_model_config = args.model_config
    output_only_negatives = args.output_only_negatives

    if action == ACTIVE_SAMPLING and not delft_model_config:
        print("For active sampling a DeLFT model is required. "
              "Use the option --model-config to specify a configuration file. ")
        exit(-1)

    positives, negatives = load_examples(input_path, input_negative_path)

    negatives_to_inject = int(ratio * len(positives))
    num_positives = len(positives)
    num_negatives = len(negatives)
    print("Positives:", num_positives)
    print("Negatives:", num_negatives)
    print("Negatives needed:", negatives_to_inject)

    sampling = []
    sampling_type = ""
    if action == RANDOM_SAMPLING:
        sampling_type = "random"
        sampling = random_sampling(negatives, negatives_to_inject)
    elif action == ACTIVE_SAMPLING:
        sampling_type = "active"
        model_config = ModelConfig.load(delft_model_config)
        model = Sequence(model_config.model_name)
        model.load()
        sampling = active_sampling(negatives, negatives_to_inject, model)

    data = sampling + positives
    if output_only_negatives:
        data = sampling

    output_file_name = os.path.join(output_dir,
                                    "sampling-" + sampling_type + "-" + str(ratio) + "r-" + str(
                                        num_positives) + "p-" + str(
                                        negatives_to_inject) + "n" + ".train")

    with open(output_file_name, 'w') as fo:
        for idx in range(0, len(data)):
            s, l, f = data[idx]
            for t_idx in range(0, len(s)):
                # Change prefixes: I- to nothing and B- to I
                label = l[t_idx]
                if l[t_idx].startswith("I-"):
                    label = label.replace("I-", "")
                elif l[t_idx].startswith("B-"):
                    label = label.replace("B-", "I-")
                elif l[t_idx] == "O":
                    label = "<other>"

                fo.write(s[t_idx] + "\t" + "\t".join(f[t_idx]) + '\t' + label + '\n')

            fo.write("\n\n")
