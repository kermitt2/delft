"""
Derived from https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py

Produce standard evaluation metrics for sequence labeling tasks (precision, recall, f-score at field level). 

From the seqeval version, we modify: 

* the classification_report() method so that micro average is computed over 
actual total predictions and not weighted by number of expected pred in each 
label - otherwise average micro f1 scores are not consistent with f1_score() 

* an evaluation report is provided both as a string for simple console output
and as a map for further processing (e.g. n-fold average)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np


def get_entities(seq):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        #type_ = chunk.split('-')[-1]
        type_ = "-".join(chunk.split('-')[1:])

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro'):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro'):
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro'):
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(y_true, y_pred, digits=2):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
    Returns:
        report : string. Text summary of the precision, recall, F1 score and support number for each class.
        evaluation: map. Map with three keys:
            - 'labels', containing a map for all class with precision, recall, F1 score and support number,
            - 'micro' with precision, recall, F1 score and support number micro average
            - 'macro' with precision, recall, F1 score and support number macro average
    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> eval, _ = classification_report(y_true, y_pred)
        >>> print(eval)
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    evaluation = compute_metrics(y_true, y_pred)

    return get_report(evaluation, digits=digits), evaluation


def compute_metrics(y_true, y_pred):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
    Returns:
        report : string. Text summary of the precision, recall, F1 score and support number for each class.
        evaluation: map. Map with three keys:
            - 'labels', containing a map for all class with precision, recall, F1 score and support number,
            - 'micro' with precision, recall, F1 score and support number micro average
            - 'macro' with precision, recall, F1 score and support number macro average
    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> eval = compute_metrics(y_true, y_pred)
        >>> print(eval)

    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))

    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    evaluation = {'labels': {}}

    ps, rs, f1s, s = [], [], [], []
    total_nb_correct = 0
    total_nb_pred = 0
    total_nb_true = 0
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        eval_block = {}
        eval_block["precision"] = p
        eval_block["recall"] = r
        eval_block["f1"] = f1
        eval_block["support"] = nb_true

        evaluation['labels'][type_name] = eval_block

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

        total_nb_correct += nb_correct
        total_nb_true += nb_true
        total_nb_pred += nb_pred

    # compute averages

    # marco aveage
    # macro_eval_block = {
    #     "precision": np.average(ps, weights=s),
    #     "recall": np.average(rs, weights=s),
    #     "f1": np.average(f1s, weights=s),
    #     "support": np.sum(s)
    # }
    #
    # evaluation['macro'] = macro_eval_block

    # micro average
    micro_precision = total_nb_correct / total_nb_pred if total_nb_pred > 0 else 0
    micro_recall = total_nb_correct / total_nb_true if total_nb_true > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (
                micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    micro_eval_block = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "support": np.sum(s)
    }
    evaluation["micro"] = micro_eval_block

    return evaluation


def get_report(evaluation, digits=2, include_avgs=['micro']):
    """
    Calculate the report from the evaluation metrics.

    :param evaluation: the map for evaluation containing three keys:
        - 'labels', a map of maps contains the values of precision, recall, f1 and support for each label
        - 'macro' and 'micro', containing the micro and macro average, respectively (precision, recall, f1 and support)

    :param digits: the number of digits to use in the report
    :param include_avgs: the average to include in the report, default: 'micro'
    :return:
    """
    name_width = max([len(e) for e in evaluation.keys()])

    last_line_heading = {
        'micro': 'all (micro avg.)',
        'macro': 'all (macro avg.)'
    }
    width = max(name_width, len(last_line_heading['micro']), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    block = evaluation['labels']
    labels = sorted(block.keys())

    for label in labels:
        p = block[label]['precision']
        r = block[label]['recall']
        f1 = block[label]['f1']
        s = block[label]['support']

        report += row_fmt.format(*[label, p, r, f1, s], width=width, digits=digits)

    report += u'\n'
    for average in include_avgs:
        avg = evaluation[average]
        report += row_fmt.format(last_line_heading[average],
                                 avg['precision'],
                                 avg['recall'],
                                 avg['f1'],
                                 avg['support'] if 'support' in avg else "",
                                 width=width, digits=digits)

    return report