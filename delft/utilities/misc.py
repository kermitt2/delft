from typing import List, Dict, Tuple

__author__ = "@de-code"

"""
Utility class from: 
https://github.com/elifesciences/sciencebeam-trainer-delft/blob/develop/sciencebeam_trainer_delft/utils/misc.py
"""


def parse_number_range(expr: str) -> List[int]:
    fragments = expr.split("-")
    if len(fragments) == 1:
        return [int(expr)]
    if len(fragments) == 2:
        return list(range(int(fragments[0]), int(fragments[1]) + 1))
    raise ValueError("invalid number range: %s" % fragments)


def parse_number_ranges(expr: str) -> List[int]:
    if not expr:
        return []
    numbers = []
    for fragment in expr.split(","):
        numbers.extend(parse_number_range(fragment))
    return numbers


def parse_key_value(expr: str) -> Tuple[str, str]:
    key, value = expr.split("=", maxsplit=1)
    return key.strip(), value.strip()


def parse_dict(expr: str, delimiter: str = "|") -> Dict[str, str]:
    if not expr:
        return {}
    d = {}
    for fragment in expr.split(delimiter):
        key, value = parse_key_value(fragment)
        d[key] = value
    return d


def merge_dicts(dict_list: List[dict]) -> dict:
    result = {}
    for d in dict_list:
        result.update(d)
    return result


def print_parameters(model_config, training_config):
    print("---")
    print("max_epoch:", training_config.max_epoch)
    print("early_stop:", training_config.early_stop)
    if training_config.early_stop:
        print("patience:", training_config.patience)
    print("batch_size (training):", model_config.batch_size)

    if hasattr(model_config, "max_sequence_length"):
        print("max_sequence_length:", model_config.max_sequence_length)

    if hasattr(model_config, "maxlen"):
        print("maxlen:", model_config.maxlen)

    print("model_name:", model_config.model_name)
    print("learning_rate: ", training_config.learning_rate)

    if (
        hasattr(training_config, "class_weights")
        and training_config.class_weights != None
        and hasattr(model_config, "list_classes")
    ):
        list_classes = model_config.list_classes
        weight_summary = ""
        for indx, class_name in enumerate(model_config.list_classes):
            if indx != 0:
                weight_summary += ", "
            weight_summary += (
                class_name + ": " + str(training_config.class_weights[indx])
            )
        print("class_weights:", weight_summary)

    print("---")


def to_wandb_table(report_as_map):
    """Convert evaluation report to wandb Table format.
    
    Args:
        report_as_map: Dictionary containing evaluation metrics from classification_report
        
    Returns:
        Tuple of (columns, data) for creating wandb.Table
    """
    columns = ["", "precision", "recall", "f1-score", "support"]
    data = []
    
    # Add each label's metrics
    if 'labels' in report_as_map:
        for label, metrics in report_as_map['labels'].items():
            row = [
                label,
                round(metrics['precision'], 4),
                round(metrics['recall'], 4),
                round(metrics['f1'], 4),
                int(metrics['support'])
            ]
            data.append(row)
    
    # Add micro average
    if 'micro' in report_as_map:
        micro = report_as_map['micro']
        micro_row = [
            "all (micro avg.)",
            round(micro['precision'], 4),
            round(micro['recall'], 4),
            round(micro['f1'], 4),
            int(micro['support'])
        ]
        data.append(micro_row)

    # Add macro average
    if 'macro' in report_as_map:
        macro = report_as_map['macro']
        macro_row = [
            "all (macro avg.)",
            round(macro['precision'], 4),
            round(macro['recall'], 4),
            round(macro['f1'], 4),
            int(macro['support'])
        ]
        data.append(macro_row)

    return columns, data
