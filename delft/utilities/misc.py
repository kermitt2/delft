from typing import List, Dict, Tuple

__author__ = "@de-code"

"""
Utility class from: 
https://github.com/elifesciences/sciencebeam-trainer-delft/blob/develop/sciencebeam_trainer_delft/utils/misc.py
"""

def parse_number_range(expr: str) -> List[int]:
    fragments = expr.split('-')
    if len(fragments) == 1:
        return [int(expr)]
    if len(fragments) == 2:
        return list(range(int(fragments[0]), int(fragments[1]) + 1))
    raise ValueError('invalid number range: %s' % fragments)


def parse_number_ranges(expr: str) -> List[int]:
    if not expr:
        return []
    numbers = []
    for fragment in expr.split(','):
        numbers.extend(parse_number_range(fragment))
    return numbers


def parse_key_value(expr: str) -> Tuple[str, str]:
    key, value = expr.split('=', maxsplit=1)
    return key.strip(), value.strip()


def parse_dict(expr: str, delimiter: str = '|') -> Dict[str, str]:
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
