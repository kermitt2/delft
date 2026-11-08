"""
The tokenizers of the transformers, loaded once for the process.

The data loaders are built for every call to tag or classify, and each loaded the
tokenizer of the model again with ``AutoTokenizer.from_pretrained``: its files were read
and parsed again and, unless the Hub is set offline, huggingface.co was asked whether
they had changed. That is 0.5 to 0.8 second per call for a BERT-like tokenizer, several
seconds for one with a large vocabulary, and a request to the Hub for every sequence a
server such as GROBID labels.

``get_tokenizer`` loads a tokenizer the first time it is asked for and gives the same
one afterwards. A tokenizer is then shared by the threads of the process, and it is not
safe to call from several of them: a call sets the truncation and the padding it was
asked for on the tokenizer, where a call of another thread may find them instead of its
own, and return a sequence cut at another length, or not cut. Call a shared tokenizer
with ``call_tokenizer``, which lets one call through at a time.

The tokenizer of a loaded model is the one it saved, when it saved it (``use_saved_tokenizer``).
"""

import logging
import os
import threading

from delft.utilities.transformer_models import TRANSFORMER_TOKENIZER_DIR

LOGGER = logging.getLogger(__name__)

_load_lock = threading.Lock()
_tokenizers = {}
# the directory a loaded model saved the tokenizer of a transformer in, by transformer name
_saved_tokenizers = {}

# held for the time of a call: tokenizing a sequence takes a millisecond or so, which is
# nothing next to the forward pass of the transformer that follows it
_call_lock = threading.RLock()


def get_tokenizer(name, **kwargs):
    """
    The tokenizer ``AutoTokenizer.from_pretrained(name, **kwargs)`` gives, loaded at the
    first call for these arguments and kept for the life of the process.

    With ``add_prefix_space=True``, the tokenizer is also made to mark the start of every
    word of a pre-tokenized sequence (see ``mark_word_starts``): the generic tokenizer of
    some byte-level BPE models ignores the option.
    """
    key = (str(name), tuple(sorted((option, repr(value)) for option, value in kwargs.items())))
    with _load_lock:
        tokenizer = _tokenizers.get(key)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(_saved_tokenizers.get(str(name), name), **kwargs)
            if kwargs.get("add_prefix_space"):
                mark_word_starts(tokenizer)
            _tokenizers[key] = tokenizer
    return tokenizer


# two words, the second of which must be tokenized the same whether the text is given
# whole or split into words
_PROBE_WORDS = ["hello", "world"]


def loses_word_starts(tokenizer):
    """
    Whether the tokenizer, given a sequence split into words, encodes each word as if it
    were glued to the previous one: a byte-level BPE (RoBERTa, GPT-2...) or a
    sentencepiece tokenizer marks the start of a word with the space before it ("Ġ", "▁"),
    and finds no space between words given apart unless it adds one itself. The
    transformer then reads the sequence as one word, "helloworld".
    """
    if not getattr(tokenizer, "is_fast", False):
        return False
    whole = tokenizer(" ".join(_PROBE_WORDS), add_special_tokens=False)
    split = tokenizer(_PROBE_WORDS, is_split_into_words=True, add_special_tokens=False)

    def last_word(encoding):
        word_ids = encoding.word_ids()
        last = max(index for index in word_ids if index is not None)
        return [piece for piece, index in zip(encoding["input_ids"], word_ids) if index == last]

    return last_word(whole) != last_word(split)


def _with_word_starts(pre_tokenizer):
    """The JSON description of a pre-tokenizer, set to mark the start of every word."""
    if pre_tokenizer.get("type") == "Sequence":
        return {**pre_tokenizer, "pretokenizers": [_with_word_starts(p) for p in pre_tokenizer["pretokenizers"]]}
    if pre_tokenizer.get("type") == "ByteLevel":
        return {**pre_tokenizer, "add_prefix_space": True}
    if pre_tokenizer.get("type") == "Metaspace":
        return {**pre_tokenizer, "prepend_scheme": "always"}
    return pre_tokenizer


def mark_word_starts(tokenizer):
    """
    Make a fast tokenizer mark the start of every word of a sequence split into words, as
    ``add_prefix_space=True`` does for the tokenizers that know the option: the generic
    ``PreTrainedTokenizerFast`` of some models (e.g. scilons/SciLaD-M-custom, a RoBERTa)
    ignores it, and with transformers 5 no tokenizer refuses split words without it any
    more. Returns whether the tokenizer was changed. A tokenizer that keeps word starts
    (BERT, WordPiece) is left as it is.
    """
    if not loses_word_starts(tokenizer):
        return False
    import json

    from tokenizers import Tokenizer

    backend = tokenizer.backend_tokenizer
    description = json.loads(backend.to_str())
    if description.get("pre_tokenizer") is None:
        return False
    description["pre_tokenizer"] = _with_word_starts(description["pre_tokenizer"])
    backend.pre_tokenizer = Tokenizer.from_str(json.dumps(description)).pre_tokenizer
    if loses_word_starts(tokenizer):
        LOGGER.warning(
            "the tokenizer %s does not mark the start of the words of a split sequence: its "
            "transformer reads the words as glued together",
            getattr(tokenizer, "name_or_path", tokenizer),
        )
    return True


def use_saved_tokenizer(name, model_dir):
    """
    Read the tokenizer of the transformer ``name`` from the model directory ``model_dir``, when the
    model saved it there, rather than from the Hub.
    """
    directory = os.path.join(model_dir, TRANSFORMER_TOKENIZER_DIR)
    if os.path.isdir(directory):
        with _load_lock:
            _saved_tokenizers[str(name)] = directory


def call_tokenizer(tokenizer, *args, **kwargs):
    """``tokenizer(*args, **kwargs)``, one call at a time over all the threads."""
    with _call_lock:
        return tokenizer(*args, **kwargs)


def clear_tokenizers():
    """Forget the tokenizers loaded so far: the next ``get_tokenizer`` loads again."""
    with _load_lock:
        _tokenizers.clear()
        _saved_tokenizers.clear()
