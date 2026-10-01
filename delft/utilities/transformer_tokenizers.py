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
"""

import threading

_load_lock = threading.Lock()
_tokenizers = {}

# held for the time of a call: tokenizing a sequence takes a millisecond or so, which is
# nothing next to the forward pass of the transformer that follows it
_call_lock = threading.RLock()


def get_tokenizer(name, **kwargs):
    """
    The tokenizer ``AutoTokenizer.from_pretrained(name, **kwargs)`` gives, loaded at the
    first call for these arguments and kept for the life of the process.
    """
    key = (str(name), tuple(sorted((option, repr(value)) for option, value in kwargs.items())))
    with _load_lock:
        tokenizer = _tokenizers.get(key)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
            _tokenizers[key] = tokenizer
    return tokenizer


def call_tokenizer(tokenizer, *args, **kwargs):
    """``tokenizer(*args, **kwargs)``, one call at a time over all the threads."""
    with _call_lock:
        return tokenizer(*args, **kwargs)


def clear_tokenizers():
    """Forget the tokenizers loaded so far: the next ``get_tokenizer`` loads again."""
    with _load_lock:
        _tokenizers.clear()
