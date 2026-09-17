"""
Stacked embeddings, i.e. several embeddings used together, the vector of a word
being the concatenation of the vectors given by each of them.

The typical usage is a static embedding next to a contextual one, the way ELMo
was used together with glove in the past:

    --embedding glove-840B+scibert-contextual

The static vector brings what is known about the word whatever the sentence,
which is still informative for rare words and cheap to get, and the contextual
one brings the reading of the word in this sentence.

Each component keeps the behaviour it has when it is used alone: number
normalization for the static embeddings, none for the contextual ones, its own
LMDB database or cache, its own way to be serialized for the DataLoader
workers. This class only holds them together.
"""

import numpy as np

# separator of the embedding names, when the stack is given as a name
STACKED_SEPARATOR = "+"


def split_stacked_name(name):
    """Names of the components of a stack given as a name, e.g. ``glove-840B+scibert-contextual``."""
    return [part.strip() for part in name.split(STACKED_SEPARATOR)]


class StackedEmbeddings:
    """
    ``components`` are the embeddings to concatenate, in the order of the
    concatenation, as ``delft.utilities.Embeddings.Embeddings`` instances.
    """

    def __init__(self, components):
        components = list(components)
        if len(components) < 2:
            raise ValueError("at least two embeddings are required to stack them")
        self.components = components
        self.embed_size = sum(component.embed_size for component in components)

    def __repr__(self):
        return "StackedEmbeddings(%s, dimensions=%d)" % (
            STACKED_SEPARATOR.join(str(component.name) for component in self.components),
            self.embed_size,
        )

    def get_word_vector(self, word):
        """
        Vector of a word out of any context, for the code that works word by
        word. The sequence labelling data loaders go through the components
        instead, see to_vector_single, to embed a sentence as a whole.
        """
        return np.concatenate(
            [np.asarray(component.get_word_vector(word), dtype=np.float32) for component in self.components]
        )

    def precompute(self, token_lists, **kwargs):
        """
        Let the components that compute their vectors per sentence (contextual
        embeddings) do it beforehand, see ContextualEmbeddings.precompute.
        Returns the number of sentences that were embedded.
        """
        embedded = 0
        for component in self.components:
            if hasattr(component, "precompute"):
                embedded += component.precompute(token_lists, **kwargs)
        return embedded

    def reopen_lmdb(self):
        """Called by the DataLoader workers, see Embeddings.reopen_lmdb."""
        for component in self.components:
            component.reopen_lmdb()
