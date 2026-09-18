"""
The features channel of a text classifier (issue #152).

A text can come with features: a row of values, one per column, as the tokens of a
sequence do in sequence labelling, the columns being here those of the text as a
whole (the section it is in, the type of the document, ...). A categorical column is
embedded, one vector per value, and the vectors of the columns are concatenated to the
pooled representation of the text before the classifier. A continuous column, given
in ``continuous_features_indices``, is a number scaled to [0, 1] with the range seen
when training, and concatenated as it is.

The values of the columns are mapped to indices with the ``FeaturesPreprocessor`` of
sequence labelling, every text being a sequence of one row. What it learns when
fitting is kept in the model config, so that a saved model needs no other file.
"""

import numpy as np

from delft.utilities.preprocess import FeaturesPreprocessor


def fit_features_preprocessor(features, model_config):
    """
    Fit a ``FeaturesPreprocessor`` on ``features``, one row of values per text, with
    the columns and the sizes the model config asks for, and record in the model config
    what it learned, along with ``use_features``.
    """
    preprocessor = FeaturesPreprocessor(
        features_indices=model_config.features_indices,
        features_vocabulary_size=model_config.features_vocabulary_size,
        continuous_features_indices=model_config.continuous_features_indices,
    )
    preprocessor.fit([[row] for row in features])
    model_config.use_features = True
    model_config.features_indices = preprocessor.features_indices
    model_config.features_map_to_index = preprocessor.features_map_to_index
    model_config.continuous_features_indices = preprocessor.continuous_features_indices
    model_config.continuous_features_ranges = preprocessor.continuous_features_ranges
    return preprocessor


def features_preprocessor_from_config(model_config):
    """
    The ``FeaturesPreprocessor`` a model config describes, or ``None`` when the model
    takes no features. The keys of the map of the values, which JSON turned into
    strings, are made integers again, in the config as well.
    """
    if not getattr(model_config, "use_features", False):
        return None
    mapping = {int(column): values for column, values in (model_config.features_map_to_index or {}).items()}
    model_config.features_map_to_index = mapping
    preprocessor = FeaturesPreprocessor(
        features_indices=model_config.features_indices,
        features_vocabulary_size=model_config.features_vocabulary_size,
        features_map_to_index=mapping,
        continuous_features_indices=model_config.continuous_features_indices,
    )
    preprocessor.continuous_features_ranges = model_config.continuous_features_ranges or []
    return preprocessor


def encode_features(preprocessor, features):
    """
    The indices of the categorical values of every text, an int64 array of shape
    [texts, columns], and the scaled numbers of its continuous columns, a float32
    array of shape [texts, continuous columns] or ``None`` without such columns.
    """
    documents = [[row] for row in features]
    indices = np.asarray(preprocessor.transform(documents), dtype=np.int64)[:, 0, :]
    continuous = None
    if preprocessor.continuous_features_indices:
        continuous = np.asarray(preprocessor.transform_continuous(documents), dtype=np.float32)[:, 0, :]
    return indices, continuous


def features_size(model_config):
    """The width of the features channel, what it adds to the input of the classifier."""
    if not getattr(model_config, "use_features", False):
        return 0
    categorical = len(model_config.features_indices or ())
    continuous = len(getattr(model_config, "continuous_features_indices", None) or ())
    return categorical * model_config.features_embedding_size + continuous


def features_vocabulary(model_config):
    """The number of indices of the categorical values, 0 being the padding."""
    columns = len(model_config.features_indices or ()) or 1
    return model_config.features_vocabulary_size * columns + 1


def check_features(model_config, features, what="classify"):
    """
    Raise a ``ValueError`` when ``features`` are given to a model that takes none, or
    not given to a model that takes some.
    """
    if getattr(model_config, "use_features", False):
        if features is None:
            raise ValueError(f"The model {model_config.model_name} takes features, which were not given to {what}")
    elif features is not None:
        raise ValueError(
            f"Features were given to {what}, but the model {model_config.model_name} takes none: train it with features"
        )
