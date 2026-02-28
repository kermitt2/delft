"""
Utility functions for multiprocessing configuration in training.
"""

def get_multiprocessing_config(training_config, model_config, embeddings=None):
    """
    Determine the number of workers and multiprocessing mode for training.

    Args:
        training_config: TrainingConfig object with num_workers and multiprocessing settings
        model_config: ModelConfig object with transformer_name
        embeddings: Optional embeddings object with use_ELMo attribute

    Returns:
        tuple: (nb_workers, use_multiprocessing)
    """
    multiprocessing = True

    if hasattr(training_config, 'num_workers') and training_config.num_workers is not None:
        nb_workers = training_config.num_workers
        if nb_workers == 0:
            nb_workers = 1
            multiprocessing = False
        else:
            multiprocessing = training_config.multiprocessing and nb_workers > 1
    else:
        nb_workers = 6
        multiprocessing = training_config.multiprocessing

    if model_config.transformer_name is not None or (embeddings and embeddings.use_ELMo):
        nb_workers = 1
        multiprocessing = False

    return nb_workers, multiprocessing