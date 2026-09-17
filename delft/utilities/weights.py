"""
Weight files of the models, in either of two formats told apart by the file extension:

- safetensors (``model.safetensors``), the format written by default. It holds tensors
  and nothing else;
- a pickled torch state dict (``model_weights.pt``, ``model_weights.pth`` for text
  classification), the format DeLFT wrote up to 1.1.0. Unpickling a file runs whatever
  code it contains, which is why the Hugging Face Hub flags such weights as unsafe.
"""

import os

import torch

SAFETENSORS_WEIGHT_FILE_NAME = "model.safetensors"
SAFETENSORS_EXTENSION = ".safetensors"

# the names the wrappers give to the weights of a model
WEIGHT_FILE_NAMES = (SAFETENSORS_WEIGHT_FILE_NAME, "model_weights.pt", "model_weights.pth")


def is_safetensors(path):
    return str(path).endswith(SAFETENSORS_EXTENSION)


def save_weights(model, path):
    """Save the weights of ``model`` in the format the extension of ``path`` tells."""
    if is_safetensors(path):
        # save_model rather than save_file(state_dict): it deals with the tensors that
        # several parameters share, such as tied embeddings, which save_file refuses
        from safetensors.torch import save_model

        save_model(model, str(path))
    else:
        torch.save(model.state_dict(), path)


def remove_other_weights(model_path, weight_file):
    """
    Remove from the directory ``model_path`` the weights the wrappers saved under another
    name than ``weight_file``, which was just written. Saving a model again in another
    format would otherwise leave the weights of the previous training next to the new
    ones, for whatever reads that file to load without a complaint.
    """
    for name in WEIGHT_FILE_NAMES:
        path = os.path.join(model_path, name)
        if name != weight_file and os.path.isfile(path):
            os.remove(path)


def load_weights(model, path, device=None):
    """Load into ``model`` the weights saved at ``path`` by ``save_weights``."""
    if is_safetensors(path):
        from safetensors.torch import load_model

        load_model(model, str(path), device="cpu" if device is None else str(device))
    else:
        model.load_state_dict(torch.load(path, map_location=device))


def find_weight_file(model_path, weight_file):
    """
    Path of the weights of the model in the directory ``model_path``: ``weight_file``
    when it is there, else the weights in the other format, so that a model loads the
    same whichever format it was saved or published in.
    """
    requested = os.path.join(model_path, weight_file)
    if os.path.isfile(requested) or not os.path.isdir(model_path):
        return requested

    if is_safetensors(weight_file):
        # the pickled weights are named differently by each wrapper
        candidates = [name for name in sorted(os.listdir(model_path)) if name.endswith((".pt", ".pth"))]
    else:
        candidates = [SAFETENSORS_WEIGHT_FILE_NAME]

    for candidate in candidates:
        path = os.path.join(model_path, candidate)
        if os.path.isfile(path):
            return path

    # let the caller fail on the file it asked for
    return requested
