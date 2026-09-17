"""
Publishing models on the Hugging Face Hub, where ``delft.utilities.hub_models`` gets
them from: a model directory becomes the folder of the same name of a model repository
or of a bucket.
"""

import json
import logging
import os
import shutil
import tempfile

from delft.utilities.hub_models import HubReference, hub_references, list_models, parse_reference
from delft.utilities.model_names import split_model_name
from delft.utilities.weights import SAFETENSORS_WEIGHT_FILE_NAME, is_safetensors

LOGGER = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.json"
README_FILE_NAME = "README.md"
PICKLED_WEIGHTS_EXTENSIONS = (".pt", ".pth")

# the table of the models is rewritten between these two lines, the rest of a README
# being left as it was written
TABLE_START = "<!-- models: start, this table is generated -->"
TABLE_END = "<!-- models: end -->"

MODEL_CARD_HEADER = """---
library_name: delft
tags:
- delft
- sequence-labelling
---

# {title}

Models for [DeLFT](https://github.com/kermitt2/delft). Each folder is a model, to load with

```python
from delft.sequenceLabelling import Sequence

model = Sequence("<folder>")
model.load("{reference}/<folder>")
```

"""


def convert_to_safetensors(pickled_path, safetensors_path):
    """Write as safetensors the state dict pickled at ``pickled_path``."""
    import torch
    from safetensors.torch import save_file

    state_dict = torch.load(pickled_path, map_location="cpu", weights_only=True)
    try:
        save_file({name: tensor.contiguous() for name, tensor in state_dict.items()}, safetensors_path)
    except RuntimeError as e:
        raise RuntimeError(
            f"{pickled_path} cannot be converted to safetensors as it is, several of its tensors sharing their "
            'memory. Load the model and save it again with weight_file="model.safetensors".'
        ) from e


def prepare_model(model_dir, staging_dir, safetensors=True):
    """
    Copy to ``staging_dir`` what is published of the model in ``model_dir``, and return
    the names of the files: everything but the hidden files, such as the record of
    where a model was downloaded from, with the weights as safetensors instead of
    pickled unless ``safetensors`` is False.
    """
    names = sorted(name for name in os.listdir(model_dir) if not name.startswith("."))
    names = [name for name in names if os.path.isfile(os.path.join(model_dir, name))]
    pickled = [name for name in names if name.endswith(PICKLED_WEIGHTS_EXTENSIONS)]
    convert = safetensors and not any(is_safetensors(name) for name in names)
    if convert and len(pickled) > 1:
        raise ValueError(f"{model_dir} holds several weight files, {pickled}: which to publish is ambiguous")

    os.makedirs(staging_dir, exist_ok=True)
    published = []
    for name in names:
        if safetensors and name in pickled:
            if convert:
                convert_to_safetensors(
                    os.path.join(model_dir, name), os.path.join(staging_dir, SAFETENSORS_WEIGHT_FILE_NAME)
                )
                published.append(SAFETENSORS_WEIGHT_FILE_NAME)
            continue
        shutil.copy(os.path.join(model_dir, name), os.path.join(staging_dir, name))
        published.append(name)
    return sorted(published)


def _read_remote_text(location, path, token=None):
    """Content of a text file of a repository or bucket, None when it is not there."""
    from huggingface_hub import errors

    with tempfile.TemporaryDirectory() as directory:
        try:
            if location.bucket:
                from huggingface_hub import download_bucket_files

                local_path = os.path.join(directory, "file")
                download_bucket_files(location.repo_id, files=[(path, local_path)], token=token)
            else:
                from huggingface_hub import hf_hub_download

                local_path = hf_hub_download(
                    location.repo_id, path, revision=location.revision, local_dir=directory, token=token
                )
            with open(local_path, encoding="utf-8") as f:
                return f.read()
        except (errors.EntryNotFoundError, FileNotFoundError):
            return None


def models_table(configs):
    """Markdown table of models, ``configs`` giving the config.json content of each by
    name, or None for a model without one such as a Wapiti model."""
    lines = [
        "| Model | Architecture | Embeddings | Transformer | Suffix |",
        "|---|---|---|---|---|",
    ]
    for model_name in sorted(configs):
        config = configs[model_name] or {}
        split = split_model_name(model_name)
        cells = [
            f"`{model_name}`",
            config.get("architecture") or (split.architecture if split else ""),
            config.get("embeddings_name") or "",
            config.get("transformer_name") or "",
            (split.suffix if split else None) or "",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines)


def update_readme(readme, table, location):
    """``readme`` with its table of models replaced by ``table``. A README without one
    gets it at its end, and a new README starts with a model card."""
    block = f"{TABLE_START}\n{table}\n{TABLE_END}"
    if readme is None:
        title = location.repo_id.split("/")[-1]
        return MODEL_CARD_HEADER.format(title=title, reference=str(location)) + "## Models\n\n" + block + "\n"
    if TABLE_START in readme and TABLE_END in readme:
        start = readme.index(TABLE_START)
        end = readme.index(TABLE_END) + len(TABLE_END)
        return readme[:start] + block + readme[end:]
    return readme.rstrip("\n") + "\n\n## Models\n\n" + block + "\n"


def _upload(location, staging_dir, model_name, readme, token, commit_message):
    if location.bucket:
        from huggingface_hub import batch_bucket_files

        files = [(os.path.join(staging_dir, name), f"{model_name}/{name}") for name in sorted(os.listdir(staging_dir))]
        if readme is not None:
            files.append((readme.encode("utf-8"), README_FILE_NAME))
        batch_bucket_files(location.repo_id, add=files, token=token)
    else:
        from huggingface_hub import HfApi

        api = HfApi()
        # delete_patterns: what a previous version of the model held and this one does
        # not, its pickled weights for instance, does not stay next to the new files
        api.upload_folder(
            repo_id=location.repo_id,
            folder_path=staging_dir,
            path_in_repo=model_name,
            commit_message=commit_message,
            revision=location.revision,
            delete_patterns=["*"],
            token=token,
        )
        if readme is not None:
            api.upload_file(
                path_or_fileobj=readme.encode("utf-8"),
                path_in_repo=README_FILE_NAME,
                repo_id=location.repo_id,
                revision=location.revision,
                commit_message=f"List {model_name}",
                token=token,
            )


def push_model(model_dir, location, token=None, private=None, safetensors=True, readme=True):
    """
    Publish the model in ``model_dir`` as the folder of its name in ``location``, a
    repository (``hf://owner/repository``) or a bucket (``hf://buckets/owner/bucket``),
    created when it does not exist, and return the reference of the published model.

    The weights are published as safetensors, converted from the pickled ones when the
    model has no others, and the table of the models in the README is brought up to
    date, unless ``safetensors`` or ``readme`` is False.
    """
    location = parse_reference(location)
    if location.model_name is not None:
        raise ValueError(f"{location} names a model: the place to publish to is {location.for_model(None)}")

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    model_name = os.path.basename(os.path.normpath(model_dir))
    if not os.path.isdir(model_dir) or not any(not name.startswith(".") for name in os.listdir(model_dir)):
        raise FileNotFoundError(f"No model in {model_dir}")

    if location.bucket:
        from huggingface_hub import create_bucket

        create_bucket(location.repo_id, private=private, exist_ok=True, token=token)
    else:
        from huggingface_hub import create_repo

        create_repo(location.repo_id, private=private, exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as staging_dir:
        published = prepare_model(model_dir, staging_dir, safetensors=safetensors)

        new_readme = None
        if readme:
            configs = {}
            for other in list_models(location, token=token):
                content = _read_remote_text(location, f"{other}/{CONFIG_FILE_NAME}", token=token)
                configs[other] = json.loads(content) if content else None
            configs[model_name] = None
            if os.path.isfile(os.path.join(staging_dir, CONFIG_FILE_NAME)):
                with open(os.path.join(staging_dir, CONFIG_FILE_NAME), encoding="utf-8") as f:
                    configs[model_name] = json.load(f)
            current = _read_remote_text(location, README_FILE_NAME, token=token)
            new_readme = update_readme(current, models_table(configs), location)

        LOGGER.info("publishing %s to %s: %s", model_dir, location, ", ".join(published))
        _upload(location, staging_dir, model_name, new_readme, token, commit_message=f"Publish {model_name}")

    return location.for_model(model_name)


def push_location(model_name, registry, bucket=False):
    """Where the resources registry says to publish the model ``model_name``: its
    repository, or the bucket. The revision of the registry is the one to read models
    at, often a tag: a model is published to the default branch."""
    for reference in hub_references(model_name, registry):
        if reference.bucket == bucket:
            return HubReference(reference.repo_id, bucket=reference.bucket)
    place = "bucket" if bucket else "repository"
    raise ValueError(f"The resources registry gives the model {model_name} no {place} to be published to")
