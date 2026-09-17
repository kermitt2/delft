"""
List and download the models published on the Hugging Face Hub.

    python -m delft.applications.hub_models list header
    python -m delft.applications.hub_models pull header --architecture BidLSTM_CRF_FEATURES
    python -m delft.applications.hub_models pull header --all
    python -m delft.applications.hub_models pull hf://lfoppiano/grobid-model-header@v1.1.0 --match "*potion*"

A task such as "header" stands for its repository as the resources registry tells it,
with --bucket for the models of that task in the bucket of the registry instead.
"""

import argparse
import os
import sys

from delft import DELFT_PROJECT_DIR
from delft.utilities.Embeddings import load_resource_registry
from delft.utilities.hub_models import (
    REGISTRY_SECTION,
    HubReference,
    is_hub_reference,
    list_models,
    parse_reference,
    resolve_model,
    select_models,
)
from delft.utilities.model_names import split_model_name

DEFAULT_OUTPUT = "data/models/sequenceLabelling"


def locate(name, registry, bucket=False):
    """The repository or bucket ``name`` stands for, and the task to keep the models of
    when that place holds the models of several."""
    if is_hub_reference(name):
        return parse_reference(name), None

    section = (registry or {}).get(REGISTRY_SECTION) or {}
    if bucket:
        if not section.get("bucket"):
            raise ValueError(f'The resources registry names no bucket, under "{REGISTRY_SECTION}"')
        return HubReference(section["bucket"], bucket=True), name
    if not section.get("repo"):
        raise ValueError(f'The resources registry names no repository, under "{REGISTRY_SECTION}"')
    return HubReference(section["repo"].format(short_name=name), revision=section.get("revision") or None), None


def models_of(location, task=None, architecture=None, suffix=None, match=None):
    model_names = list_models(location)
    if task is not None:
        model_names = [name for name in model_names if getattr(split_model_name(name), "short_name", None) == task]
    return select_models(model_names, architecture=architecture, suffix=suffix, match=match)


def main(argv=None):
    parser = argparse.ArgumentParser(description="List and download the DeLFT models of the Hugging Face Hub")
    parser.add_argument("action", choices=["list", "pull"])
    parser.add_argument("location", help='A task, e.g. "header", or a repository or bucket, e.g. hf://owner/repository')
    parser.add_argument("--bucket", action="store_true", help="With a task, look in the bucket of the registry")
    parser.add_argument("--architecture", help="Models of this architecture, with any suffix or none")
    parser.add_argument("--suffix", help='Models with this suffix, "" for the models without one')
    parser.add_argument("--match", help='Models whose name matches this pattern, e.g. "*potion*"')
    parser.add_argument("--all", action="store_true", help="pull: every model, when nothing else selects some")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"pull: models directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--force", action="store_true", help="pull: download again the models already there")
    args = parser.parse_args(argv)

    registry = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))
    location, task = locate(args.location, registry, bucket=args.bucket)
    selection = {"architecture": args.architecture, "suffix": args.suffix, "match": args.match}

    if args.action == "pull" and not args.all and all(value is None for value in selection.values()):
        parser.error("pull needs a selection (--architecture, --suffix, --match) or --all")

    model_names = models_of(location, task, **selection)
    if args.action == "list":
        print(f"{location}: {len(model_names)} model(s)")
        for model_name in model_names:
            print("  " + model_name)
        return model_names

    if not model_names:
        print(f"{location}: no model selected", file=sys.stderr)
    directories = []
    for model_name in model_names:
        directory = resolve_model(location.for_model(model_name), cache_dir=args.output, force=args.force)
        print(directory)
        directories.append(directory)
    return directories


if __name__ == "__main__":
    main()
