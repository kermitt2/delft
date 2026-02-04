#!/usr/bin/env python3
"""
Script to update existing wandb runs in the delft project to add the short_model_name field.

The short_model_name is extracted from the run name by:
1. Removing the "grobid-" prefix if present
2. Removing the architecture suffix (e.g., "-BidLSTM_CRF", "-BERT_CRF", etc.)

Examples:
- "grobid-date-BidLSTM_CRF" -> "date"
- "grobid-header-BERT_CRF" -> "header"
- "grobid-reference-segmenter-BidLSTM_CRF_FEATURES" -> "reference-segmenter"

Usage:
    python scripts/update_wandb_short_model_name.py [--project PROJECT_NAME] [--dry-run]
"""

import argparse
import re


def extract_short_model_name(run_name: str) -> str:
    """Extract short model name from a full run name.
    
    Args:
        run_name: Full run name like "grobid-date-BidLSTM_CRF"
        
    Returns:
        Short model name like "date"
    """
    # Known architectures to strip from the end
    architectures = [
        "BidLSTM_CRF_FEATURES",
        "BidLSTM_ChainCRF_FEATURES", 
        "BidLSTM_CRF_CASING",
        "BidLSTM_ChainCRF",
        "BidLSTM_CNN_CRF",
        "BidLSTM_CRF",
        "BidLSTM_CNN",
        "BidLSTM",
        "BidGRU_CRF",
        "BERT_CRF_CHAR_FEATURES",
        "BERT_ChainCRF_FEATURES",
        "BERT_CRF_FEATURES",
        "BERT_CRF_CHAR",
        "BERT_ChainCRF",
        "BERT_FEATURES",
        "BERT_CRF",
        "BERT",
    ]
    
    name = run_name
    
    # Remove "grobid-" prefix if present
    if name.startswith("grobid-"):
        name = name[7:]
    
    # Remove architecture suffix
    for arch in architectures:
        if name.endswith("-" + arch):
            name = name[: -(len(arch) + 1)]
            break
    
    return name


def update_runs(project: str = "delft", entity: str = None, dry_run: bool = False):
    """Update all runs in the project to add short_model_name config.
    
    Args:
        project: Wandb project name
        entity: Wandb entity/username (optional, uses default if not specified)
        dry_run: If True, only print what would be done without making changes
    """
    try:
        import wandb
    except ImportError:
        print("Error: wandb is not installed. Install with: pip install wandb")
        return
    
    api = wandb.Api()
    
    # Get all runs in the project
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from project: {path}")
    
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for run in runs:
        run_name = run.name
        
        # Check if short_model_name already exists
        if run.config.get("short_model_name"):
            print(f"  Skipping '{run_name}' - already has short_model_name: {run.config['short_model_name']}")
            skipped_count += 1
            continue
        
        short_name = extract_short_model_name(run_name)
        
        if dry_run:
            print(f"  [DRY RUN] Would update '{run_name}' -> short_model_name: '{short_name}'")
            updated_count += 1
        else:
            try:
                run.config["short_model_name"] = short_name
                run.update()
                print(f"  Updated '{run_name}' -> short_model_name: '{short_name}'")
                updated_count += 1
            except Exception as e:
                print(f"  Error updating '{run_name}': {e}")
                error_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    if dry_run:
        print("\n[DRY RUN] No changes were made. Remove --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Update existing wandb runs to add short_model_name field"
    )
    parser.add_argument(
        "--project",
        default="delft",
        help="Wandb project name (default: delft)"
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Wandb entity/username (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    update_runs(
        project=args.project,
        entity=args.entity,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
