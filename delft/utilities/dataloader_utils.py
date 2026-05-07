"""Shared helpers for PyTorch DataLoader configuration.

Used by both ``delft.sequenceLabelling.data_loader`` and
``delft.textClassification.data_loader``. Keeping the logic here prevents the
two task families from drifting as PyTorch / platform behaviour changes.
"""

import platform

import torch


def effective_num_workers(requested: int, dataset_size: int, batch_size: int, *, role: str = "loader") -> int:
    """Cap ``requested`` worker count for the dataset size and log the result.

    With ``persistent_workers=True`` (recommended at the call site) the pool
    is spawned once per fit, but extra workers still sit idle when batches
    are produced faster than the GPU/MPS step consumes them. We aim for
    ~2 batches per worker. Returns 0 (in-process loading) when the dataset
    is small enough that any worker is wasted.

    ``role`` is a free-form label (e.g. ``"train"``, ``"valid"``) used only
    in the log line so successive calls during one fit can be distinguished.
    """
    if requested <= 0 or dataset_size <= 0 or batch_size <= 0:
        print(
            f"DataLoader[{role}]: num_workers=0 (in-process); "
            f"dataset_size={dataset_size}, batch_size={batch_size}, "
            f"requested={requested}."
        )
        return 0
    num_batches = max(1, dataset_size // batch_size)
    ideal = num_batches // 2
    effective = max(0, min(requested, ideal))
    if effective != requested:
        print(
            f"DataLoader[{role}]: num_workers={effective} "
            f"(requested {requested}, capped for dataset_size={dataset_size}, "
            f"batches={num_batches})."
        )
    else:
        print(f"DataLoader[{role}]: num_workers={effective} (dataset_size={dataset_size}, batches={num_batches}).")
    return effective


def safe_multiprocessing_context():
    """Pick a GPU-safe multiprocessing context for DataLoader workers.

    The risk we're avoiding is GPU-driver state being inherited across a
    ``fork`` boundary, which segfaults workers on first use:

    * **Linux + CUDA**: the default ``fork`` is unsafe, and ``forkserver``
      only helps if the helper itself was started before CUDA initialised.
      Once ``self.model.to(device)`` has run in the parent, ``forkserver``
      forks from a CUDA-tainted parent and every worker dies. ``spawn``
      starts a fresh interpreter per worker — no inheritance, no driver
      state to corrupt. ``persistent_workers=True`` (set at the call
      sites) amortises the spawn cost across the whole fit.
    * **Linux + CPU only**: no CUDA to worry about, so ``forkserver`` is
      both safe and noticeably cheaper than ``spawn``.
    * **macOS** (CPU or MPS): Python's default is already ``spawn`` since
      PEP 690, for unrelated Objective-C / CoreFoundation fork-safety
      reasons. Return ``"spawn"`` explicitly so the choice survives any
      future Python-default change, and so MPS gets the same protection
      CUDA gets (Metal contexts are no more fork-safe than CUDA ones).
    * Other platforms: defer to PyTorch's default.
    """
    system = platform.system()
    if system == "Linux":
        if torch.cuda.is_initialized():
            return "spawn"
        return "forkserver"
    if system == "Darwin":
        return "spawn"
    return None
