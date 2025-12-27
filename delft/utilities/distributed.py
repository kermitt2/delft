"""
Distributed training utilities for PyTorch DistributedDataParallel (DDP).

This module provides helper functions for setting up and managing distributed
training across multiple GPUs using PyTorch's DDP.

Usage:
    Launch with torchrun:
    ```bash
    torchrun --nproc_per_node=4 -m delft.applications.grobidTagger train --multi-gpu
    ```
"""

import os
import logging
from typing import Dict, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available() and dist.is_initialized()


def setup_distributed(backend: str = "nccl") -> int:
    """
    Initialize the distributed process group.

    This should be called at the start of training when multi-GPU is enabled.
    Uses environment variables set by torchrun/torch.distributed.launch.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)

    Returns:
        local_rank: The local GPU rank for this process
    """
    # Get environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size <= 1:
        logger.info("World size is 1, running in single-GPU mode")
        return local_rank

    # Set device before initializing process group
    torch.cuda.set_device(local_rank)

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        logger.info(
            f"Initialized distributed training: "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

    return local_rank


def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Use this to ensure operations like saving checkpoints and logging
    are only performed once across all processes.

    Returns:
        True if this is the main process or if not in distributed mode
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the global rank of this process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node) of this process."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict: Dict[str, Any], average: bool = True) -> Dict[str, float]:
    """
    Reduce a dictionary of tensors/values across all processes.

    Args:
        input_dict: Dictionary with string keys and numeric values
        average: If True, average the values; if False, sum them

    Returns:
        Dictionary with reduced values (only valid on rank 0, but returned on all)
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return {k: float(v) for k, v in input_dict.items()}

    world_size = get_world_size()

    # Convert to tensors and gather
    keys = sorted(input_dict.keys())
    values = torch.tensor(
        [float(input_dict[k]) for k in keys],
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )

    # All-reduce
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    if average:
        values /= world_size

    return {k: v.item() for k, v in zip(keys, values)}


def reduce_metric(value: float, average: bool = True) -> float:
    """
    Reduce a single metric value across all processes.

    Args:
        value: The metric value to reduce
        average: If True, average; if False, sum

    Returns:
        Reduced value
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return value

    tensor = torch.tensor(
        value, dtype=torch.float32, device=torch.cuda.current_device()
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if average:
        tensor /= get_world_size()

    return tensor.item()


def print_once(*args, **kwargs):
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)
