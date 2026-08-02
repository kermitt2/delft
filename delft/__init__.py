import os

import torch

DELFT_PROJECT_DIR = os.path.dirname(__file__)

# Floor for the intra-op pool when a host has clamped torch down to almost
# nothing. Measured on GROBID-shaped work (one sequence per call, a few dozen
# tokens): 4 threads is already at the plateau, and going past the physical
# core count costs more in synchronisation than it buys in parallelism.
MIN_INTRA_OP_THREADS = 4


def configure_threads():
    """
    Make sure torch has a usable CPU thread pool, without oversubscribing.

    Left alone, torch sizes its intra-op pool to the physical core count, which
    is what TensorFlow's intra_op default effectively gave us and which
    measures at the plateau for inference. What we are guarding against is an
    embedding host — GROBID through JEP — handing torch a much smaller pool
    than the machine can support; there we raise it, but never above what torch
    picked for itself.

    Deliberately *not* os.cpu_count(): that counts hyperthreads, and pinning
    the pool to it made per-sequence tagging ~70% slower on a 12-core/24-thread
    Xeon than leaving torch's default in place.

    OMP_NUM_THREADS (and MKL_NUM_THREADS) steer oneDNN/OpenMP and are read when
    the native libraries load, i.e. at ``import torch`` — setting them from
    Python afterwards is too late. When either is set we leave the pool alone,
    since the operator has already sized it.

    This has to run before any parallel work starts: ``set_num_interop_threads``
    raises once the inter-op pool is initialised, hence the call at package
    import, below. Harmless on GPU runs — it only sizes CPU thread pools.
    """
    if "OMP_NUM_THREADS" in os.environ or "MKL_NUM_THREADS" in os.environ:
        return

    n_logical = os.cpu_count() or 1
    intra_op = max(torch.get_num_threads(), min(MIN_INTRA_OP_THREADS, n_logical))
    torch.set_num_threads(intra_op)

    try:
        torch.set_num_interop_threads(max(1, intra_op // 2))
    except RuntimeError:
        # Inter-op pool already initialised (parallel work has begun, or
        # configure_threads() ran twice) — the existing setting stands.
        pass


configure_threads()
