import os

import torch

DELFT_PROJECT_DIR = os.path.dirname(__file__)

# Floor for the intra-op pool when a host has clamped torch down to almost
# nothing. Measured on GROBID-shaped work (one sequence per call, a few dozen
# tokens): 4 threads is already at the plateau, and going past the physical
# core count costs more in synchronisation than it buys in parallelism.
MIN_INTRA_OP_THREADS = 4


def cgroup_cpu_limit():
    """
    CPU cores this process may actually use, or None when unrestricted.

    torch sizes its pool from the host's core count and knows nothing about
    cgroup quotas, so a container run with ``--cpus=2`` on a 64-core host still
    gets a 32-thread pool: every inference then spends its time in OpenMP
    barriers waiting for threads the scheduler will not run. Reading the quota
    ourselves is the only way to see the limit from inside the container.
    """
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()  # cgroup v2
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass

    try:  # cgroup v1, where a negative quota means unrestricted
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass

    return None


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

    Where torch's default is wrong is inside a CPU-limited container: it counts
    the host's cores and ignores the cgroup quota, so we cap it at the quota.

    OMP_NUM_THREADS (and MKL_NUM_THREADS) steer oneDNN/OpenMP and are read when
    the native libraries load, i.e. at ``import torch`` — setting them from
    Python afterwards is too late. When either is set we leave the pool alone,
    since the operator has already sized it. Note that pinning them to 1 is a
    real handicap and not a safe default: measured on GROBID-shaped work with
    ten documents in flight, one thread tags 43% slower than torch's default.

    This has to run before any parallel work starts: ``set_num_interop_threads``
    raises once the inter-op pool is initialised, hence the call at package
    import, below. Harmless on GPU runs — it only sizes CPU thread pools.
    """
    if "OMP_NUM_THREADS" in os.environ or "MKL_NUM_THREADS" in os.environ:
        return

    n_logical = os.cpu_count() or 1
    intra_op = max(torch.get_num_threads(), min(MIN_INTRA_OP_THREADS, n_logical))

    limit = cgroup_cpu_limit()
    if limit is not None:
        intra_op = max(1, min(intra_op, int(limit)))

    torch.set_num_threads(intra_op)

    try:
        torch.set_num_interop_threads(max(1, intra_op // 2))
    except RuntimeError:
        # Inter-op pool already initialised (parallel work has begun, or
        # configure_threads() ran twice) — the existing setting stands.
        pass


configure_threads()
