"""CUDA runtime configuration shared by both task families.

Houses two related-but-distinct policies for pre-Turing GPUs (V100 sm_70,
P100 sm_60, etc.):

1. ``configure_cudnn_for_device``: PyTorch >= 2.4 bundles cuDNN 9, which
   raises ``RuntimeError: cuDNN version ... is not compatible with
   devices with SM < 7.5`` the first time a cuDNN-backed RNN module
   (e.g. ``nn.LSTM`` inside the BidLSTM_* architectures) is moved to a
   pre-Turing GPU. Mitigated by disabling cuDNN so PyTorch falls back
   to its native (non-fused) RNN kernels — slower, but functional.

2. ``validate_device_arch_compatibility``: PyTorch's CUDA 12.8+ wheels
   dropped sm_70 *binary kernels* entirely, so the first generic CUDA
   op (e.g. ``nn.Embedding`` lookup, which is not a cuDNN op) crashes
   with ``cudaErrorNoKernelImageForDevice``. Disabling cuDNN cannot
   help — the missing kernels are not cuDNN-backed. Mitigated by
   detecting the mismatch and raising a clear, actionable error before
   the user wastes a job slot mid-training.

Both checks are capability-based and run once per process, called from
each wrapper's ``__init__`` immediately after ``pick_device()``.
"""

import torch

# cuDNN 9 dropped support for compute capabilities below this. Set as the
# constant the policy compares against so the rule matches the upstream
# constraint rather than enumerating specific violator GPUs.
CUDNN9_MIN_COMPUTE_CAPABILITY = (7, 5)


def configure_cudnn_for_device(device: torch.device) -> None:
    """Disable cuDNN if the selected device pre-dates the cuDNN 9 floor.

    Called once per training/inference run, right after the device has
    been picked and *before* any model is constructed or moved to the
    device — cuDNN is initialised lazily on the first cuDNN-backed op,
    so flipping the switch later than that is too late.

    No-op for non-CUDA devices (CPU, MPS), since cuDNN is CUDA-only.
    No-op if the user (or another part of the code) has already disabled
    cuDNN — we don't want to second-guess explicit configuration.

    Args:
        device: the device returned by ``pick_device()``.
    """
    if device.type != "cuda":
        return
    if not torch.backends.cudnn.enabled:
        return

    capability = torch.cuda.get_device_capability(device)
    if capability < CUDNN9_MIN_COMPUTE_CAPABILITY:
        torch.backends.cudnn.enabled = False
        name = torch.cuda.get_device_name(device)
        cap_str = f"sm_{capability[0]}{capability[1]}"
        floor_str = f"sm_{CUDNN9_MIN_COMPUTE_CAPABILITY[0]}{CUDNN9_MIN_COMPUTE_CAPABILITY[1]}"
        print(
            f"cuDNN disabled: {name} ({cap_str}) is below the cuDNN 9 floor "
            f"({floor_str}); RNN training will use PyTorch's native kernels "
            f"and may be slower"
        )


def validate_device_arch_compatibility(device: torch.device) -> None:
    """Fail fast if the installed torch wheel lacks kernels for ``device``.

    PyTorch's CUDA 12.8+ wheels dropped sm_70 binary kernels, so V100/P100
    cards crash with ``cudaErrorNoKernelImageForDevice`` on the first
    generic CUDA op (e.g. ``nn.Embedding`` lookup). The cuDNN fallback in
    ``configure_cudnn_for_device`` does not help — the missing kernels
    are not cuDNN-backed. Detect the mismatch up front and raise a clear,
    actionable error rather than letting the user hit a cryptic crash
    mid-training.

    The arch list reports the targets the wheel was actually compiled
    for; we require an exact match for the device's compute capability.
    PyTorch will JIT-compile PTX for forward-compatible higher arches,
    but the wheels we're guarding against ship neither sm_70 SASS nor
    sm_70-compatible PTX, so a strict membership check is correct here.

    No-op for non-CUDA devices (CPU, MPS).
    """
    if device.type != "cuda":
        return

    arch_list = torch.cuda.get_arch_list()
    if not arch_list:
        return

    capability = torch.cuda.get_device_capability(device)
    cap_str = f"sm_{capability[0]}{capability[1]}"
    if cap_str in arch_list:
        return

    name = torch.cuda.get_device_name(device)
    raise RuntimeError(
        f"Installed torch wheel ({torch.__version__}) has no CUDA kernels for "
        f"{name} ({cap_str}). Wheel ships kernels for: {arch_list}. "
        f"This would crash mid-training with `cudaErrorNoKernelImageForDevice` "
        f"on the first CUDA op. For pre-Turing GPUs (V100 sm_70, P100 sm_60), "
        f"reinstall with the dedicated extra:\n"
        f'    pip install -e ".[gpu-pre-turing]" '
        f"--extra-index-url https://download.pytorch.org/whl/cu126\n"
        f"See doc/Install-DeLFT.md for details."
    )
