"""
Tests for delft/utilities/cuda_setup.py

Both policy functions are exercised without an actual GPU by patching
torch.cuda primitives. The arch-list validator is the most important
to cover: it raises a clear error before training starts when the
installed torch wheel lacks kernels for the device's compute capability
(e.g. cu128/cu130 wheels on a V100 sm_70).
"""

from unittest.mock import patch

import pytest
import torch

from delft.utilities.cuda_setup import (
    CUDNN9_MIN_COMPUTE_CAPABILITY,
    configure_cudnn_for_device,
    validate_device_arch_compatibility,
)


class TestValidateDeviceArchCompatibility:
    def test_cpu_device_is_noop(self):
        validate_device_arch_compatibility(torch.device("cpu"))

    def test_mps_device_is_noop(self):
        validate_device_arch_compatibility(torch.device("mps"))

    def test_v100_on_cu128_wheel_raises(self):
        # V100 sm_70 against the arch list shipped by cu128/cu130 wheels.
        device = torch.device("cuda:0")
        arch_list = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
        with (
            patch("torch.cuda.get_arch_list", return_value=arch_list),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
            patch("torch.cuda.get_device_name", return_value="Tesla V100-SXM2-32GB"),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                validate_device_arch_compatibility(device)
        message = str(excinfo.value)
        assert "sm_70" in message
        assert "Tesla V100" in message
        assert "gpu-pre-turing" in message
        assert "cu126" in message

    def test_modern_gpu_in_arch_list_is_noop(self):
        device = torch.device("cuda:0")
        arch_list = ["sm_75", "sm_80", "sm_86", "sm_90"]
        with (
            patch("torch.cuda.get_arch_list", return_value=arch_list),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
            patch("torch.cuda.get_device_name", return_value="NVIDIA RTX A6000"),
        ):
            validate_device_arch_compatibility(device)

    def test_l40s_runs_on_lower_minor_kernels(self):
        # L40S is sm_89; the cu130 wheel ships sm_86 but not sm_89. CUDA's
        # SASS forward-compat (sm_86 cubin runs on sm_89) means this is
        # supported, even though sm_89 is absent from the arch list.
        device = torch.device("cuda:0")
        arch_list = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
        with (
            patch("torch.cuda.get_arch_list", return_value=arch_list),
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            patch("torch.cuda.get_device_name", return_value="NVIDIA L40S"),
        ):
            validate_device_arch_compatibility(device)

    def test_empty_arch_list_is_noop(self):
        # CPU-only torch builds report an empty arch list; nothing to
        # check, and we do not want to raise on those.
        device = torch.device("cuda:0")
        with (
            patch("torch.cuda.get_arch_list", return_value=[]),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
        ):
            validate_device_arch_compatibility(device)


class TestConfigureCudnnForDevice:
    def test_cpu_device_is_noop(self):
        original = torch.backends.cudnn.enabled
        configure_cudnn_for_device(torch.device("cpu"))
        assert torch.backends.cudnn.enabled is original

    def test_disables_cudnn_for_v100(self):
        # `torch.backends.cudnn.enabled` is a module-level property descriptor,
        # so `patch.object` cannot restore it on teardown — save/restore manually.
        device = torch.device("cuda:0")
        original = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            with (
                patch("torch.cuda.get_device_capability", return_value=(7, 0)),
                patch("torch.cuda.get_device_name", return_value="Tesla V100-SXM2-32GB"),
            ):
                configure_cudnn_for_device(device)
                assert torch.backends.cudnn.enabled is False
        finally:
            torch.backends.cudnn.enabled = original

    def test_keeps_cudnn_for_modern_gpu(self):
        device = torch.device("cuda:0")
        original = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            with patch("torch.cuda.get_device_capability", return_value=CUDNN9_MIN_COMPUTE_CAPABILITY):
                configure_cudnn_for_device(device)
                assert torch.backends.cudnn.enabled is True
        finally:
            torch.backends.cudnn.enabled = original
