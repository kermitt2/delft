from unittest.mock import mock_open, patch

from delft import cgroup_cpu_limit, cpu_affinity_count

CGROUP_V2 = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def fake_cgroup(files):
    """Stand in for open(), serving `files` and raising OSError for anything else."""

    def _open(path, *args, **kwargs):
        if path not in files:
            raise FileNotFoundError(path)
        return mock_open(read_data=files[path])()

    return patch("builtins.open", _open)


class TestCgroupCpuLimit:
    def test_reads_the_quota_from_cgroup_v2(self):
        with fake_cgroup({CGROUP_V2: "200000 100000"}):
            assert cgroup_cpu_limit() == 2.0

    def test_returns_none_when_cgroup_v2_is_unlimited(self):
        with fake_cgroup({CGROUP_V2: "max 100000"}):
            assert cgroup_cpu_limit() is None

    def test_reads_a_fractional_quota_from_cgroup_v2(self):
        with fake_cgroup({CGROUP_V2: "150000 100000"}):
            assert cgroup_cpu_limit() == 1.5

    def test_reads_the_quota_from_cgroup_v1(self):
        with fake_cgroup({CGROUP_V1_QUOTA: "400000", CGROUP_V1_PERIOD: "100000"}):
            assert cgroup_cpu_limit() == 4.0

    def test_returns_none_when_cgroup_v1_is_unlimited(self):
        with fake_cgroup({CGROUP_V1_QUOTA: "-1", CGROUP_V1_PERIOD: "100000"}):
            assert cgroup_cpu_limit() is None

    def test_falls_back_to_cgroup_v1_when_v2_is_absent(self):
        with fake_cgroup({CGROUP_V1_QUOTA: "300000", CGROUP_V1_PERIOD: "100000"}):
            assert cgroup_cpu_limit() == 3.0

    def test_returns_none_when_no_cgroup_interface_exists(self):
        with fake_cgroup({}):
            assert cgroup_cpu_limit() is None

    def test_returns_none_when_the_quota_is_unparseable(self):
        with fake_cgroup({CGROUP_V2: "garbage"}):
            assert cgroup_cpu_limit() is None


class TestCpuAffinityCount:
    def test_counts_the_cores_in_the_affinity_mask(self):
        with patch("os.sched_getaffinity", return_value={0, 1, 2, 5}):
            assert cpu_affinity_count() == 4

    def test_returns_none_when_the_platform_has_no_affinity_call(self):
        with patch("os.sched_getaffinity", side_effect=AttributeError):
            assert cpu_affinity_count() is None

    def test_returns_none_when_the_affinity_call_fails(self):
        with patch("os.sched_getaffinity", side_effect=OSError):
            assert cpu_affinity_count() is None
