from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools import pod_resources as pr


def write(root: Path, absolute: str, content: str) -> Path:
    path = root.joinpath(*[part for part in absolute.split("/") if part])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def gib(value: int | float) -> int:
    return int(value * pr.GIB)


def host_meminfo(total_gib: int = 64, available_gib: int = 48) -> str:
    total_kib = gib(total_gib) // 1024
    available_kib = gib(available_gib) // 1024
    free_kib = max(0, available_kib - gib(4) // 1024)
    return f"""\
MemTotal:       {total_kib} kB
MemFree:        {free_kib} kB
MemAvailable:   {available_kib} kB
Buffers:        0 kB
Cached:         {gib(4) // 1024} kB
SReclaimable:   0 kB
SwapTotal:      {gib(8) // 1024} kB
SwapFree:       {gib(7) // 1024} kB
"""


def setup_proc(root: Path, cgroup: str, mountinfo: str) -> None:
    write(root, "/proc/meminfo", host_meminfo())
    write(root, "/proc/self/cgroup", cgroup)
    write(root, "/proc/self/mountinfo", mountinfo)


def write_verified_assignment(root: Path, capacity_bytes: int) -> Path:
    receipt = write(
        root,
        pr.VERIFIED_MEMORY_RECEIPT,
        f"{capacity_bytes}\n",
    )
    receipt.parent.chmod(0o755)
    receipt.chmod(0o444)
    return receipt


V2_MOUNT = (
    "29 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime "
    "- cgroup2 cgroup rw\n"
)


def setup_v2_leaf(root: Path, membership: str) -> Path:
    setup_proc(root, f"0::{membership}\n", V2_MOUNT)
    leaf = root / "sys/fs/cgroup" / membership.lstrip("/")
    leaf.mkdir(parents=True, exist_ok=True)
    return leaf


V1_MOUNTS = """\
31 23 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
32 23 0:28 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct
33 23 0:29 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset
"""


class PodResourcesTests(unittest.TestCase):
    def fixture_root(self):
        return tempfile.TemporaryDirectory()

    def test_v2_uses_finite_parent_limit_and_parent_usage(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pods/p1/container")
            parent = leaf.parent

            write(root, "/sys/fs/cgroup/pods/p1/container/memory.max", "max\n")
            write(root, "/sys/fs/cgroup/pods/p1/container/memory.current", f"{gib(5)}\n")
            write(root, "/sys/fs/cgroup/pods/p1/container/memory.stat", "inactive_file 0\n")
            write(root, "/sys/fs/cgroup/pods/p1/memory.max", f"{gib(16)}\n")
            write(root, "/sys/fs/cgroup/pods/p1/memory.current", f"{gib(6)}\n")
            write(
                root,
                "/sys/fs/cgroup/pods/p1/memory.stat",
                f"anon {gib(4)}\ninactive_file {gib(2)}\n",
            )
            write(root, "/sys/fs/cgroup/pods/p1/memory.high", f"{gib(12)}\n")
            write(root, "/sys/fs/cgroup/pods/p1/memory.swap.max", "0\n")
            write(root, "/sys/fs/cgroup/pods/p1/memory.swap.current", "0\n")
            write(root, "/sys/fs/cgroup/pods/p1/container/cpuset.cpus.effective", "0-7\n")
            write(root, "/sys/fs/cgroup/pods/p1/container/cpu.max", "max 100000\n")
            write(root, "/sys/fs/cgroup/pods/p1/cpu.max", "250000 100000\n")

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_CPU_COUNT": "4"},
                affinity_count=8,
                host_cpu_count=16,
            )

            self.assertEqual(result.cgroup_version, 2)
            self.assertTrue(result.memory.limited)
            self.assertEqual(result.memory.limit_bytes, gib(16))
            # The constraining parent includes sibling usage and is authoritative.
            self.assertEqual(result.memory.current_bytes, gib(6))
            self.assertEqual(result.memory.inactive_file_bytes, gib(2))
            self.assertEqual(result.memory.working_set_bytes, gib(4))
            self.assertEqual(result.memory.available_bytes, gib(12))
            self.assertEqual(result.memory.high_bytes, gib(12))
            self.assertEqual(result.memory.swap_limit_bytes, 0)
            self.assertTrue(result.memory.limit_source.endswith("/sys/fs/cgroup/pods/p1"))
            self.assertEqual(result.cpu.cpuset_count, 8)
            self.assertEqual(result.cpu.quota_cores, 2.5)
            self.assertEqual(result.cpu.quota_job_count, 2)
            self.assertEqual(result.cpu.effective_count, 2)
            self.assertEqual(result.build.suggested_jobs, 1)
            self.assertTrue(parent.exists())

    def test_v2_private_cgroup_namespace_maps_slash_to_mount_root(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            mountinfo = (
                "29 23 0:26 /docker/abc /sys/fs/cgroup rw,nosuid,nodev,noexec "
                "- cgroup2 cgroup rw\n"
            )
            setup_proc(root, "0::/\n", mountinfo)
            write(root, "/sys/fs/cgroup/memory.max", f"{gib(8)}\n")
            write(root, "/sys/fs/cgroup/memory.current", f"{gib(2)}\n")
            write(root, "/sys/fs/cgroup/memory.stat", f"inactive_file {gib(1)}\n")
            write(root, "/sys/fs/cgroup/cpu.max", "100000 100000\n")
            write(root, "/sys/fs/cgroup/cpuset.cpus.effective", "2-5\n")

            result = pr.collect_resources(
                filesystem_root=root,
                env={},
                affinity_count=6,
                host_cpu_count=12,
            )

            self.assertEqual(result.cgroup_path, "/")
            self.assertEqual(result.memory.limit_bytes, gib(8))
            self.assertEqual(result.memory.working_set_bytes, gib(1))
            self.assertEqual(result.cpu.cpuset_count, 4)
            self.assertEqual(result.cpu.effective_count, 1)

    def test_v2_unlimited_falls_back_to_host_memory(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/docker/abc")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(3)), encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")

            result = pr.collect_resources(
                filesystem_root=root,
                env={},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertFalse(result.memory.limited)
            self.assertEqual(result.memory.limit_source, "host")
            self.assertEqual(result.memory.limit_bytes, gib(64))
            self.assertEqual(result.memory.current_bytes, gib(16))
            self.assertEqual(result.memory.available_bytes, gib(48))

    def test_finite_limit_equal_to_host_total_remains_authoritative(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/equal")
            (leaf / "memory.max").write_text(str(gib(64)), encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(7)), encoding="ascii")
            (leaf / "memory.stat").write_text(
                f"inactive_file {gib(1)}\n", encoding="ascii"
            )

            result = pr.collect_resources(
                filesystem_root=root,
                env={},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertTrue(result.memory.limited)
            self.assertEqual(result.memory.limit_bytes, gib(64))
            self.assertEqual(result.memory.current_bytes, gib(7))
            self.assertEqual(result.memory.working_set_bytes, gib(6))
            self.assertTrue(result.memory.limit_source.startswith("cgroup-v2:"))

    def test_zero_is_a_finite_v2_limit(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/stopped")
            (leaf / "memory.max").write_text("0\n", encoding="ascii")
            (leaf / "memory.current").write_text("0\n", encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")

            result = pr.collect_resources(
                filesystem_root=root,
                env={},
                affinity_count=1,
                host_cpu_count=1,
            )
            self.assertTrue(result.memory.limited)
            self.assertEqual(result.memory.limit_bytes, 0)
            self.assertEqual(result.build.suggested_jobs, 1)

    def test_v1_hierarchical_limit_cpu_quota_and_cpuset(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            setup_proc(
                root,
                "7:memory:/docker/abc\n6:cpu,cpuacct:/docker/abc\n5:cpuset:/docker/abc\n",
                V1_MOUNTS,
            )
            memory = root / "sys/fs/cgroup/memory/docker/abc"
            memory.mkdir(parents=True)
            (memory / "memory.limit_in_bytes").write_text(
                "9223372036854771712\n", encoding="ascii"
            )
            (memory / "memory.usage_in_bytes").write_text(str(gib(4)), encoding="ascii")
            (memory / "memory.stat").write_text(
                f"hierarchical_memory_limit {gib(12)}\n"
                f"total_inactive_file {gib(1)}\n",
                encoding="ascii",
            )
            (memory / "memory.memsw.limit_in_bytes").write_text(
                str(gib(16)), encoding="ascii"
            )
            (memory / "memory.memsw.usage_in_bytes").write_text(
                str(gib(5)), encoding="ascii"
            )

            cpu = root / "sys/fs/cgroup/cpu,cpuacct/docker/abc"
            cpu.mkdir(parents=True)
            (cpu / "cpu.cfs_quota_us").write_text("180000\n", encoding="ascii")
            (cpu / "cpu.cfs_period_us").write_text("100000\n", encoding="ascii")

            cpuset = root / "sys/fs/cgroup/cpuset/docker/abc"
            cpuset.mkdir(parents=True)
            (cpuset / "cpuset.cpus").write_text("0-7\n", encoding="ascii")
            (cpuset.parent / "cpuset.cpus").write_text("0-3\n", encoding="ascii")

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_CPU_COUNT": "8"},
                affinity_count=6,
                host_cpu_count=32,
            )

            self.assertEqual(result.cgroup_version, 1)
            self.assertTrue(result.memory.limited)
            self.assertEqual(result.memory.limit_bytes, gib(12))
            self.assertEqual(result.memory.current_bytes, gib(4))
            self.assertEqual(result.memory.working_set_bytes, gib(3))
            self.assertEqual(result.memory.available_bytes, gib(9))
            self.assertEqual(result.memory.swap_limit_bytes, gib(4))
            self.assertEqual(result.memory.swap_current_bytes, gib(1))
            self.assertEqual(result.cpu.cpuset_count, 4)
            self.assertEqual(result.cpu.quota_cores, 1.8)
            self.assertEqual(result.cpu.quota_job_count, 1)
            self.assertEqual(result.cpu.effective_count, 1)

    def test_missing_cgroup_usage_assumes_no_headroom(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/broken")
            (leaf / "memory.max").write_text(str(gib(16)), encoding="ascii")
            (leaf / "memory.stat").write_text(
                f"inactive_file {gib(12)}\n", encoding="ascii"
            )
            # Keep the marker present but make it unreadable as an integer.
            (leaf / "memory.current").write_text("not-a-number\n", encoding="ascii")

            result = pr.collect_resources(
                filesystem_root=root,
                env={},
                affinity_count=4,
                host_cpu_count=4,
            )
            self.assertEqual(result.memory.current_bytes, gib(16))
            self.assertEqual(result.memory.available_bytes, 0)
            self.assertEqual(result.build.suggested_jobs, 1)
            self.assertTrue(
                any("cannot read current memory usage" in item for item in result.warnings)
            )

    def test_runpod_assignment_with_scoped_leaf_supplies_capacity(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pods/p1/container")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(4)), encoding="ascii")
            (leaf / "memory.stat").write_text(
                f"inactive_file {gib(1)}\n", encoding="ascii"
            )
            (leaf / "memory.peak").write_text(str(gib(5)), encoding="ascii")
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={
                    "RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32)),
                    "POD_BUILD_MAX_JOBS": "2",
                },
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertFalse(result.memory.limited)
            self.assertEqual(result.memory.limit_source, "host")
            self.assertEqual(result.memory.limit_bytes, gib(64))
            self.assertEqual(result.memory.capacity_bytes, gib(32))
            self.assertEqual(
                result.memory.capacity_source, "runpod-api-assignment"
            )
            self.assertFalse(result.memory.capacity_is_hard_limit)
            self.assertEqual(result.memory.assigned_capacity_bytes, gib(32))
            self.assertEqual(result.memory.usage_current_bytes, gib(4))
            self.assertEqual(result.memory.current_bytes, gib(4))
            self.assertEqual(result.memory.available_bytes, gib(29))
            self.assertTrue(result.memory.usage_trustworthy)
            self.assertTrue(result.memory.usage_peak_eligible)
            self.assertEqual(result.memory.usage_scope, "pod-cgroup")
            self.assertTrue(
                result.memory.usage_source.endswith(
                    "/sys/fs/cgroup/pods/p1/container"
                )
            )
            self.assertFalse(result.build.forced_single_job)
            self.assertEqual(result.build.suggested_jobs, 2)

    def test_runpod_assignment_accepts_ambiguous_namespaced_root_for_peak_only(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(4)), encoding="ascii")
            (leaf / "memory.stat").write_text(
                f"inactive_file {gib(1)}\n", encoding="ascii"
            )
            (leaf / "memory.peak").write_text(str(gib(5)), encoding="ascii")
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={
                    "RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32)),
                    "MAX_JOBS": "4",
                },
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertEqual(result.cgroup_path, "/")
            self.assertEqual(
                result.memory.capacity_source, "runpod-api-assignment"
            )
            self.assertFalse(result.memory.capacity_is_hard_limit)
            self.assertFalse(result.memory.usage_trustworthy)
            self.assertTrue(result.memory.usage_peak_eligible)
            self.assertEqual(result.memory.usage_scope, "ambiguous-cgroup-root")
            self.assertEqual(
                result.memory.usage_source, "cgroup-v2:/sys/fs/cgroup"
            )
            self.assertEqual(result.memory.usage_current_bytes, gib(4))
            self.assertEqual(result.memory.available_bytes, gib(28))
            self.assertTrue(result.build.forced_single_job)
            self.assertEqual(result.build.suggested_jobs, 1)
            self.assertEqual(result.build.max_jobs_cap, 1)
            self.assertTrue(
                any("both '/'" in warning for warning in result.warnings)
            )

    def test_ambiguous_root_still_exposes_insufficient_one_job_headroom(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(31)), encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")
            (leaf / "memory.peak").write_text(str(gib(31)), encoding="ascii")
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertTrue(result.memory.usage_peak_eligible)
            self.assertFalse(result.memory.usage_trustworthy)
            self.assertEqual(result.memory.available_bytes, gib(1))
            required_headroom = (
                result.build.reserve_bytes + result.build.memory_per_job_bytes
            )
            self.assertLess(result.memory.available_bytes, required_headroom)
            self.assertTrue(result.build.forced_single_job)
            self.assertEqual(result.build.suggested_jobs, 1)

    def test_verified_assignment_without_cgroup_counter_selects_rss_fallback(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            write(root, "/proc/meminfo", host_meminfo(128, 112))
            write(root, "/proc/self/cgroup", "")
            write(root, "/proc/self/mountinfo", "")
            write_verified_assignment(root, gib(64))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(64))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertEqual(
                result.memory.capacity_source, "runpod-api-assignment"
            )
            self.assertEqual(result.memory.capacity_bytes, gib(64))
            self.assertIsNone(result.memory.usage_current_bytes)
            self.assertEqual(result.memory.usage_source, "")
            self.assertEqual(result.memory.usage_scope, "unavailable")
            self.assertFalse(result.memory.usage_trustworthy)
            self.assertFalse(result.memory.usage_peak_eligible)
            self.assertEqual(
                result.memory.peak_evidence_mode, "process-group-rss"
            )
            self.assertTrue(result.build.forced_single_job)
            self.assertEqual(result.build.suggested_jobs, 1)

    def test_readable_usage_over_assignment_cannot_select_rss_fallback(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pod")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(33)), encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertEqual(result.memory.usage_current_bytes, gib(33))
            self.assertNotEqual(result.memory.usage_source, "")
            self.assertEqual(result.memory.peak_evidence_mode, "unavailable")
            self.assertFalse(result.memory.usage_peak_eligible)
            self.assertTrue(result.build.forced_single_job)

    def test_malformed_usage_counter_cannot_select_rss_fallback(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pod")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text("not-a-counter\n", encoding="ascii")
            write_verified_assignment(root, gib(64))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(64))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertIsNone(result.memory.usage_current_bytes)
            self.assertEqual(result.memory.peak_evidence_mode, "unavailable")
            self.assertTrue(
                any("counter is malformed" in warning for warning in result.warnings)
            )

    def test_smaller_finite_cgroup_limit_wins_over_assignment(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pod")
            (leaf / "memory.max").write_text(str(gib(16)), encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(2)), encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertEqual(result.memory.capacity_bytes, gib(16))
            self.assertTrue(result.memory.capacity_is_hard_limit)
            self.assertTrue(result.memory.capacity_source.startswith("cgroup-v2:"))
            self.assertEqual(result.memory.assigned_capacity_bytes, gib(32))
            self.assertTrue(result.memory.usage_trustworthy)

    def test_assignment_smaller_than_cgroup_uses_assignment_and_leaf_usage(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pod")
            (leaf / "memory.max").write_text(str(gib(48)), encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(4)), encoding="ascii")
            (leaf / "memory.stat").write_text(
                f"inactive_file {gib(1)}\n", encoding="ascii"
            )
            write_verified_assignment(root, gib(32))

            result = pr.collect_resources(
                filesystem_root=root,
                env={"RUNPOD_ASSIGNED_MEMORY_BYTES": str(gib(32))},
                affinity_count=8,
                host_cpu_count=8,
            )

            self.assertTrue(result.memory.limited)
            self.assertEqual(result.memory.limit_bytes, gib(48))
            self.assertEqual(
                result.memory.capacity_source, "runpod-api-assignment"
            )
            self.assertEqual(result.memory.capacity_bytes, gib(32))
            self.assertTrue(result.memory.usage_trustworthy)
            self.assertEqual(result.memory.available_bytes, gib(29))

    def test_invalid_runpod_assignment_is_rejected(self) -> None:
        invalid_values = ("0", "-1", "01", " 34359738368", "32GiB", str(gib(65)))
        for value in invalid_values:
            with self.subTest(value=value), self.fixture_root() as raw_root:
                root = Path(raw_root)
                leaf = setup_v2_leaf(root, "/pod")
                (leaf / "memory.max").write_text("max\n", encoding="ascii")
                (leaf / "memory.current").write_text("1\n", encoding="ascii")
                with self.assertRaises(ValueError):
                    pr.collect_resources(
                        filesystem_root=root,
                        env={"RUNPOD_ASSIGNED_MEMORY_BYTES": value},
                        affinity_count=4,
                        host_cpu_count=4,
                    )

    def test_assignment_requires_an_exact_verified_receipt(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/pod")
            (leaf / "memory.max").write_text("max\n", encoding="ascii")
            (leaf / "memory.current").write_text("1\n", encoding="ascii")
            assignment = str(gib(32))

            with self.assertRaisesRegex(ValueError, "receipt"):
                pr.collect_resources(
                    filesystem_root=root,
                    env={"RUNPOD_ASSIGNED_MEMORY_BYTES": assignment},
                    affinity_count=4,
                    host_cpu_count=4,
                )

            write_verified_assignment(root, gib(31))
            with self.assertRaisesRegex(ValueError, "does not match"):
                pr.collect_resources(
                    filesystem_root=root,
                    env={"RUNPOD_ASSIGNED_MEMORY_BYTES": assignment},
                    affinity_count=4,
                    host_cpu_count=4,
                )

    def test_conservative_sage_build_job_recommendation(self) -> None:
        cases = [
            (16, 16, 1),
            (32, 32, 3),
            (32, 25, 2),  # About 7 GiB is already occupied.
            (48, 48, 4),
            (64, 64, 4),
        ]
        for total_gib, available_gib, expected in cases:
            with self.subTest(total=total_gib, available=available_gib):
                current = gib(total_gib - available_gib)
                memory = pr.MemorySnapshot(
                    limited=True,
                    host_total_bytes=gib(128),
                    limit_bytes=gib(total_gib),
                    current_bytes=current,
                    working_set_bytes=current,
                    inactive_file_bytes=0,
                    free_bytes=gib(available_gib),
                    available_bytes=gib(available_gib),
                    high_bytes=None,
                    limit_source="fixture",
                    swap_limit_bytes=0,
                    swap_current_bytes=0,
                    usage_trustworthy=True,
                    usage_peak_eligible=True,
                    usage_scope="cgroup-capacity",
                )
                cpu = pr.CpuSnapshot(
                    host_count=32,
                    affinity_count=32,
                    cpuset_count=32,
                    quota_cores=None,
                    quota_job_count=None,
                    runpod_count=32,
                    effective_count=32,
                    limiting_sources=("host",),
                )
                recommendation = pr.recommend_build_jobs(memory, cpu, {})
                self.assertEqual(recommendation.suggested_jobs, expected)
                self.assertEqual(recommendation.memory_per_job_bytes, gib(8))
                self.assertEqual(
                    recommendation.reserve_bytes,
                    max(gib(4), (gib(total_gib) * 15 + 99) // 100),
                )
                self.assertEqual(recommendation.max_jobs_cap, 4)

    def test_explicit_build_caps_and_memory_assumptions_are_honored(self) -> None:
        memory = pr.MemorySnapshot(
            limited=True,
            host_total_bytes=gib(128),
            limit_bytes=gib(64),
            current_bytes=0,
            working_set_bytes=0,
            inactive_file_bytes=0,
            free_bytes=gib(64),
            available_bytes=gib(64),
            high_bytes=None,
            limit_source="fixture",
            swap_limit_bytes=0,
            swap_current_bytes=0,
            usage_trustworthy=True,
            usage_peak_eligible=True,
            usage_scope="cgroup-capacity",
        )
        cpu = pr.CpuSnapshot(32, 32, 32, None, None, 32, 32, ("host",))
        recommendation = pr.recommend_build_jobs(
            memory,
            cpu,
            {
                "MAX_JOBS": "2",
                "POD_BUILD_MEMORY_PER_JOB_MIB": "4096",
                "POD_BUILD_RESERVE_MIB": "0",
            },
        )
        self.assertEqual(recommendation.suggested_jobs, 2)
        self.assertEqual(recommendation.max_jobs_cap, 2)
        self.assertEqual(recommendation.memory_per_job_bytes, gib(4))
        self.assertEqual(recommendation.reserve_bytes, 0)

    def test_mountinfo_unescapes_paths_and_cpuset_parser(self) -> None:
        mounts = pr.parse_mountinfo(
            "29 23 0:26 /docker\\040id /sys/fs/cgroup\\040x rw "
            "- cgroup2 cgroup rw\n"
        )
        self.assertEqual(mounts[0].root, "/docker id")
        self.assertEqual(mounts[0].mount_point, "/sys/fs/cgroup x")
        self.assertEqual(pr.parse_cpuset("0-3,6,8-9"), 7)
        self.assertIsNone(pr.parse_cpuset(""))
        self.assertIsNone(pr.parse_cpuset("3-1"))

    def test_process_group_rss_sum_is_conservative_and_handles_spaced_comm(
        self,
    ) -> None:
        with self.fixture_root() as raw_root:
            proc_root = Path(raw_root)
            for pid, group, pages, command in (
                (101, 77, 10, "python worker"),
                (102, 77, 20, "compiler (phase)"),
                (103, 88, 99, "unrelated"),
            ):
                process = proc_root / str(pid)
                process.mkdir()
                (process / "stat").write_text(
                    f"{pid} ({command}) S 1 {group} 0 0\n", encoding="ascii"
                )
                (process / "statm").write_text(
                    f"100 {pages} 0 0 0 0 0\n", encoding="ascii"
                )

            rss_bytes, process_count = pr.collect_process_group_rss(
                77, proc_root=proc_root, page_size=4096
            )

            self.assertEqual(rss_bytes, 30 * 4096)
            self.assertEqual(process_count, 2)

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux /proc")
    def test_process_group_supervisor_binds_compiler_and_writes_evidence(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            output = root / "rss.json"
            compiler = root / "c++"
            shutil.copy2("/bin/sleep", compiler)
            compiler.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(pr.__file__).resolve()),
                    "--monitor-process-group",
                    str(output),
                    "--sample-interval-ms",
                    "100",
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "raise SystemExit(subprocess.run([sys.argv[1], '.5']).returncode)"
                    ),
                    str(compiler),
                ],
                check=False,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["method"], "proc-process-group-rss-sum")
            self.assertEqual(payload["scope"], "build-process-group")
            self.assertEqual(payload["source"], "/proc/*/statm")
            self.assertEqual(payload["leader_pid"], payload["process_group_id"])
            self.assertTrue(payload["leader_observed"])
            self.assertTrue(payload["observed_compiler_executable"])
            self.assertIn("c++", payload["observed_compiler_executables"])
            self.assertEqual(payload["child_returncode"], 0)
            self.assertEqual(payload["build_exit_status"], 0)
            self.assertIsNone(payload["forwarded_signal"])
            self.assertFalse(payload["forced_kill"])
            self.assertGreaterEqual(payload["sample_count"], 2)
            self.assertGreaterEqual(payload["maximum_process_count"], 2)
            self.assertGreater(payload["peak_bytes"], 0)
            self.assertGreater(payload["end_bytes"], 0)
            self.assertGreater(
                payload["finished_monotonic_ns"], payload["started_monotonic_ns"]
            )

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux /proc")
    def test_process_group_supervisor_returns_exact_child_failure(self) -> None:
        with self.fixture_root() as raw_root:
            for label, exit_status, program in (
                ("sampled", 7, "import time; time.sleep(.25); raise SystemExit(7)"),
                ("immediate", 23, "raise SystemExit(23)"),
            ):
                with self.subTest(label=label):
                    output = Path(raw_root) / f"{label}.json"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(pr.__file__).resolve()),
                            "--monitor-process-group",
                            str(output),
                            "--",
                            sys.executable,
                            "-c",
                            program,
                        ],
                        check=False,
                        timeout=5,
                    )
                    self.assertEqual(completed.returncode, exit_status)
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    self.assertTrue(payload["complete"])
                    self.assertEqual(payload["child_returncode"], exit_status)
                    self.assertEqual(payload["build_exit_status"], exit_status)

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux /proc")
    def test_process_group_supervisor_rejects_success_without_compiler(self) -> None:
        with self.fixture_root() as raw_root:
            output = Path(raw_root) / "rss.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(pr.__file__).resolve()),
                    "--monitor-process-group",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(.25)",
                ],
                check=False,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 70)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["child_returncode"], 0)
            self.assertFalse(payload["observed_compiler_executable"])
            self.assertIn("no compiler executable", payload["monitor_error"])

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux /proc")
    def test_process_group_supervisor_forwards_hup_int_and_term(self) -> None:
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signal.Signals(signum).name), self.fixture_root() as raw_root:
                output = Path(raw_root) / "rss.json"
                supervisor = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(pr.__file__).resolve()),
                        "--monitor-process-group",
                        str(output),
                        "--termination-grace-seconds",
                        "1",
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(30)",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                try:
                    for _attempt in range(100):
                        if output.is_file():
                            break
                        if supervisor.poll() is not None:
                            self.fail("supervisor exited before initial evidence")
                        time.sleep(0.02)
                    else:
                        self.fail("supervisor did not write initial evidence")
                    supervisor.send_signal(signum)
                    self.assertEqual(supervisor.wait(timeout=5), 128 + signum)
                finally:
                    if supervisor.poll() is None:
                        supervisor.kill()
                        supervisor.wait(timeout=5)

                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(payload["complete"])
                self.assertEqual(
                    payload["forwarded_signal"], signal.Signals(signum).name
                )
                self.assertFalse(payload["forced_kill"])
                with self.assertRaises(ProcessLookupError):
                    os.killpg(payload["process_group_id"], 0)

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux /proc")
    def test_process_group_supervisor_bounds_forced_shutdown(self) -> None:
        with self.fixture_root() as raw_root:
            output = Path(raw_root) / "rss.json"
            supervisor = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(pr.__file__).resolve()),
                    "--monitor-process-group",
                    str(output),
                    "--termination-grace-seconds",
                    "1",
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import signal,time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "time.sleep(30)"
                    ),
                ]
            )
            try:
                for _attempt in range(100):
                    if output.is_file():
                        break
                    if supervisor.poll() is not None:
                        self.fail("supervisor exited before initial evidence")
                    time.sleep(0.02)
                else:
                    self.fail("supervisor did not write initial evidence")
                time.sleep(0.15)
                started = time.monotonic()
                supervisor.terminate()
                self.assertEqual(supervisor.wait(timeout=5), 137)
                self.assertLess(time.monotonic() - started, 4)
            finally:
                if supervisor.poll() is None:
                    supervisor.kill()
                    supervisor.wait(timeout=5)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["forwarded_signal"], "SIGTERM")
            self.assertTrue(payload["forced_kill"])
            self.assertEqual(payload["child_returncode"], -signal.SIGKILL)
            self.assertEqual(payload["build_exit_status"], 137)
            with self.assertRaises(ProcessLookupError):
                os.killpg(payload["process_group_id"], 0)

    def test_json_and_shell_cli_outputs(self) -> None:
        with self.fixture_root() as raw_root:
            root = Path(raw_root)
            leaf = setup_v2_leaf(root, "/cli")
            (leaf / "memory.max").write_text(str(gib(16)), encoding="ascii")
            (leaf / "memory.current").write_text(str(gib(2)), encoding="ascii")
            (leaf / "memory.stat").write_text("inactive_file 0\n", encoding="ascii")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    pr.main(["--filesystem-root", str(root), "--json"]), 0
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["memory"]["limit_bytes"], gib(16))
            self.assertEqual(payload["memory"]["capacity_bytes"], gib(16))
            self.assertTrue(payload["memory"]["capacity_is_hard_limit"])

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    pr.main(["--filesystem-root", str(root), "--shell"]), 0
                )
            shell = output.getvalue()
            self.assertIn("POD_MEMORY_LIMIT_BYTES=17179869184", shell)
            self.assertIn("POD_MEMORY_CAPACITY_BYTES=17179869184", shell)
            self.assertIn("POD_MEMORY_USAGE_PEAK_ELIGIBLE=1", shell)
            self.assertIn("POD_MEMORY_PEAK_EVIDENCE_MODE=cgroup", shell)
            self.assertIn("POD_BUILD_JOBS=", shell)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    pr.main(["--filesystem-root", str(root), "--suggested-jobs"]),
                    0,
                )
            self.assertTrue(output.getvalue().strip().isdigit())


if __name__ == "__main__":
    unittest.main()
