#!/usr/bin/env python3
"""Report the resources effectively available to this container.

Linux deliberately leaves files such as ``/proc/meminfo`` host-scoped.  This
module resolves the current process' cgroup mount and membership instead, and
uses the tightest visible ancestor constraint.  It supports cgroup v2 and the
memory/cpu/cpuset controllers on cgroup v1.

The module has no third-party dependencies.  ``--filesystem-root`` exists so
the resolver can be tested against an ordinary directory tree without Docker,
root privileges, or a particular cgroup version.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_MEMORY_PER_JOB_MIB = 8192
DEFAULT_RESERVE_BYTES = 4 * GIB
DEFAULT_MAX_JOBS = 4
UNLIMITED_THRESHOLD = 1 << 60
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MountInfo:
    root: str
    mount_point: str
    fs_type: str
    mount_options: frozenset[str]
    super_options: frozenset[str]


@dataclass(frozen=True)
class CgroupEntry:
    hierarchy_id: str
    controllers: frozenset[str]
    path: str


@dataclass(frozen=True)
class CgroupLocation:
    version: int
    membership_path: str
    mount_point: str
    current_dir: Path
    mount_dir: Path


@dataclass(frozen=True)
class MemorySnapshot:
    limited: bool
    host_total_bytes: int
    limit_bytes: int
    current_bytes: int
    working_set_bytes: int
    inactive_file_bytes: int
    free_bytes: int
    available_bytes: int
    high_bytes: int | None
    limit_source: str
    swap_limit_bytes: int
    swap_current_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "limited": self.limited,
            "host_total_bytes": self.host_total_bytes,
            "limit_bytes": self.limit_bytes,
            "current_bytes": self.current_bytes,
            "working_set_bytes": self.working_set_bytes,
            "inactive_file_bytes": self.inactive_file_bytes,
            "free_bytes": self.free_bytes,
            "available_bytes": self.available_bytes,
            "high_bytes": self.high_bytes,
            "limit_source": self.limit_source,
            "swap_limit_bytes": self.swap_limit_bytes,
            "swap_current_bytes": self.swap_current_bytes,
        }


@dataclass(frozen=True)
class CpuSnapshot:
    host_count: int
    affinity_count: int | None
    cpuset_count: int | None
    quota_cores: float | None
    quota_job_count: int | None
    runpod_count: int | None
    effective_count: int
    limiting_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "host_count": self.host_count,
            "affinity_count": self.affinity_count,
            "cpuset_count": self.cpuset_count,
            "quota_cores": self.quota_cores,
            "quota_job_count": self.quota_job_count,
            "runpod_count": self.runpod_count,
            "effective_count": self.effective_count,
            "limiting_sources": list(self.limiting_sources),
        }


@dataclass(frozen=True)
class BuildRecommendation:
    suggested_jobs: int
    jobs_by_cpu: int
    jobs_by_memory: int
    memory_per_job_bytes: int
    reserve_bytes: int
    usable_memory_bytes: int
    max_jobs_cap: int

    def to_dict(self) -> dict[str, object]:
        return {
            "suggested_jobs": self.suggested_jobs,
            "jobs_by_cpu": self.jobs_by_cpu,
            "jobs_by_memory": self.jobs_by_memory,
            "memory_per_job_bytes": self.memory_per_job_bytes,
            "reserve_bytes": self.reserve_bytes,
            "usable_memory_bytes": self.usable_memory_bytes,
            "max_jobs_cap": self.max_jobs_cap,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    cgroup_version: int | None
    cgroup_path: str | None
    cgroup_mount_point: str | None
    memory: MemorySnapshot
    cpu: CpuSnapshot
    build: BuildRecommendation
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "cgroup": {
                "version": self.cgroup_version,
                "path": self.cgroup_path,
                "mount_point": self.cgroup_mount_point,
            },
            "memory": self.memory.to_dict(),
            "cpu": self.cpu.to_dict(),
            "build": self.build.to_dict(),
            "warnings": list(self.warnings),
        }

    def shell_assignments(self) -> dict[str, object]:
        return {
            "POD_CGROUP_VERSION": self.cgroup_version or 0,
            "POD_CGROUP_PATH": self.cgroup_path or "",
            "POD_CGROUP_MOUNT_POINT": self.cgroup_mount_point or "",
            "POD_MEMORY_LIMITED": int(self.memory.limited),
            "POD_MEMORY_HOST_TOTAL_BYTES": self.memory.host_total_bytes,
            "POD_MEMORY_LIMIT_BYTES": self.memory.limit_bytes,
            "POD_MEMORY_CURRENT_BYTES": self.memory.current_bytes,
            "POD_MEMORY_WORKING_SET_BYTES": self.memory.working_set_bytes,
            "POD_MEMORY_INACTIVE_FILE_BYTES": self.memory.inactive_file_bytes,
            "POD_MEMORY_FREE_BYTES": self.memory.free_bytes,
            "POD_MEMORY_AVAILABLE_BYTES": self.memory.available_bytes,
            "POD_MEMORY_HIGH_BYTES": self.memory.high_bytes or 0,
            "POD_MEMORY_SWAP_LIMIT_BYTES": self.memory.swap_limit_bytes,
            "POD_MEMORY_SWAP_CURRENT_BYTES": self.memory.swap_current_bytes,
            "POD_CPU_COUNT": self.cpu.effective_count,
            "POD_CPU_HOST_COUNT": self.cpu.host_count,
            "POD_CPU_AFFINITY_COUNT": self.cpu.affinity_count or 0,
            "POD_CPU_CPUSET_COUNT": self.cpu.cpuset_count or 0,
            "POD_CPU_QUOTA_CORES": (
                "" if self.cpu.quota_cores is None else f"{self.cpu.quota_cores:.6g}"
            ),
            "POD_CPU_QUOTA_JOB_COUNT": self.cpu.quota_job_count or 0,
            "POD_CPU_RUNPOD_COUNT": self.cpu.runpod_count or 0,
            "POD_CPU_LIMITING_SOURCES": ",".join(self.cpu.limiting_sources),
            "POD_BUILD_JOBS": self.build.suggested_jobs,
            "POD_BUILD_JOBS_BY_CPU": self.build.jobs_by_cpu,
            "POD_BUILD_JOBS_BY_MEMORY": self.build.jobs_by_memory,
            "POD_BUILD_MEMORY_PER_JOB_BYTES": self.build.memory_per_job_bytes,
            "POD_BUILD_RESERVE_BYTES": self.build.reserve_bytes,
            "POD_BUILD_USABLE_MEMORY_BYTES": self.build.usable_memory_bytes,
            "POD_BUILD_MAX_JOBS_CAP": self.build.max_jobs_cap,
        }


class Probe:
    """Read Linux absolute paths through an optional fixture root."""

    def __init__(self, filesystem_root: str | os.PathLike[str] = "/") -> None:
        self.filesystem_root = Path(filesystem_root)

    def path(self, absolute_path: str) -> Path:
        normalized = posixpath.normpath("/" + absolute_path.lstrip("/"))
        if self.filesystem_root == Path("/"):
            return Path(normalized)
        parts = [part for part in normalized.split("/") if part]
        return self.filesystem_root.joinpath(*parts)

    def read_text(self, absolute_path: str) -> str | None:
        try:
            return self.path(absolute_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None


def _unescape_mount_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def parse_mountinfo(text: str) -> list[MountInfo]:
    mounts: list[MountInfo] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            continue
        mounts.append(
            MountInfo(
                root=_unescape_mount_field(left_fields[3]),
                mount_point=_unescape_mount_field(left_fields[4]),
                fs_type=right_fields[0],
                mount_options=frozenset(left_fields[5].split(",")),
                super_options=frozenset(right_fields[2].split(",")),
            )
        )
    return mounts


def parse_cgroup_memberships(text: str) -> list[CgroupEntry]:
    entries: list[CgroupEntry] = []
    for line in text.splitlines():
        fields = line.strip().split(":", 2)
        if len(fields) != 3:
            continue
        path = posixpath.normpath("/" + fields[2].lstrip("/"))
        entries.append(
            CgroupEntry(
                hierarchy_id=fields[0],
                controllers=frozenset(filter(None, fields[1].split(","))),
                path=path,
            )
        )
    return entries


def _mapped_cgroup_paths(mount: MountInfo, membership_path: str) -> list[str]:
    root = posixpath.normpath("/" + mount.root.lstrip("/"))
    membership = posixpath.normpath("/" + membership_path.lstrip("/"))
    rels: list[str] = []

    # A private cgroup namespace commonly exposes the container cgroup as '/'.
    if membership == "/":
        rels.append("")

    if root == "/":
        rels.append(membership.lstrip("/"))
    elif membership == root:
        rels.append("")
    elif membership.startswith(root.rstrip("/") + "/"):
        rels.append(membership[len(root) :].lstrip("/"))
    else:
        # In a cgroup namespace the membership can be relative to a hidden
        # mount root, so trying it relative to the visible mount is correct.
        rels.append(membership.lstrip("/"))

    results: list[str] = []
    for rel in rels:
        candidate = posixpath.normpath(posixpath.join(mount.mount_point, rel))
        if candidate not in results:
            results.append(candidate)
    return results


def _mount_has_controller(mount: MountInfo, controller: str) -> bool:
    return mount.fs_type == "cgroup" and (
        controller in mount.super_options
        or controller in mount.mount_options
        or controller in posixpath.basename(mount.mount_point).split(",")
    )


def locate_cgroup(
    probe: Probe,
    mounts: Sequence[MountInfo],
    memberships: Sequence[CgroupEntry],
    controller: str,
) -> CgroupLocation | None:
    v2_entries = [entry for entry in memberships if not entry.controllers]
    v2_mounts = [mount for mount in mounts if mount.fs_type == "cgroup2"]
    marker_names = {
        "memory": ("memory.current", "memory.max"),
        "cpu": ("cpu.max", "cpu.stat"),
        "cpuset": ("cpuset.cpus.effective", "cpuset.cpus"),
    }.get(controller, ())

    for entry in v2_entries:
        for mount in v2_mounts:
            candidates = _mapped_cgroup_paths(mount, entry.path)
            for logical in candidates:
                physical = probe.path(logical)
                if any((physical / marker).exists() for marker in marker_names):
                    return CgroupLocation(
                        version=2,
                        membership_path=entry.path,
                        mount_point=mount.mount_point,
                        current_dir=physical,
                        mount_dir=probe.path(mount.mount_point),
                    )

    v1_entries = [entry for entry in memberships if controller in entry.controllers]
    v1_mounts = [mount for mount in mounts if _mount_has_controller(mount, controller)]
    v1_markers = {
        "memory": ("memory.usage_in_bytes", "memory.limit_in_bytes"),
        "cpu": ("cpu.cfs_quota_us", "cpu.cfs_period_us"),
        "cpuset": ("cpuset.effective_cpus", "cpuset.cpus"),
    }.get(controller, ())
    for entry in v1_entries:
        for mount in v1_mounts:
            candidates = _mapped_cgroup_paths(mount, entry.path)
            for logical in candidates:
                physical = probe.path(logical)
                if any((physical / marker).exists() for marker in v1_markers):
                    return CgroupLocation(
                        version=1,
                        membership_path=entry.path,
                        mount_point=mount.mount_point,
                        current_dir=physical,
                        mount_dir=probe.path(mount.mount_point),
                    )
    return None


def _walk_ancestors(current: Path, mount_root: Path) -> Iterable[Path]:
    try:
        current.relative_to(mount_root)
    except ValueError:
        yield current
        return

    cursor = current
    while True:
        yield cursor
        if cursor == mount_root:
            return
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return int(value, 10)
    except (OSError, UnicodeError, ValueError):
        return None


def _read_limit(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError):
        return None
    if value in {"", "max", "-1"}:
        return None
    try:
        parsed = int(value, 10)
    except ValueError:
        return None
    if parsed < 0 or parsed >= UNLIMITED_THRESHOLD:
        return None
    return parsed


def _parse_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return values
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values[fields[0]] = int(fields[1], 10)
        except ValueError:
            continue
    return values


def _parse_meminfo(text: str | None) -> dict[str, int]:
    values: dict[str, int] = {}
    if not text:
        return values
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        if not fields:
            continue
        try:
            value = int(fields[0], 10)
        except ValueError:
            continue
        # Linux meminfo values are kB. Preserve correctness for a hypothetical
        # unit-less fixture rather than silently multiplying an unknown unit.
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024
        values[key] = value
    return values


def _host_memory(meminfo: Mapping[str, int]) -> tuple[int, int, int, int]:
    total = max(0, meminfo.get("MemTotal", 0))
    free = max(0, meminfo.get("MemFree", 0))
    cache = max(0, meminfo.get("Cached", 0)) + max(
        0, meminfo.get("SReclaimable", 0)
    )
    available = meminfo.get("MemAvailable")
    if available is None:
        available = free + cache + max(0, meminfo.get("Buffers", 0))
    available = min(total, max(0, available)) if total else max(0, available)
    current = max(0, total - available)
    return total, current, free, available


def _find_effective_limit(
    location: CgroupLocation,
    filename: str,
    host_limit: int,
) -> tuple[int, Path | None]:
    effective: int | None = host_limit if host_limit > 0 else None
    source: Path | None = None
    for directory in _walk_ancestors(location.current_dir, location.mount_dir):
        candidate = _read_limit(directory / filename)
        if candidate is None:
            continue
        # On equality keep walking outward and select the highest visible
        # ancestor. Its usage includes the complete constrained subtree. An
        # equality with host RAM is still an explicit finite cgroup limit.
        if effective is None or candidate <= effective:
            effective = candidate
            source = directory
    return effective if effective is not None else 0, source


def _logical_source(location: CgroupLocation, source: Path | None) -> str:
    if source is None:
        return "host"
    try:
        rel = source.relative_to(location.mount_dir).as_posix()
    except ValueError:
        rel = ""
    logical = location.mount_point
    if rel and rel != ".":
        logical = posixpath.join(logical, rel)
    return f"cgroup-v{location.version}:{logical}"


def collect_memory(
    probe: Probe,
    mounts: Sequence[MountInfo],
    memberships: Sequence[CgroupEntry],
    warnings: list[str],
) -> tuple[MemorySnapshot, CgroupLocation | None]:
    meminfo = _parse_meminfo(probe.read_text("/proc/meminfo"))
    host_total, host_current, host_free, host_available = _host_memory(meminfo)
    host_swap_total = max(0, meminfo.get("SwapTotal", 0))
    host_swap_free = max(0, meminfo.get("SwapFree", 0))
    host_swap_current = max(0, host_swap_total - host_swap_free)

    location = locate_cgroup(probe, mounts, memberships, "memory")
    if location is None:
        if host_total == 0:
            warnings.append("unable to read host memory or locate a memory cgroup")
        return (
            MemorySnapshot(
                limited=False,
                host_total_bytes=host_total,
                limit_bytes=host_total,
                current_bytes=host_current,
                working_set_bytes=host_current,
                inactive_file_bytes=0,
                free_bytes=host_free,
                available_bytes=host_available,
                high_bytes=None,
                limit_source="host",
                swap_limit_bytes=host_swap_total,
                swap_current_bytes=host_swap_current,
            ),
            None,
        )

    limit_name = "memory.max" if location.version == 2 else "memory.limit_in_bytes"
    usage_name = (
        "memory.current" if location.version == 2 else "memory.usage_in_bytes"
    )
    effective, source = _find_effective_limit(location, limit_name, host_total)

    # cgroup v1 exposes the effective inherited bound in memory.stat even if
    # the actual limiting parent is hidden by a cgroup namespace.
    if location.version == 1:
        leaf_stats = _parse_key_values(location.current_dir / "memory.stat")
        hierarchical = leaf_stats.get("hierarchical_memory_limit")
        if (
            hierarchical is not None
            and 0 < hierarchical < UNLIMITED_THRESHOLD
            and (effective == 0 or hierarchical <= effective)
        ):
            effective = hierarchical
            source = location.current_dir

    limited = source is not None
    if not limited:
        return (
            MemorySnapshot(
                limited=False,
                host_total_bytes=host_total,
                limit_bytes=host_total,
                current_bytes=host_current,
                working_set_bytes=host_current,
                inactive_file_bytes=0,
                free_bytes=host_free,
                available_bytes=host_available,
                high_bytes=None,
                limit_source="host",
                swap_limit_bytes=host_swap_total,
                swap_current_bytes=host_swap_current,
            ),
            location,
        )

    assert source is not None
    current = _read_integer(source / usage_name)
    if current is not None and current < 0:
        current = None
    current_known = current is not None
    if not current_known:
        # Do not turn a missing counter into large apparent headroom. A single
        # job is the safe recommendation until the runtime exposes accounting.
        warnings.append(f"cannot read current memory usage from {source / usage_name}")
        current = effective

    stats = _parse_key_values(source / "memory.stat") if current_known else {}
    inactive_key = "inactive_file" if location.version == 2 else "total_inactive_file"
    inactive = stats.get(inactive_key)
    if inactive is None and location.version == 1:
        inactive = stats.get("inactive_file", 0)
    inactive = min(max(0, inactive or 0), max(0, current))
    working = max(0, current - inactive)
    free = max(0, effective - current)
    available = max(0, effective - working)

    high_name = "memory.high" if location.version == 2 else "memory.soft_limit_in_bytes"
    high, high_source = _find_effective_limit(location, high_name, host_total)
    high_value = high if high_source is not None and high < effective else None

    swap_limit = host_swap_total
    swap_current = host_swap_current
    if location.version == 2 and host_swap_total > 0:
        cgroup_swap_limit, swap_source = _find_effective_limit(
            location, "memory.swap.max", host_swap_total
        )
        if swap_source is not None:
            swap_limit = cgroup_swap_limit
            swap_current = _read_integer(swap_source / "memory.swap.current") or 0
    elif location.version == 1 and host_swap_total > 0:
        memsw_limit, memsw_source = _find_effective_limit(
            location, "memory.memsw.limit_in_bytes", 0
        )
        if memsw_source is not None and memsw_limit >= effective:
            swap_limit = min(host_swap_total, max(0, memsw_limit - effective))
            memsw_current = _read_integer(
                memsw_source / "memory.memsw.usage_in_bytes"
            )
            memory_current = _read_integer(memsw_source / usage_name)
            if memsw_current is not None and memory_current is not None:
                swap_current = max(0, memsw_current - memory_current)

    swap_current = min(max(0, swap_current), max(0, swap_limit))
    return (
        MemorySnapshot(
            limited=True,
            host_total_bytes=host_total,
            limit_bytes=effective,
            current_bytes=max(0, current),
            working_set_bytes=working,
            inactive_file_bytes=inactive,
            free_bytes=free,
            available_bytes=available,
            high_bytes=high_value,
            limit_source=_logical_source(location, source),
            swap_limit_bytes=max(0, swap_limit),
            swap_current_bytes=swap_current,
        ),
        location,
    )


def parse_cpuset(value: str) -> int | None:
    cpus: set[int] = set()
    try:
        for item in value.strip().split(","):
            if not item:
                continue
            if "-" in item:
                first_raw, last_raw = item.split("-", 1)
                first, last = int(first_raw), int(last_raw)
                if first < 0 or last < first or last - first > 1_000_000:
                    return None
                cpus.update(range(first, last + 1))
            else:
                cpu = int(item)
                if cpu < 0:
                    return None
                cpus.add(cpu)
    except ValueError:
        return None
    return len(cpus) or None


def _read_cpuset(location: CgroupLocation) -> int | None:
    filenames = (
        ("cpuset.cpus.effective", "cpuset.cpus")
        if location.version == 2
        else ("cpuset.effective_cpus", "cpuset.cpus")
    )
    counts: list[int] = []
    for directory in _walk_ancestors(location.current_dir, location.mount_dir):
        for filename in filenames:
            try:
                content = (directory / filename).read_text(encoding="ascii").strip()
            except (OSError, UnicodeError):
                continue
            count = parse_cpuset(content)
            if count is not None:
                counts.append(count)
    # Configured child cpusets are subsets of their ancestors. Taking the
    # smallest visible count also gives a safe fallback on kernels that do not
    # expose an effective-cpus file on cgroup v1.
    return min(counts) if counts else None


def _read_cpu_quota(location: CgroupLocation) -> Fraction | None:
    effective: Fraction | None = None
    for directory in _walk_ancestors(location.current_dir, location.mount_dir):
        if location.version == 2:
            try:
                fields = (directory / "cpu.max").read_text(encoding="ascii").split()
            except (OSError, UnicodeError):
                continue
            if len(fields) != 2 or fields[0] == "max":
                continue
            try:
                quota, period = int(fields[0]), int(fields[1])
            except ValueError:
                continue
        else:
            quota = _read_integer(directory / "cpu.cfs_quota_us") or -1
            period = _read_integer(directory / "cpu.cfs_period_us") or 0
        if quota <= 0 or period <= 0:
            continue
        candidate = Fraction(quota, period)
        if effective is None or candidate < effective:
            effective = candidate
    return effective


def _positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip(), 10)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _system_affinity_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        return len(get_affinity(0)) or None
    except OSError:
        return None


def collect_cpu(
    probe: Probe,
    mounts: Sequence[MountInfo],
    memberships: Sequence[CgroupEntry],
    env: Mapping[str, str],
    *,
    affinity_count: int | None = None,
    host_cpu_count: int | None = None,
) -> CpuSnapshot:
    host_count = max(1, host_cpu_count or os.cpu_count() or 1)
    if affinity_count is None:
        affinity_count = _system_affinity_count()
    if affinity_count is not None and affinity_count <= 0:
        affinity_count = None

    cpuset_location = locate_cgroup(probe, mounts, memberships, "cpuset")
    cpuset_count = _read_cpuset(cpuset_location) if cpuset_location else None

    cpu_location = locate_cgroup(probe, mounts, memberships, "cpu")
    quota = _read_cpu_quota(cpu_location) if cpu_location else None
    quota_jobs = None
    if quota is not None:
        # Floor fractional quotas for compiler parallelism. A 1.8 CPU quota is
        # a useful two-thread runtime hint, but one concurrent compiler is the
        # conservative choice requested by this tool.
        quota_jobs = max(1, quota.numerator // quota.denominator)

    runpod_count = _positive_int(env.get("RUNPOD_CPU_COUNT"))
    candidates: list[tuple[str, int]] = [("host", host_count)]
    if affinity_count is not None:
        candidates.append(("affinity", affinity_count))
    if cpuset_count is not None:
        candidates.append(("cpuset", cpuset_count))
    if quota_jobs is not None:
        candidates.append(("quota", quota_jobs))
    if runpod_count is not None:
        candidates.append(("runpod", runpod_count))
    effective = max(1, min(value for _, value in candidates))
    sources = tuple(name for name, value in candidates if value == effective)

    return CpuSnapshot(
        host_count=host_count,
        affinity_count=affinity_count,
        cpuset_count=cpuset_count,
        quota_cores=(round(float(quota), 6) if quota is not None else None),
        quota_job_count=quota_jobs,
        runpod_count=runpod_count,
        effective_count=effective,
        limiting_sources=sources,
    )


def _nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip(), 10)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def recommend_build_jobs(
    memory: MemorySnapshot,
    cpu: CpuSnapshot,
    env: Mapping[str, str],
    *,
    memory_per_job_mib: int | None = None,
    reserve_mib: int | None = None,
) -> BuildRecommendation:
    configured_per_job = memory_per_job_mib
    if configured_per_job is None:
        configured_per_job = _positive_int(env.get("POD_BUILD_MEMORY_PER_JOB_MIB"))
    if configured_per_job is None:
        configured_per_job = DEFAULT_MEMORY_PER_JOB_MIB
    memory_per_job = max(1, configured_per_job) * MIB

    configured_reserve = reserve_mib
    if configured_reserve is None:
        configured_reserve = _nonnegative_int(env.get("POD_BUILD_RESERVE_MIB"))
    if configured_reserve is None:
        # Keep both a fixed floor and proportional headroom for the Python
        # process, linker, filesystem cache, and non-build services.
        configured_reserve_bytes = max(
            DEFAULT_RESERVE_BYTES,
            (memory.limit_bytes * 15 + 99) // 100,
        )
    else:
        configured_reserve_bytes = configured_reserve * MIB

    usable = max(0, memory.available_bytes - configured_reserve_bytes)
    jobs_by_memory = max(1, usable // memory_per_job)
    jobs_by_cpu = max(1, cpu.effective_count)

    max_jobs_cap = _positive_int(env.get("POD_BUILD_MAX_JOBS"))
    if max_jobs_cap is None:
        max_jobs_cap = _positive_int(env.get("MAX_JOBS"))
    if max_jobs_cap is None:
        max_jobs_cap = DEFAULT_MAX_JOBS
    candidates = [jobs_by_cpu, jobs_by_memory, max_jobs_cap]
    suggested = max(1, min(candidates))

    return BuildRecommendation(
        suggested_jobs=suggested,
        jobs_by_cpu=jobs_by_cpu,
        jobs_by_memory=jobs_by_memory,
        memory_per_job_bytes=memory_per_job,
        reserve_bytes=configured_reserve_bytes,
        usable_memory_bytes=usable,
        max_jobs_cap=max_jobs_cap,
    )


def collect_resources(
    *,
    filesystem_root: str | os.PathLike[str] = "/",
    env: Mapping[str, str] | None = None,
    affinity_count: int | None = None,
    host_cpu_count: int | None = None,
    memory_per_job_mib: int | None = None,
    reserve_mib: int | None = None,
) -> ResourceSnapshot:
    probe = Probe(filesystem_root)
    effective_env = dict(os.environ if env is None else env)
    warnings: list[str] = []

    mountinfo_text = probe.read_text("/proc/self/mountinfo") or ""
    cgroup_text = probe.read_text("/proc/self/cgroup") or ""
    mounts = parse_mountinfo(mountinfo_text)
    memberships = parse_cgroup_memberships(cgroup_text)
    if not mountinfo_text:
        warnings.append("cannot read /proc/self/mountinfo")
    if not cgroup_text:
        warnings.append("cannot read /proc/self/cgroup")

    memory, memory_location = collect_memory(
        probe, mounts, memberships, warnings
    )
    cpu = collect_cpu(
        probe,
        mounts,
        memberships,
        effective_env,
        affinity_count=affinity_count,
        host_cpu_count=host_cpu_count,
    )
    build = recommend_build_jobs(
        memory,
        cpu,
        effective_env,
        memory_per_job_mib=memory_per_job_mib,
        reserve_mib=reserve_mib,
    )

    location = memory_location
    if location is None:
        location = locate_cgroup(probe, mounts, memberships, "cpu")
    return ResourceSnapshot(
        cgroup_version=location.version if location else None,
        cgroup_path=location.membership_path if location else None,
        cgroup_mount_point=location.mount_point if location else None,
        memory=memory,
        cpu=cpu,
        build=build,
        warnings=tuple(warnings),
    )


def _shell_output(snapshot: ResourceSnapshot) -> str:
    lines = []
    for name, value in snapshot.shell_assignments().items():
        lines.append(f"{name}={shlex.quote(str(value))}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report cgroup-aware memory, CPU, and safe build parallelism."
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="emit JSON (default)")
    output.add_argument("--shell", action="store_true", help="emit sourceable shell assignments")
    output.add_argument(
        "--suggested-jobs",
        action="store_true",
        help="print only the conservative build job count",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    parser.add_argument(
        "--filesystem-root",
        default="/",
        metavar="PATH",
        help="read /proc and cgroup mounts below PATH (primarily for tests)",
    )
    parser.add_argument(
        "--memory-per-job-mib",
        type=int,
        help=f"assumed compiler memory per job (default {DEFAULT_MEMORY_PER_JOB_MIB} MiB)",
    )
    parser.add_argument(
        "--reserve-mib",
        type=int,
        help="memory kept outside compiler jobs (default max(4 GiB, 15%%))",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.memory_per_job_mib is not None and args.memory_per_job_mib <= 0:
        raise SystemExit("--memory-per-job-mib must be positive")
    if args.reserve_mib is not None and args.reserve_mib < 0:
        raise SystemExit("--reserve-mib cannot be negative")

    snapshot = collect_resources(
        filesystem_root=args.filesystem_root,
        memory_per_job_mib=args.memory_per_job_mib,
        reserve_mib=args.reserve_mib,
    )
    if args.suggested_jobs:
        print(snapshot.build.suggested_jobs)
    elif args.shell:
        print(_shell_output(snapshot))
    else:
        print(
            json.dumps(
                snapshot.to_dict(),
                indent=2 if args.pretty else None,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
