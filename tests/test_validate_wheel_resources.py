from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sageattention_validate_wheel_resources",
    ROOT / "scripts" / "validate-wheel.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)

GIB = 1024 ** 3
RESOURCES = {
    "minimum_cpus": 4,
    "minimum_memory_gib": 32,
    "recommended_memory_gib": 64,
}


def resource_evidence(*, assignment_capacity: bool = False) -> dict:
    assigned = 64_000_000_000
    capacity = assigned if assignment_capacity else 40 * GIB
    capacity_source = (
        "runpod-api-assignment"
        if assignment_capacity
        else "cgroup-v2:/sys/fs/cgroup/pod/memory.max"
    )
    usage_source = "cgroup-v2:/sys/fs/cgroup/pod"
    usage_scope = "pod-cgroup" if assignment_capacity else "cgroup-capacity"
    memory = {
        "assigned_capacity_bytes": assigned,
        "capacity_bytes": capacity,
        "capacity_is_hard_limit": not assignment_capacity,
        "capacity_source": capacity_source,
        "limited": not assignment_capacity,
        "limit_bytes": capacity if not assignment_capacity else 128 * GIB,
        "peak_evidence_mode": "cgroup",
        "usage_current_bytes": 2 * GIB,
        "usage_peak_eligible": True,
        "usage_scope": usage_scope,
        "usage_source": usage_source,
        "usage_trustworthy": True,
    }
    snapshot = {
        "schema_version": 2,
        "cgroup": {"version": 2},
        "memory": memory,
        "cpu": {"effective_count": 4, "runpod_count": 8},
        "build": {
            "forced_single_job": False,
            "max_jobs_cap": 2,
            "memory_per_job_bytes": 8 * GIB,
            "reserve_bytes": 6 * GIB,
            "suggested_jobs": 2,
        },
    }
    memory_policy = {
        "assigned_capacity_bytes": memory["assigned_capacity_bytes"],
        "capacity_bytes": memory["capacity_bytes"],
        "capacity_is_hard_limit": memory["capacity_is_hard_limit"],
        "capacity_source": memory["capacity_source"],
        "forced_single_job": False,
        "peak_evidence_mode": memory["peak_evidence_mode"],
        "usage_peak_eligible": memory["usage_peak_eligible"],
        "usage_scope": memory["usage_scope"],
        "usage_source": memory["usage_source"],
        "usage_trustworthy": memory["usage_trustworthy"],
    }
    return {
        "memory_policy": memory_policy,
        "resource_start": deepcopy(snapshot),
        "resource_end": deepcopy(snapshot),
        "runpod_assignment": {
            "memory_bytes": assigned,
            "vcpu_count": 8,
        },
        "cgroup_peak": {
            "available": True,
            "start_bytes": 2 * GIB,
            "end_bytes": 8 * GIB,
            "monotonic": True,
            "scope": usage_scope,
            "source": usage_source,
            "usage_trustworthy": True,
            "within_capacity": True,
        },
        "memory_peak": {
            "available": True,
            "complete": True,
            "end_bytes": 8 * GIB,
            "includes_file_cache": True,
            "kernel_enforced": not assignment_capacity,
            "method": "kernel-cgroup-peak",
            "mode": "cgroup",
            "peak_bytes": 8 * GIB,
            "sample_interval_ms": None,
            "scope": usage_scope,
            "source": usage_source,
            "start_bytes": 2 * GIB,
            "within_selected_capacity": True,
        },
        "selected_parallelism": {
            "extension_parallelism": 1,
            "low_resource_override": False,
            "max_jobs": 2,
            "unsafe_override": False,
        },
    }


def rss_fallback_evidence() -> dict:
    assigned = 80 * GIB
    reserve = 12 * GIB
    peak = 20 * GIB
    memory = {
        "assigned_capacity_bytes": assigned,
        "capacity_bytes": assigned,
        "capacity_is_hard_limit": False,
        "capacity_source": "runpod-api-assignment",
        "limited": False,
        "limit_bytes": 128 * GIB,
        "peak_evidence_mode": "process-group-rss",
        "usage_current_bytes": None,
        "usage_peak_eligible": False,
        "usage_scope": "unavailable",
        "usage_source": "",
        "usage_trustworthy": False,
    }
    build = {
        "forced_single_job": True,
        "max_jobs_cap": 1,
        "memory_per_job_bytes": 8 * GIB,
        "reserve_bytes": reserve,
        "suggested_jobs": 1,
    }
    snapshot = {
        "schema_version": 2,
        "cgroup": {"version": None},
        "memory": memory,
        "cpu": {"effective_count": 4, "runpod_count": 8},
        "build": build,
    }
    return {
        "memory_policy": {
            "assigned_capacity_bytes": assigned,
            "capacity_bytes": assigned,
            "capacity_is_hard_limit": False,
            "capacity_source": "runpod-api-assignment",
            "forced_single_job": True,
            "peak_evidence_mode": "process-group-rss",
            "usage_peak_eligible": False,
            "usage_scope": "unavailable",
            "usage_source": "",
            "usage_trustworthy": False,
        },
        "resource_start": deepcopy(snapshot),
        "resource_end": deepcopy(snapshot),
        "runpod_assignment": {"memory_bytes": assigned, "vcpu_count": 8},
        "cgroup_peak": {
            "available": False,
            "end_bytes": None,
            "monotonic": None,
            "scope": "unavailable",
            "source": "",
            "start_bytes": None,
            "usage_trustworthy": False,
            "within_capacity": None,
        },
        "memory_peak": {
            "available": True,
            "build_exit_status": 0,
            "child_returncode": 0,
            "checkpoint_interval_ms": 5000,
            "command_sha256": "a" * 64,
            "complete": True,
            "covers_other_pod_processes": False,
            "duration_ms": 2000,
            "end_bytes": 18 * GIB,
            "finished_monotonic_ns": 3_000_000_000,
            "forced_kill": False,
            "forwarded_signal": None,
            "includes_file_cache": False,
            "kernel_enforced": False,
            "leader_observed": True,
            "leader_pid": 4242,
            "maximum_process_count": 4,
            "method": "proc-process-group-rss-sum",
            "mode": "process-group-rss",
            "monitor_error": None,
            "observed_compiler_executable": True,
            "observed_compiler_executables": ["c++", "nvcc"],
            "peak_bytes": peak,
            "process_group_id": 4242,
            "sample_count": 100,
            "sample_interval_ms": 100,
            "sampled_peak_plus_reserve_within_assignment": True,
            "scope": "build-process-group",
            "shared_pages_may_be_double_counted": True,
            "source": "/proc/*/statm",
            "start_bytes": GIB,
            "started_monotonic_ns": 1_000_000_000,
            "termination_grace_seconds": 10,
            "whole_pod_enforced": False,
        },
        "selected_parallelism": {
            "extension_parallelism": 1,
            "low_resource_override": False,
            "max_jobs": 1,
            "unsafe_override": False,
        },
    }


class ResourcePromotionTests(unittest.TestCase):
    def assert_rejected(self, evidence: dict, message: str) -> None:
        with self.assertRaisesRegex(VALIDATOR.ValidationError, message):
            VALIDATOR.validate_resource_evidence(evidence, RESOURCES)

    def test_accepts_bound_cgroup_and_assignment_capacity_policies(self) -> None:
        for assignment_capacity in (False, True):
            with self.subTest(assignment_capacity=assignment_capacity):
                VALIDATOR.validate_resource_evidence(
                    resource_evidence(assignment_capacity=assignment_capacity),
                    RESOURCES,
                )

    def test_builder_emits_verified_runpod_assignment_receipt(self) -> None:
        build_script = (ROOT / "scripts" / "build-wheel.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"runpod_assignment": {', build_script)
        self.assertIn(
            '"memory_bytes": start_memory["assigned_capacity_bytes"]',
            build_script,
        )
        self.assertIn(
            '"vcpu_count": resource_start["cpu"]["runpod_count"]',
            build_script,
        )

    def test_requires_policy_and_schema_two_snapshots(self) -> None:
        evidence = resource_evidence()
        del evidence["memory_policy"]
        self.assert_rejected(evidence, "no memory_policy")

        for name in ("resource_start", "resource_end"):
            evidence = resource_evidence()
            evidence[name]["schema_version"] = 1
            self.assert_rejected(evidence, "schema version 2")

    def test_rejects_missing_mutated_or_lost_assignment(self) -> None:
        for stage in ("resource_start", "resource_end"):
            evidence = resource_evidence()
            evidence[stage]["memory"]["assigned_capacity_bytes"] = None
            self.assert_rejected(evidence, "lost the verified Runpod memory assignment")

        evidence = resource_evidence()
        evidence["resource_end"]["memory"]["assigned_capacity_bytes"] -= 1
        self.assert_rejected(evidence, "resource assignment changed")

    def test_binds_verified_runpod_vcpu_assignment(self) -> None:
        for stage in ("resource_start", "resource_end"):
            evidence = resource_evidence()
            evidence[stage]["cpu"]["runpod_count"] = None
            self.assert_rejected(evidence, "lost the verified Runpod vCPU assignment")

        evidence = resource_evidence()
        evidence["resource_end"]["cpu"]["runpod_count"] = 16
        self.assert_rejected(evidence, "vCPU assignment changed during build")

        evidence = resource_evidence()
        evidence["runpod_assignment"]["vcpu_count"] = 16
        self.assert_rejected(evidence, "runpod_assignment does not match")

        evidence = resource_evidence()
        evidence["runpod_assignment"]["memory_bytes"] -= 1
        self.assert_rejected(evidence, "runpod_assignment does not match")

        evidence = resource_evidence()
        del evidence["runpod_assignment"]
        self.assert_rejected(evidence, "runpod_assignment does not match")

        evidence = resource_evidence()
        evidence["resource_start"]["cpu"]["effective_count"] = 9
        self.assert_rejected(evidence, "effective CPU exceeds the verified assignment")

    def test_binds_start_and_end_memory_policy_fields(self) -> None:
        mutations = {
            "capacity_bytes": 41 * GIB,
            "capacity_source": "cgroup-v2:/sys/fs/cgroup/other/memory.max",
            "usage_source": "cgroup-v2:/sys/fs/cgroup/other",
            "usage_scope": "pod-cgroup",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                evidence = resource_evidence()
                evidence["resource_end"]["memory"][field] = value
                if field == "capacity_bytes":
                    evidence["resource_end"]["memory"]["limit_bytes"] = value
                self.assert_rejected(evidence, "resource assignment changed")

        evidence = resource_evidence()
        evidence["resource_end"]["memory"]["capacity_is_hard_limit"] = False
        self.assert_rejected(evidence, "capacity policy is inconsistent")

        evidence = resource_evidence()
        evidence["resource_end"]["memory"]["usage_peak_eligible"] = False
        self.assert_rejected(evidence, "not eligible for cgroup peak evidence")

    def test_cgroup_capacity_and_usage_scope_are_fail_closed(self) -> None:
        for field, value in (("limited", False), ("limit_bytes", 41 * GIB)):
            with self.subTest(field=field):
                evidence = resource_evidence()
                evidence["resource_start"]["memory"][field] = value
                self.assert_rejected(evidence, "cgroup capacity policy is inconsistent")

        evidence = resource_evidence()
        evidence["resource_start"]["memory"]["usage_scope"] = "host"
        self.assert_rejected(evidence, "peak usage scope is unsupported")

        evidence = resource_evidence()
        evidence["resource_start"]["memory"]["usage_trustworthy"] = False
        self.assert_rejected(evidence, "untrusted usage is not safely constrained")

    def test_untrusted_usage_forces_single_job_in_policy_and_selection(self) -> None:
        evidence = resource_evidence(assignment_capacity=True)
        for stage in ("resource_start", "resource_end"):
            evidence[stage]["memory"]["usage_scope"] = "ambiguous-cgroup-root"
            evidence[stage]["memory"]["usage_trustworthy"] = False
            evidence[stage]["build"]["forced_single_job"] = True
        evidence["memory_policy"]["usage_scope"] = "ambiguous-cgroup-root"
        evidence["memory_policy"]["usage_trustworthy"] = False
        evidence["memory_policy"]["forced_single_job"] = True
        evidence["cgroup_peak"]["scope"] = "ambiguous-cgroup-root"
        evidence["cgroup_peak"]["usage_trustworthy"] = False
        evidence["memory_peak"]["scope"] = "ambiguous-cgroup-root"
        evidence["selected_parallelism"]["max_jobs"] = 1
        evidence["selected_parallelism"]["extension_parallelism"] = 1
        VALIDATOR.validate_resource_evidence(evidence, RESOURCES)

        evidence["selected_parallelism"]["max_jobs"] = 2
        self.assert_rejected(evidence, "requires max_jobs=1")

        evidence["selected_parallelism"]["max_jobs"] = 1
        evidence["selected_parallelism"]["extension_parallelism"] = 2
        self.assert_rejected(evidence, "requires extension_parallelism=1")

    def test_accepts_only_the_restricted_process_group_rss_fallback(self) -> None:
        VALIDATOR.validate_resource_evidence(rss_fallback_evidence(), RESOURCES)

        mutations = (
            ("available", 1),
            ("complete", 1),
            ("kernel_enforced", True),
            ("whole_pod_enforced", True),
            ("whole_pod_enforced", 0),
            ("includes_file_cache", True),
            ("covers_other_pod_processes", True),
            ("sampled_peak_plus_reserve_within_assignment", False),
            ("sample_interval_ms", 250),
            ("sample_interval_ms", 100.0),
            ("checkpoint_interval_ms", 4000),
            ("termination_grace_seconds", 20),
            ("sample_count", 1),
            ("maximum_process_count", 1),
            ("leader_observed", False),
            ("observed_compiler_executable", False),
            ("child_returncode", 1),
            ("build_exit_status", 1),
            ("forwarded_signal", "SIGTERM"),
            ("forced_kill", True),
            ("monitor_error", "sampler failed"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                evidence = rss_fallback_evidence()
                evidence["memory_peak"][field] = value
                self.assert_rejected(evidence, "RSS memory_peak")

        evidence = rss_fallback_evidence()
        evidence["cgroup_peak"]["available"] = True
        self.assert_rejected(evidence, "must not claim cgroup peak evidence")

    def test_rss_fallback_binds_process_group_compiler_and_lifecycle(self) -> None:
        evidence = rss_fallback_evidence()
        evidence["memory_peak"]["process_group_id"] += 1
        self.assert_rejected(evidence, "isolated process-group leader")

        for compiler_executables in (
            [],
            ["ninja"],
            ["nvcc", "nvcc"],
            ["nvcc", ["not-a-string"]],
        ):
            with self.subTest(compiler_executables=compiler_executables):
                evidence = rss_fallback_evidence()
                evidence["memory_peak"]["observed_compiler_executables"] = (
                    compiler_executables
                )
                self.assert_rejected(evidence, "compiler executable evidence")

        evidence = rss_fallback_evidence()
        evidence["memory_peak"]["peak_bytes"] = evidence["memory_peak"]["end_bytes"] - 1
        self.assert_rejected(evidence, "lower than a sampled endpoint")

        evidence = rss_fallback_evidence()
        evidence["memory_peak"]["finished_monotonic_ns"] = evidence["memory_peak"][
            "started_monotonic_ns"
        ]
        self.assert_rejected(evidence, "lifecycle timestamps")

        evidence = rss_fallback_evidence()
        evidence["memory_peak"]["duration_ms"] += 1
        self.assert_rejected(evidence, "duration does not match")

        evidence = rss_fallback_evidence()
        evidence["memory_peak"]["command_sha256"] = "A" * 64
        self.assert_rejected(evidence, "command_sha256")

    def test_rss_fallback_requires_recommended_capacity_and_no_overrides(self) -> None:
        evidence = rss_fallback_evidence()
        below_recommended = 63 * GIB
        for stage in ("resource_start", "resource_end"):
            evidence[stage]["memory"]["assigned_capacity_bytes"] = below_recommended
            evidence[stage]["memory"]["capacity_bytes"] = below_recommended
        evidence["memory_policy"]["assigned_capacity_bytes"] = below_recommended
        evidence["memory_policy"]["capacity_bytes"] = below_recommended
        evidence["runpod_assignment"]["memory_bytes"] = below_recommended
        self.assert_rejected(evidence, "below the matrix recommendation")

        for override in ("low_resource_override", "unsafe_override"):
            with self.subTest(override=override):
                evidence = rss_fallback_evidence()
                evidence["selected_parallelism"][override] = True
                self.assert_rejected(evidence, "fallback forbids")

        evidence = rss_fallback_evidence()
        evidence["resource_start"]["memory"]["usage_current_bytes"] = GIB
        self.assert_rejected(evidence, "genuinely unavailable cgroup usage")

        for field in ("max_jobs", "extension_parallelism"):
            with self.subTest(field=field):
                evidence = rss_fallback_evidence()
                evidence["selected_parallelism"][field] = 2
                self.assert_rejected(evidence, "requires")

    def test_rss_peak_plus_reserve_is_bound_to_verified_capacity(self) -> None:
        evidence = rss_fallback_evidence()
        capacity = evidence["memory_policy"]["capacity_bytes"]
        reserve = evidence["resource_start"]["build"]["reserve_bytes"]
        evidence["memory_peak"]["peak_bytes"] = capacity - reserve + 1
        self.assert_rejected(evidence, "peak plus reserve exceeds")

    def test_memory_policy_must_equal_snapshot_derived_fields(self) -> None:
        for field in tuple(resource_evidence()["memory_policy"]):
            with self.subTest(field=field):
                evidence = resource_evidence()
                value = evidence["memory_policy"][field]
                evidence["memory_policy"][field] = not value if isinstance(value, bool) else None
                self.assert_rejected(evidence, "memory_policy does not match")

    def test_enforces_matrix_cpu_and_memory_minima(self) -> None:
        evidence = resource_evidence()
        for stage in ("resource_start", "resource_end"):
            evidence[stage]["memory"]["capacity_bytes"] = 31 * GIB
        evidence["memory_policy"]["capacity_bytes"] = 31 * GIB
        self.assert_rejected(evidence, "memory capacity is below the matrix minimum")

        for stage in ("resource_start", "resource_end"):
            evidence = resource_evidence()
            evidence[stage]["cpu"]["effective_count"] = 3
            self.assert_rejected(evidence, "effective CPU count is below")

    def test_cgroup_peak_is_bound_monotonic_and_within_capacity(self) -> None:
        mutations = {
            "source": "cgroup-v2:/sys/fs/cgroup/other",
            "scope": "ambiguous-cgroup-root",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                evidence = resource_evidence()
                evidence["cgroup_peak"][field] = value
                self.assert_rejected(evidence, f"cgroup peak {field} does not match")

        evidence = resource_evidence()
        evidence["cgroup_peak"]["start_bytes"] = 0
        self.assert_rejected(evidence, "no positive measured cgroup peak")

        evidence = resource_evidence()
        evidence["cgroup_peak"]["end_bytes"] = GIB
        self.assert_rejected(evidence, "cgroup peak decreased")

        evidence = resource_evidence()
        evidence["cgroup_peak"]["end_bytes"] = 41 * GIB
        self.assert_rejected(evidence, "cgroup peak exceeds selected memory capacity")

        for field, message in (
            ("monotonic", "monotonic policy"),
            ("within_capacity", "capacity policy"),
        ):
            evidence = resource_evidence()
            evidence["cgroup_peak"][field] = False
            self.assert_rejected(evidence, message)


if __name__ == "__main__":
    unittest.main()
