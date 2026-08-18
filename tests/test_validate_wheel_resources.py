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
RESOURCES = {"minimum_cpus": 4, "minimum_memory_gib": 32}


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
        "build": {"forced_single_job": False},
    }
    memory_policy = {
        "assigned_capacity_bytes": memory["assigned_capacity_bytes"],
        "capacity_bytes": memory["capacity_bytes"],
        "capacity_is_hard_limit": memory["capacity_is_hard_limit"],
        "capacity_source": memory["capacity_source"],
        "forced_single_job": False,
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
            "start_bytes": 2 * GIB,
            "end_bytes": 8 * GIB,
            "monotonic": True,
            "scope": usage_scope,
            "source": usage_source,
            "usage_trustworthy": True,
            "within_capacity": True,
        },
        "selected_parallelism": {"max_jobs": 2},
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
        evidence["selected_parallelism"]["max_jobs"] = 1
        evidence["selected_parallelism"]["extension_parallelism"] = 1
        VALIDATOR.validate_resource_evidence(evidence, RESOURCES)

        evidence["selected_parallelism"]["max_jobs"] = 2
        self.assert_rejected(evidence, "requires max_jobs=1")

        evidence["selected_parallelism"]["max_jobs"] = 1
        evidence["selected_parallelism"]["extension_parallelism"] = 2
        self.assert_rejected(evidence, "requires extension_parallelism=1")

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
