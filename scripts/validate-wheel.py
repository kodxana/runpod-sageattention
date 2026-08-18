#!/usr/bin/env python3
"""Validate wheel metadata, cubin coverage, manifest, and checksums without a GPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any


class ValidationError(RuntimeError):
    """A release-blocking wheel validation failed."""


def load_inspector() -> ModuleType:
    path = Path(__file__).with_name("inspect-wheel.py")
    spec = importlib.util.spec_from_file_location("sageattention_wheel_inspector", path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"Cannot load wheel inspector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _runtime_error(stage: str, error: BaseException | str) -> dict[str, str]:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error) or error_type
    else:
        error_type = "ValidationError"
        message = error
    return {
        "stage": stage,
        "type": error_type,
        "message": message,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def normalized_requirement(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement.split(";", 1)[0]).lower()


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _compiler_executable_name(value: object) -> bool:
    if not isinstance(value, str) or not value or Path(value).name != value:
        return False
    lowered = value.lower()
    return (
        lowered in {"c++", "cc1plus", "nvcc", "ptxas"}
        or "nvcc" in lowered
        or "ptxas" in lowered
        or "g++" in lowered
        or lowered.endswith("-c++")
    )


def validate_resource_evidence(
    build_evidence: dict[str, Any],
    resources: dict[str, Any],
) -> None:
    """Bind promotion evidence to one stable, verified Pod assignment."""

    minimum_cpus = resources.get("minimum_cpus")
    minimum_memory_gib = resources.get("minimum_memory_gib")
    recommended_memory_gib = resources.get("recommended_memory_gib")
    require(_positive_integer(minimum_cpus), "matrix minimum CPU count is invalid")
    require(
        _positive_integer(minimum_memory_gib),
        "matrix minimum memory GiB is invalid",
    )
    require(
        _positive_integer(recommended_memory_gib)
        and recommended_memory_gib >= minimum_memory_gib,
        "matrix recommended memory GiB is invalid",
    )
    minimum_memory_bytes = minimum_memory_gib * 1024 ** 3
    recommended_memory_bytes = recommended_memory_gib * 1024 ** 3

    memory_policy = build_evidence.get("memory_policy")
    require(isinstance(memory_policy, dict), "build evidence has no memory_policy")

    snapshots: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for evidence_name in ("resource_start", "resource_end"):
        snapshot = build_evidence.get(evidence_name)
        require(isinstance(snapshot, dict), f"build evidence has no {evidence_name}")
        require(
            snapshot.get("schema_version") == 2,
            f"{evidence_name} must use pod-resources schema version 2",
        )
        memory = snapshot.get("memory")
        cpu = snapshot.get("cpu")
        build = snapshot.get("build")
        cgroup = snapshot.get("cgroup")
        require(isinstance(memory, dict), f"{evidence_name} has no memory snapshot")
        require(isinstance(cpu, dict), f"{evidence_name} has no CPU snapshot")
        require(isinstance(build, dict), f"{evidence_name} has no build recommendation")
        require(isinstance(cgroup, dict), f"{evidence_name} has no cgroup snapshot")

        assigned_capacity = memory.get("assigned_capacity_bytes")
        capacity = memory.get("capacity_bytes")
        capacity_source = memory.get("capacity_source")
        capacity_is_hard_limit = memory.get("capacity_is_hard_limit")
        usage_source = memory.get("usage_source")
        usage_scope = memory.get("usage_scope")
        usage_peak_eligible = memory.get("usage_peak_eligible")
        usage_trustworthy = memory.get("usage_trustworthy")
        peak_evidence_mode = memory.get("peak_evidence_mode")
        effective_cpus = cpu.get("effective_count")
        runpod_cpus = cpu.get("runpod_count")
        cgroup_version = cgroup.get("version")

        require(
            _positive_integer(assigned_capacity),
            f"{evidence_name} lost the verified Runpod memory assignment",
        )
        require(
            _positive_integer(capacity),
            f"{evidence_name} memory capacity is missing or invalid",
        )
        require(
            capacity >= minimum_memory_bytes,
            f"{evidence_name} memory capacity is below the matrix minimum",
        )
        require(
            capacity <= assigned_capacity,
            f"{evidence_name} memory capacity exceeds the verified assignment",
        )
        require(
            isinstance(capacity_source, str) and bool(capacity_source),
            f"{evidence_name} memory capacity source is missing",
        )
        require(
            isinstance(capacity_is_hard_limit, bool),
            f"{evidence_name} memory hard-limit policy is malformed",
        )
        require(
            isinstance(usage_trustworthy, bool),
            f"{evidence_name} usage trust policy is malformed",
        )
        require(
            peak_evidence_mode in {"cgroup", "process-group-rss"},
            f"{evidence_name} memory peak evidence mode is unsupported",
        )
        cgroup_prefix = f"cgroup-v{cgroup_version}:"
        if capacity_source == "runpod-api-assignment":
            require(
                capacity == assigned_capacity and capacity_is_hard_limit is False,
                f"{evidence_name} Runpod assignment capacity policy is inconsistent",
            )
        else:
            require(
                _positive_integer(cgroup_version)
                and cgroup_version in {1, 2}
                and capacity_source.startswith(cgroup_prefix)
                and capacity_is_hard_limit is True
                and memory.get("limited") is True
                and memory.get("limit_bytes") == capacity,
                f"{evidence_name} cgroup capacity policy is inconsistent",
            )
        require(
            _positive_integer(effective_cpus) and effective_cpus >= minimum_cpus,
            f"{evidence_name} effective CPU count is below the matrix minimum",
        )
        require(
            _positive_integer(runpod_cpus) and runpod_cpus >= minimum_cpus,
            f"{evidence_name} lost the verified Runpod vCPU assignment",
        )
        require(
            effective_cpus <= runpod_cpus,
            f"{evidence_name} effective CPU exceeds the verified assignment",
        )
        require(
            isinstance(build.get("forced_single_job"), bool),
            f"{evidence_name} forced-single-job policy is malformed",
        )

        if peak_evidence_mode == "cgroup":
            require(
                _positive_integer(cgroup_version) and cgroup_version in {1, 2},
                f"{evidence_name} has no supported memory cgroup",
            )
            require(
                isinstance(usage_source, str)
                and usage_source.startswith(cgroup_prefix),
                f"{evidence_name} peak usage source is not a cgroup",
            )
            require(
                usage_scope
                in {"cgroup-capacity", "pod-cgroup", "ambiguous-cgroup-root"},
                f"{evidence_name} peak usage scope is unsupported",
            )
            require(
                usage_peak_eligible is True,
                f"{evidence_name} is not eligible for cgroup peak evidence",
            )
            if usage_trustworthy is False:
                require(
                    usage_scope == "ambiguous-cgroup-root"
                    and build["forced_single_job"] is True,
                    f"{evidence_name} untrusted usage is not safely constrained",
                )
        else:
            require(
                capacity_source == "runpod-api-assignment"
                and capacity == assigned_capacity
                and capacity_is_hard_limit is False,
                f"{evidence_name} RSS fallback requires the verified assignment capacity",
            )
            require(
                capacity >= recommended_memory_bytes,
                f"{evidence_name} RSS fallback capacity is below the matrix recommendation",
            )
            require(
                cgroup_version is None
                or (_positive_integer(cgroup_version) and cgroup_version in {1, 2}),
                f"{evidence_name} RSS fallback cgroup metadata is malformed",
            )
            require(
                memory.get("usage_current_bytes") is None
                and usage_source == ""
                and usage_scope == "unavailable"
                and usage_peak_eligible is False
                and usage_trustworthy is False,
                f"{evidence_name} RSS fallback requires genuinely unavailable cgroup usage",
            )
            require(
                build["forced_single_job"] is True
                and _positive_integer(build.get("suggested_jobs"))
                and build["suggested_jobs"] == 1
                and _positive_integer(build.get("max_jobs_cap"))
                and build["max_jobs_cap"] == 1,
                f"{evidence_name} RSS fallback is not constrained to one build job",
            )
        snapshots[evidence_name] = (memory, cpu, build)

    start_memory, start_cpu, start_build = snapshots["resource_start"]
    end_memory, end_cpu, end_build = snapshots["resource_end"]
    bound_memory_fields = (
        "assigned_capacity_bytes",
        "capacity_bytes",
        "capacity_source",
        "capacity_is_hard_limit",
        "usage_source",
        "usage_scope",
        "usage_peak_eligible",
        "peak_evidence_mode",
        "usage_trustworthy",
    )
    for field in bound_memory_fields:
        require(
            end_memory.get(field) == start_memory.get(field),
            f"resource assignment changed during build: {field}",
        )
    require(
        end_build.get("forced_single_job") == start_build.get("forced_single_job"),
        "forced-single-job policy changed during build",
    )
    require(
        end_cpu.get("runpod_count") == start_cpu.get("runpod_count"),
        "Runpod vCPU assignment changed during build",
    )

    derived_runpod_assignment = {
        "memory_bytes": start_memory["assigned_capacity_bytes"],
        "vcpu_count": start_cpu["runpod_count"],
    }
    require(
        build_evidence.get("runpod_assignment") == derived_runpod_assignment,
        "build evidence runpod_assignment does not match resource snapshots",
    )

    derived_memory_policy = {
        "assigned_capacity_bytes": start_memory["assigned_capacity_bytes"],
        "capacity_bytes": start_memory["capacity_bytes"],
        "capacity_is_hard_limit": start_memory["capacity_is_hard_limit"],
        "capacity_source": start_memory["capacity_source"],
        "forced_single_job": start_build["forced_single_job"],
        "peak_evidence_mode": start_memory["peak_evidence_mode"],
        "usage_peak_eligible": start_memory["usage_peak_eligible"],
        "usage_scope": start_memory["usage_scope"],
        "usage_source": start_memory["usage_source"],
        "usage_trustworthy": start_memory["usage_trustworthy"],
    }
    require(
        memory_policy == derived_memory_policy,
        "build evidence memory_policy does not match resource snapshots",
    )
    selected_parallelism = build_evidence.get("selected_parallelism")
    require(
        isinstance(selected_parallelism, dict),
        "build evidence has no selected_parallelism",
    )
    if memory_policy["forced_single_job"] is True:
        require(
            _positive_integer(selected_parallelism.get("max_jobs"))
            and selected_parallelism["max_jobs"] == 1,
            "forced-single-job memory policy requires max_jobs=1",
        )
        require(
            _positive_integer(selected_parallelism.get("extension_parallelism"))
            and selected_parallelism["extension_parallelism"] == 1,
            "forced-single-job memory policy requires extension_parallelism=1",
        )

    peak_evidence_mode = memory_policy["peak_evidence_mode"]
    if peak_evidence_mode == "process-group-rss":
        for field in (
            "suggested_jobs",
            "max_jobs_cap",
            "reserve_bytes",
            "memory_per_job_bytes",
        ):
            require(
                end_build.get(field) == start_build.get(field),
                f"RSS fallback build policy changed during build: {field}",
            )
        require(
            _positive_integer(start_build.get("reserve_bytes")),
            "RSS fallback build reserve is missing or invalid",
        )
        require(
            _positive_integer(start_build.get("memory_per_job_bytes")),
            "RSS fallback compiler memory assumption is missing or invalid",
        )
        require(
            selected_parallelism.get("low_resource_override") is False,
            "RSS fallback forbids the low-resource override",
        )
        require(
            selected_parallelism.get("unsafe_override") is False,
            "RSS fallback forbids the unsafe-parallelism override",
        )

    cgroup_peak = build_evidence.get("cgroup_peak")
    require(isinstance(cgroup_peak, dict), "build evidence has no cgroup_peak")
    selected_capacity = memory_policy["capacity_bytes"]
    memory_peak = build_evidence.get("memory_peak")
    require(isinstance(memory_peak, dict), "build evidence has no memory_peak")

    if peak_evidence_mode == "cgroup":
        require(
            cgroup_peak.get("available") is True,
            "cgroup peak is not marked available",
        )
        require(
            cgroup_peak.get("source") == memory_policy["usage_source"],
            "cgroup peak source does not match memory policy",
        )
        require(
            cgroup_peak.get("scope") == memory_policy["usage_scope"],
            "cgroup peak scope does not match memory policy",
        )
        require(
            cgroup_peak.get("usage_trustworthy")
            == memory_policy["usage_trustworthy"],
            "cgroup peak trust policy does not match memory policy",
        )
        peak_start = cgroup_peak.get("start_bytes")
        peak_end = cgroup_peak.get("end_bytes")
        require(
            _positive_integer(peak_start) and _positive_integer(peak_end),
            "build evidence has no positive measured cgroup peak",
        )
        require(peak_end >= peak_start, "cgroup peak decreased during build")
        require(
            cgroup_peak.get("monotonic") is True,
            "cgroup peak monotonic policy is missing or inconsistent",
        )
        require(
            peak_start <= selected_capacity and peak_end <= selected_capacity,
            "cgroup peak exceeds selected memory capacity",
        )
        require(
            cgroup_peak.get("within_capacity") is True,
            "cgroup peak capacity policy is missing or inconsistent",
        )
        expected_memory_peak = {
            "available": True,
            "complete": True,
            "end_bytes": peak_end,
            "includes_file_cache": True,
            "kernel_enforced": memory_policy["capacity_is_hard_limit"],
            "method": "kernel-cgroup-peak",
            "mode": "cgroup",
            "peak_bytes": peak_end,
            "sample_interval_ms": None,
            "scope": memory_policy["usage_scope"],
            "source": memory_policy["usage_source"],
            "start_bytes": peak_start,
            "within_selected_capacity": True,
        }
        require(
            memory_peak == expected_memory_peak,
            "cgroup memory_peak does not match the measured cgroup evidence",
        )
    else:
        unavailable_cgroup_peak = {
            "available": False,
            "end_bytes": None,
            "monotonic": None,
            "scope": "unavailable",
            "source": "",
            "start_bytes": None,
            "usage_trustworthy": False,
            "within_capacity": None,
        }
        require(
            cgroup_peak == unavailable_cgroup_peak,
            "RSS fallback must not claim cgroup peak evidence",
        )
        expected_memory_peak_fields = {
            "available",
            "build_exit_status",
            "child_returncode",
            "checkpoint_interval_ms",
            "command_sha256",
            "complete",
            "covers_other_pod_processes",
            "duration_ms",
            "end_bytes",
            "finished_monotonic_ns",
            "forced_kill",
            "forwarded_signal",
            "includes_file_cache",
            "kernel_enforced",
            "leader_observed",
            "leader_pid",
            "maximum_process_count",
            "method",
            "mode",
            "monitor_error",
            "observed_compiler_executable",
            "observed_compiler_executables",
            "peak_bytes",
            "process_group_id",
            "sample_count",
            "sample_interval_ms",
            "sampled_peak_plus_reserve_within_assignment",
            "scope",
            "shared_pages_may_be_double_counted",
            "source",
            "start_bytes",
            "started_monotonic_ns",
            "termination_grace_seconds",
            "whole_pod_enforced",
        }
        require(
            set(memory_peak) == expected_memory_peak_fields,
            "RSS memory_peak fields are incomplete or unexpected",
        )
        expected_rss_policy = {
            "available": True,
            "build_exit_status": 0,
            "child_returncode": 0,
            "complete": True,
            "covers_other_pod_processes": False,
            "forced_kill": False,
            "forwarded_signal": None,
            "includes_file_cache": False,
            "kernel_enforced": False,
            "leader_observed": True,
            "method": "proc-process-group-rss-sum",
            "mode": "process-group-rss",
            "monitor_error": None,
            "observed_compiler_executable": True,
            "sampled_peak_plus_reserve_within_assignment": True,
            "scope": "build-process-group",
            "shared_pages_may_be_double_counted": True,
            "source": "/proc/*/statm",
            "whole_pod_enforced": False,
        }
        for field, expected in expected_rss_policy.items():
            matches = (
                memory_peak.get(field) is expected
                if isinstance(expected, bool) or expected is None
                else memory_peak.get(field) == expected
            )
            require(
                matches,
                f"RSS memory_peak has invalid {field}",
            )
        for field, expected in (
            ("sample_interval_ms", 100),
            ("checkpoint_interval_ms", 5000),
            ("termination_grace_seconds", 10),
        ):
            value = memory_peak.get(field)
            require(
                _positive_integer(value) and value == expected,
                f"RSS memory_peak has invalid {field}",
            )
        for field in ("child_returncode", "build_exit_status"):
            value = memory_peak.get(field)
            require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value == 0,
                f"RSS memory_peak has invalid {field}",
            )
        command_sha256 = memory_peak.get("command_sha256")
        require(
            isinstance(command_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", command_sha256) is not None,
            "RSS memory_peak has invalid command_sha256",
        )
        start_bytes = memory_peak.get("start_bytes")
        end_bytes = memory_peak.get("end_bytes")
        peak_bytes = memory_peak.get("peak_bytes")
        require(
            _positive_integer(start_bytes)
            and _positive_integer(end_bytes)
            and _positive_integer(peak_bytes),
            "RSS memory_peak has no positive sampled endpoints and peak",
        )
        require(
            peak_bytes >= max(start_bytes, end_bytes),
            "RSS memory_peak is lower than a sampled endpoint",
        )
        require(
            _positive_integer(memory_peak.get("sample_count"))
            and memory_peak["sample_count"] >= 2,
            "RSS memory_peak has too few samples",
        )
        require(
            _positive_integer(memory_peak.get("maximum_process_count"))
            and memory_peak["maximum_process_count"] >= 2,
            "RSS memory_peak did not observe the build leader and a compiler",
        )
        process_group_id = memory_peak.get("process_group_id")
        leader_pid = memory_peak.get("leader_pid")
        require(
            _positive_integer(process_group_id)
            and _positive_integer(leader_pid)
            and process_group_id == leader_pid,
            "RSS memory_peak is not bound to its isolated process-group leader",
        )
        compiler_executables = memory_peak.get("observed_compiler_executables")
        require(
            isinstance(compiler_executables, list)
            and bool(compiler_executables)
            and all(_compiler_executable_name(item) for item in compiler_executables)
            and compiler_executables == sorted(set(compiler_executables)),
            "RSS memory_peak has invalid compiler executable evidence",
        )
        started_monotonic_ns = memory_peak.get("started_monotonic_ns")
        finished_monotonic_ns = memory_peak.get("finished_monotonic_ns")
        duration_ms = memory_peak.get("duration_ms")
        require(
            _positive_integer(started_monotonic_ns)
            and _positive_integer(finished_monotonic_ns)
            and finished_monotonic_ns > started_monotonic_ns,
            "RSS memory_peak lifecycle timestamps are invalid",
        )
        require(
            _positive_integer(duration_ms)
            and duration_ms
            == (finished_monotonic_ns - started_monotonic_ns) // 1_000_000,
            "RSS memory_peak duration does not match its lifecycle timestamps",
        )
        require(
            peak_bytes + start_build["reserve_bytes"] <= selected_capacity,
            "RSS peak plus reserve exceeds the verified assignment capacity",
        )


def _evaluate_runtime_case(
    *,
    torch_module: Any,
    implementation: Any,
    implementation_name: str,
    causal: bool,
    reference: Any,
    expected_output_shape: list[int],
    expected_output_dtype: str,
    expected_dtype: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "causal": causal,
        "expected_output_dtype": expected_output_dtype,
        "expected_output_shape": expected_output_shape,
        "implementation": implementation_name,
    }
    errors: list[dict[str, str]] = []

    try:
        raw_actual = implementation(causal)
        torch_module.cuda.synchronize()
    except Exception as error:
        result["errors"] = [_runtime_error("execution", error)]
        return result

    if not isinstance(raw_actual, torch_module.Tensor):
        result["errors"] = [
            _runtime_error(
                "output-validation",
                f"non-tensor output for {implementation_name}, causal={causal}",
            )
        ]
        return result

    output_shape = list(raw_actual.shape)
    output_dtype = str(raw_actual.dtype).removeprefix("torch.")
    result["output_dtype"] = output_dtype
    result["output_shape"] = output_shape
    output_is_cuda = bool(raw_actual.is_cuda)
    shape_matches = output_shape == expected_output_shape
    dtype_matches = raw_actual.dtype == expected_dtype

    if not output_is_cuda:
        errors.append(_runtime_error(
            "output-validation",
            f"non-CUDA output for {implementation_name}, causal={causal}",
        ))
    if not shape_matches:
        errors.append(_runtime_error(
            "output-validation",
            f"output shape {output_shape} != {expected_output_shape} for "
            f"{implementation_name}, causal={causal}",
        ))
    if not dtype_matches:
        errors.append(_runtime_error(
            "output-validation",
            f"output dtype {output_dtype} != {expected_output_dtype} for "
            f"{implementation_name}, causal={causal}",
        ))

    try:
        actual = raw_actual.float()
        output_is_finite = bool(torch_module.isfinite(actual).all())
    except Exception as error:
        errors.append(_runtime_error("output-validation", error))
        output_is_finite = False
        actual = None

    if actual is not None and not output_is_finite:
        errors.append(_runtime_error(
            "output-validation",
            f"non-finite output for {implementation_name}, causal={causal}",
        ))

    if actual is not None and output_is_cuda and shape_matches and output_is_finite:
        try:
            reference_flat = reference.flatten()
            actual_flat = actual.flatten()
            cosine = float(
                torch_module.dot(reference_flat, actual_flat)
                / (
                    torch_module.linalg.vector_norm(reference_flat)
                    * torch_module.linalg.vector_norm(actual_flat)
                ).clamp_min(1e-12)
            )
            relative_l2 = float(
                torch_module.linalg.vector_norm(actual_flat - reference_flat)
                / torch_module.linalg.vector_norm(reference_flat).clamp_min(1e-12)
            )
            if not math.isfinite(cosine):
                errors.append(_runtime_error(
                    "numeric-validation",
                    f"non-finite cosine for {implementation_name}, causal={causal}",
                ))
            else:
                result["cosine_similarity"] = cosine
                if cosine < policy["minimum_cosine_similarity"]:
                    errors.append(_runtime_error(
                        "numeric-validation",
                        f"cosine {cosine:.6f} below threshold for "
                        f"{implementation_name}, causal={causal}",
                    ))
            if not math.isfinite(relative_l2):
                errors.append(_runtime_error(
                    "numeric-validation",
                    f"non-finite relative L2 for {implementation_name}, causal={causal}",
                ))
            else:
                result["relative_l2"] = relative_l2
                if relative_l2 > policy["maximum_relative_l2"]:
                    errors.append(_runtime_error(
                        "numeric-validation",
                        f"relative L2 {relative_l2:.6f} above threshold for "
                        f"{implementation_name}, causal={causal}",
                    ))
        except Exception as error:
            errors.append(_runtime_error("numeric-validation", error))

    if errors:
        result["errors"] = errors
    return result


def run_runtime_validation(
    matrix: dict[str, Any],
    build: dict[str, Any],
    artifact: dict[str, Any],
    expected_capability: str,
    report_path: Path,
) -> dict[str, Any]:
    import importlib.metadata

    import torch
    import torch.nn.functional as functional
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    known_capabilities = {
        entry["compute_capability"]
        for entry in matrix["validation"]["representative_gpu_capabilities"]
        if entry["required"]
    }
    require(expected_capability in known_capabilities,
            f"capability {expected_capability} is not a required matrix representative")
    require(torch.cuda.is_available(), "CUDA is not available for runtime validation")
    require(str(torch.__version__) == build["torch_version"],
            f"runtime torch mismatch: expected {build['torch_version']}, got {torch.__version__}")
    require(str(torch.version.cuda) == build["torch_cuda_version"],
            f"runtime torch CUDA mismatch: expected {build['torch_cuda_version']}, "
            f"got {torch.version.cuda}")
    require(importlib.metadata.version("sageattention") == build["wheel_version"],
            "installed SageAttention version does not match the selected wheel")
    compiled_modules = {}
    for module_name in (
        "sageattention._fused",
        "sageattention._qattn_sm80",
        "sageattention._qattn_sm89",
        "sageattention._qattn_sm90",
    ):
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        require(bool(module_path), f"compiled module has no file path: {module_name}")
        compiled_modules[module_name] = str(Path(module_path).resolve())

    major, minor = torch.cuda.get_device_capability(0)
    actual_capability = f"{major}.{minor}"
    require(actual_capability == expected_capability,
            f"scheduler GPU mismatch: expected {expected_capability}, got {actual_capability}")
    cuda_device_name = torch.cuda.get_device_name(0)

    policy = matrix["validation"]["runtime_numeric"]
    case = policy["canonical_case"]
    dtype_by_name = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    require(case["dtype"] in dtype_by_name, f"unsupported canonical dtype: {case['dtype']}")
    require(case["tensor_layout"] == "HND", "runtime validator currently requires HND")
    dtype = dtype_by_name[case["dtype"]]
    shape = (
        case["batch_size"],
        case["query_heads"],
        case["sequence_length"],
        case["head_dimension"],
    )
    require(case["query_heads"] == case["key_value_heads"],
            "canonical runtime validator requires equal Q and KV head counts")

    torch.manual_seed(case["seed"])
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    implementations = {
        "sageattn_dispatch": lambda causal: sageattn(
            q, k, v,
            tensor_layout=case["tensor_layout"],
            is_causal=causal,
        ),
        "qattn_sm80_cuda": lambda causal: sageattn_qk_int8_pv_fp16_cuda(
            q, k, v,
            tensor_layout=case["tensor_layout"],
            is_causal=causal,
            pv_accum_dtype="fp32",
        ),
        "qattn_sm89_cuda": lambda causal: sageattn_qk_int8_pv_fp8_cuda(
            q, k, v,
            tensor_layout=case["tensor_layout"],
            is_causal=causal,
            qk_quant_gran="per_warp" if actual_capability == "12.0" else "per_thread",
            pv_accum_dtype="fp32+fp16",
        ),
        "qattn_sm90_cuda": lambda causal: sageattn_qk_int8_pv_fp8_cuda_sm90(
            q, k, v,
            tensor_layout=case["tensor_layout"],
            is_causal=causal,
            pv_accum_dtype="fp32+fp32",
        ),
    }
    required_implementations = policy["implementations_by_capability"][actual_capability]
    require(len(required_implementations) == len(set(required_implementations)),
            f"duplicate runtime implementation policy for {actual_capability}")
    require(set(required_implementations) <= set(implementations),
            f"unknown runtime implementation policy for {actual_capability}")

    runtime_image_ref = os.environ.get("RUNTIME_IMAGE_REF", "").strip()
    require(bool(runtime_image_ref),
            "RUNTIME_IMAGE_REF must contain the selected immutable image ref")

    references: dict[bool, Any] = {}
    reference_errors: dict[bool, dict[str, str]] = {}
    for causal in case["causal_modes"]:
        try:
            with sdpa_kernel(SDPBackend.MATH):
                references[causal] = functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), is_causal=causal)
            torch.cuda.synchronize()
        except Exception as error:
            reference_errors[causal] = _runtime_error("reference", error)

    expected_output_shape = list(shape)
    expected_output_dtype = case["dtype"]
    results: list[dict[str, Any]] = []
    for implementation_name in required_implementations:
        implementation = implementations[implementation_name]
        for causal in case["causal_modes"]:
            if causal in reference_errors:
                results.append({
                    "causal": causal,
                    "errors": [reference_errors[causal]],
                    "expected_output_dtype": expected_output_dtype,
                    "expected_output_shape": expected_output_shape,
                    "implementation": implementation_name,
                })
                continue
            results.append(_evaluate_runtime_case(
                torch_module=torch,
                implementation=implementation,
                implementation_name=implementation_name,
                causal=causal,
                reference=references[causal],
                expected_output_dtype=expected_output_dtype,
                expected_output_shape=expected_output_shape,
                expected_dtype=dtype,
                policy=policy,
            ))

    failures = [
        {
            "causal": result["causal"],
            "errors": result["errors"],
            "implementation": result["implementation"],
        }
        for result in results
        if result.get("errors")
    ]
    report = {
        "build_id": build["id"],
        "status": "fail" if failures else "pass",
        "wheel_asset": artifact["asset"],
        "wheel_sha256": artifact["sha256"],
        "expected_compute_capability": expected_capability,
        "actual_compute_capability": actual_capability,
        "cuda_device_name": cuda_device_name,
        "compiled_modules": compiled_modules,
        "expected_runtime_image": build["comfyui_runtime_image"],
        "runtime_image_ref": runtime_image_ref,
        "results": results,
        "runtime_numeric_policy": policy,
        "schema_version": 1,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "sageattention_version": build["wheel_version"],
    }
    if failures:
        report["failures"] = failures
    _write_json_atomic(report_path, report)
    if failures:
        first = failures[0]
        first_error = first["errors"][0]["message"]
        raise ValidationError(
            f"runtime numerical validation failed for {len(failures)} of "
            f"{len(results)} outcomes; diagnostic report: {report_path}; "
            f"first failure: {first['implementation']}, causal={first['causal']}: "
            f"{first_error}"
        )
    return report


def validate(
    *,
    matrix_path: Path,
    build_id: str,
    wheel: Path,
    manifest_path: Path | None,
    checksums_path: Path | None,
    cuobjdump: str | None,
) -> dict[str, Any]:
    inspector = load_inspector()
    matrix = inspector.load_matrix(matrix_path)
    build = inspector.find_build(matrix, build_id)
    package = matrix["package"]
    platform = matrix["platform"]
    policy = matrix["cuda_policy"]
    thresholds = matrix["validation"]

    require(wheel.name == build["wheel_filename"],
            f"wheel filename mismatch: expected {build['wheel_filename']}, got {wheel.name}")
    require(wheel.stat().st_size >= thresholds["minimum_wheel_size_bytes"],
            f"wheel is unexpectedly small: {wheel.stat().st_size} bytes")
    require(wheel.stat().st_size <= thresholds["maximum_wheel_size_bytes"],
            f"wheel exceeds size ceiling: {wheel.stat().st_size} bytes")

    info = inspector.read_wheel(wheel, cuobjdump=cuobjdump)
    require(str(info["distribution"]).lower() == package["distribution"].lower(),
            f"distribution mismatch: {info['distribution']}")
    require(str(info["version"]) == build["wheel_version"],
            f"version mismatch: expected {build['wheel_version']}, got {info['version']}")

    exact_torch = f"torch=={build['torch_version']}".lower()
    requirements = {normalized_requirement(str(value)) for value in info["requires_dist"]}
    require(exact_torch in requirements,
            f"missing exact runtime requirement {exact_torch}; got {sorted(requirements)}")

    expected_tag = f"{platform['python_tag']}-{platform['abi_tag']}-{platform['platform_tag']}"
    require(expected_tag in {str(tag) for tag in info["wheel_tags"]},
            f"missing wheel tag {expected_tag}; got {info['wheel_tags']}")

    expected_extensions = set(thresholds["required_extensions"])
    actual_extensions = set(info["extension_members"])
    require(actual_extensions == expected_extensions,
            f"extension set mismatch: expected {sorted(expected_extensions)}, "
            f"got {sorted(actual_extensions)}")

    expected_architectures = {
        name: sorted(values)
        for name, values in policy["extension_cubin_architectures"].items()
    }
    actual_architectures = {
        name: sorted(values)
        for name, values in info["extension_architectures"].items()
    }
    require(actual_architectures == expected_architectures,
            "cubin matrix mismatch:\n"
            f"expected={json.dumps(expected_architectures, sort_keys=True)}\n"
            f"actual={json.dumps(actual_architectures, sort_keys=True)}")
    require(not info["ptx_extensions"],
            f"PTX is forbidden but found in: {info['ptx_extensions']}")

    with __import__("zipfile").ZipFile(wheel) as archive:
        suspicious = [
            name for name in archive.namelist()
            if "libcuda" in name.lower() or "/stubs/" in name.lower()
        ]
    require(not suspicious, f"CUDA driver stubs must not be packaged: {suspicious}")

    build_evidence = None
    if manifest_path is not None:
        with manifest_path.open("r", encoding="utf-8") as handle:
            actual_manifest = json.load(handle)
        artifacts = actual_manifest.get("artifacts", [])
        require(len(artifacts) == 1, "per-build manifest must contain exactly one artifact")
        build_evidence = artifacts[0].get("build_evidence")
        require(isinstance(build_evidence, dict), "manifest is missing build evidence")
        required_evidence = {
            "builder_image",
            "cuda_visible_devices",
            "cgroup_peak",
            "elapsed_seconds",
            "matrix_sha256",
            "memory_peak",
            "memory_policy",
            "patch_sha256",
            "resource_end",
            "resource_start",
            "runpod_assignment",
            "selected_gpu_id",
            "selected_parallelism",
            "tool_versions",
        }
        require(required_evidence <= set(build_evidence),
                f"build evidence is incomplete: missing {sorted(required_evidence - set(build_evidence))}")
        require(build_evidence["cuda_visible_devices"] == "",
                "build evidence does not prove that the GPU was hidden")
        selected_gpu_id = build_evidence["selected_gpu_id"]
        require(
            selected_gpu_id is None
            or (
                isinstance(selected_gpu_id, str)
                and bool(selected_gpu_id)
                and selected_gpu_id == selected_gpu_id.strip()
                and "\n" not in selected_gpu_id
                and "\r" not in selected_gpu_id
                and "," not in selected_gpu_id
            ),
            "build evidence selected_gpu_id is malformed",
        )
        patch_path = (
            matrix_path.parent / "patches" / "sageattention"
            / package["upstream_version"] / "setup.py.patch"
        )
        require(build_evidence["matrix_sha256"] == inspector.sha256_file(matrix_path),
                "build evidence matrix hash mismatch")
        require(build_evidence["patch_sha256"] == inspector.sha256_file(patch_path),
                "build evidence patch hash mismatch")
        builder_evidence = build_evidence["builder_image"]
        require(builder_evidence.get("expected") == build["builder_image"],
                "build evidence expected builder does not match matrix")
        require(bool(builder_evidence.get("ref")), "build evidence has no builder image ref")
        validate_resource_evidence(build_evidence, matrix["resources"])
        selected = build_evidence["selected_parallelism"]
        exceeds_default = (
            selected.get("max_jobs", 0) > matrix["resources"]["default_max_jobs"]
            or selected.get("extension_parallelism", 0)
            > matrix["resources"]["default_extension_parallelism"]
        )
        require(not exceeds_default or selected.get("unsafe_override") is True,
                "build evidence exceeds reviewed parallelism without explicit override")

    expected_manifest = inspector.build_manifest(
        matrix, build_id, wheel, info, build_evidence)
    if manifest_path is not None:
        require(actual_manifest == expected_manifest, "manifest does not match inspected wheel")

    artifact = expected_manifest["artifacts"][0]
    if checksums_path is not None:
        checksum_lines = checksums_path.read_text(encoding="ascii").splitlines()
        expected_line = f"{artifact['sha256']}  {wheel.name}"
        require(checksum_lines == [expected_line],
                f"checksum file must contain exactly: {expected_line}")

    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checksums", type=Path)
    parser.add_argument("--cuobjdump", default=None)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--expected-capability")
    parser.add_argument("--runtime-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = validate(
        matrix_path=args.matrix.resolve(),
        build_id=args.build_id,
        wheel=args.wheel.resolve(),
        manifest_path=args.manifest.resolve() if args.manifest else None,
        checksums_path=args.checksums.resolve() if args.checksums else None,
        cuobjdump=args.cuobjdump,
    )
    if args.runtime:
        if not args.expected_capability or not args.runtime_report:
            raise ValidationError(
                "--runtime requires --expected-capability and --runtime-report")
        inspector = load_inspector()
        matrix = inspector.load_matrix(args.matrix.resolve())
        build = inspector.find_build(matrix, args.build_id)
        report = run_runtime_validation(
            matrix,
            build,
            artifact,
            args.expected_capability,
            args.runtime_report.resolve(),
        )
        print(
            f"runtime validated sm{report['actual_compute_capability'].replace('.', '')}: "
            f"{args.runtime_report.resolve()}"
        )
    print(
        "validated "
        f"{artifact['asset']} ({artifact['sha256']}, {artifact['size']} bytes)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"wheel validation failed: {error}") from error
