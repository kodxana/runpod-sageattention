#!/usr/bin/env python3
"""Validate wheel metadata, cubin coverage, manifest, and checksums without a GPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
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


def normalized_requirement(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement.split(";", 1)[0]).lower()


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

    references = {}
    for causal in case["causal_modes"]:
        with sdpa_kernel(SDPBackend.MATH):
            references[causal] = functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=causal)

    results = []
    for implementation_name in required_implementations:
        implementation = implementations[implementation_name]
        for causal in case["causal_modes"]:
            raw_actual = implementation(causal)
            torch.cuda.synchronize()
            require(
                isinstance(raw_actual, torch.Tensor),
                f"non-tensor output for {implementation_name}, causal={causal}",
            )
            require(
                raw_actual.is_cuda,
                f"non-CUDA output for {implementation_name}, causal={causal}",
            )
            output_shape = list(raw_actual.shape)
            expected_output_shape = list(shape)
            require(
                output_shape == expected_output_shape,
                f"output shape {output_shape} != {expected_output_shape} for "
                f"{implementation_name}, causal={causal}",
            )
            output_dtype = str(raw_actual.dtype).removeprefix("torch.")
            expected_output_dtype = case["dtype"]
            require(
                raw_actual.dtype == dtype,
                f"output dtype {output_dtype} != {expected_output_dtype} for "
                f"{implementation_name}, causal={causal}",
            )
            actual = raw_actual.float()
            require(
                bool(torch.isfinite(actual).all()),
                f"non-finite output for {implementation_name}, causal={causal}",
            )
            reference_flat = references[causal].flatten()
            actual_flat = actual.flatten()
            cosine = float(
                torch.dot(reference_flat, actual_flat)
                / (torch.linalg.vector_norm(reference_flat)
                   * torch.linalg.vector_norm(actual_flat)).clamp_min(1e-12)
            )
            relative_l2 = float(
                torch.linalg.vector_norm(actual_flat - reference_flat)
                / torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
            )
            require(
                cosine >= policy["minimum_cosine_similarity"],
                f"cosine {cosine:.6f} below threshold for "
                f"{implementation_name}, causal={causal}",
            )
            require(
                relative_l2 <= policy["maximum_relative_l2"],
                f"relative L2 {relative_l2:.6f} above threshold for "
                f"{implementation_name}, causal={causal}",
            )
            results.append({
                "causal": causal,
                "cosine_similarity": cosine,
                "expected_output_dtype": expected_output_dtype,
                "expected_output_shape": expected_output_shape,
                "implementation": implementation_name,
                "output_dtype": output_dtype,
                "output_shape": output_shape,
                "relative_l2": relative_l2,
            })

    runtime_image_ref = os.environ.get("RUNTIME_IMAGE_REF", "").strip()
    require(bool(runtime_image_ref), "RUNTIME_IMAGE_REF must contain the selected immutable image ref")
    report = {
        "build_id": build["id"],
        "status": "pass",
        "wheel_asset": artifact["asset"],
        "wheel_sha256": artifact["sha256"],
        "expected_compute_capability": expected_capability,
        "actual_compute_capability": actual_capability,
        "cuda_device_name": torch.cuda.get_device_name(0),
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, report_path)
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
            "patch_sha256",
            "resource_end",
            "resource_start",
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
        selected = build_evidence["selected_parallelism"]
        exceeds_default = (
            selected.get("max_jobs", 0) > matrix["resources"]["default_max_jobs"]
            or selected.get("extension_parallelism", 0)
            > matrix["resources"]["default_extension_parallelism"]
        )
        require(not exceeds_default or selected.get("unsafe_override") is True,
                "build evidence exceeds reviewed parallelism without explicit override")
        peak_end = build_evidence["cgroup_peak"].get("end_bytes")
        require(isinstance(peak_end, int) and peak_end > 0,
                "build evidence has no measured cgroup peak")

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
