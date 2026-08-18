from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))


def assert_raises(exception, match: str):
    return unittest.TestCase().assertRaisesRegex(exception, match)


def load_script(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_build_matrix_matches_comfyui_base() -> None:
    assert MATRIX["schema_version"] == 1
    assert MATRIX["package"] == {
        "distribution": "sageattention",
        "upstream_version": "2.2.0",
        "source_url": "https://github.com/thu-ml/SageAttention.git",
        "source_commit": "eb615cf6cf4d221338033340ee2de1c37fbdba4a",
        "source_date_epoch": 1761613216,
    }
    assert MATRIX["platform"]["base_image"] == "ubuntu:24.04"
    assert MATRIX["platform"]["python_version"] == "3.12"
    assert MATRIX["platform"]["python_tag"] == "cp312"

    builds = {entry["id"]: entry for entry in MATRIX["builds"]}
    assert set(builds) == {
        "cp312-torch2.10.0-cu128",
        "cp312-torch2.10.0-cu130",
    }
    expected = {
        "cp312-torch2.10.0-cu128": (
            "12.8", "2.10.0+cu128", "12.8",
            "2.2.0+torch2.10.0.cu128",
            "runpod/comfyui@sha256:ce5e842ca0c7233a983ff76a83739b445172259c77a43a117453ef7e6a64d0b7",
            "madiatorlabs/sageattention-wheel-builder@sha256:556979b41a98e5331ad38a1ae599161cb37a78cb09bec34e3f982978b84816d0",
        ),
        "cp312-torch2.10.0-cu130": (
            "13.0", "2.10.0+cu130", "13.0",
            "2.2.0+torch2.10.0.cu130",
            "runpod/comfyui@sha256:0bf75436da591e0f26d299af3741e07cb8ce8ce36566d1a7d8d78aae458e5d67",
            "madiatorlabs/sageattention-wheel-builder@sha256:bed84f32bf76ba292728faf9fbe446d5d7f84d194444b2b6d006e663cccba906",
        ),
    }
    for build_id, values in expected.items():
        build = builds[build_id]
        assert (
            build["cuda_version"],
            build["torch_version"],
            build["torch_cuda_version"],
            build["wheel_version"],
            build["comfyui_runtime_image"],
            build["builder_image"],
        ) == values
        assert build["minimum_cuda_scheduler_version"] == build["cuda_version"]
        assert build["wheel_filename"] == (
            f"sageattention-{build['wheel_version']}-cp312-cp312-linux_x86_64.whl")


def test_api_complete_native_sass_matrix() -> None:
    policy = MATRIX["cuda_policy"]
    assert policy["native_sass_only"] is True
    assert policy["include_ptx"] is False
    assert policy["torch_cuda_arch_list"] == "8.0;8.6;8.9;9.0;12.0"
    assert "+PTX" not in policy["torch_cuda_arch_list"]
    assert policy["extension_compile_targets"] == {
        "sageattention._qattn_sm80": ["sm_80", "sm_86", "sm_89", "sm_90a", "sm_120"],
        "sageattention._qattn_sm89": ["sm_89", "sm_90a", "sm_120"],
        "sageattention._qattn_sm90": ["sm_90a"],
        "sageattention._fused": ["sm_80", "sm_86", "sm_89", "sm_90a", "sm_120"],
    }
    assert policy["extension_cubin_architectures"]["sageattention._qattn_sm90"] == ["sm_90"]
    for cubins in policy["extension_cubin_architectures"].values():
        assert "sm_90a" not in cubins


def test_resource_and_runtime_release_thresholds_are_conservative() -> None:
    resources = MATRIX["resources"]
    assert resources["gpu_required"] is False
    assert resources["minimum_memory_gib"] >= 32
    assert resources["recommended_memory_gib"] >= 64
    assert resources["compiler_memory_per_job_mib"] >= 8192
    assert resources["default_max_jobs"] <= 2
    assert resources["default_extension_parallelism"] == 1

    build_script = (ROOT / "scripts" / "build-wheel.sh").read_text(encoding="utf-8")
    assert 'shutil.disk_usage(os.environ["OUTPUT_DIR"])' in build_script
    assert 'shutil.disk_usage(os.environ["WORK_PARENT"])' in build_script
    assert 'if [[ "${WORK_PARENT}" == "/" ]]' in build_script

    runtime = MATRIX["validation"]["runtime_numeric"]
    assert runtime["minimum_cosine_similarity"] >= 0.995
    assert runtime["maximum_relative_l2"] <= 0.10
    assert runtime["canonical_case"]["causal_modes"] == [False, True]
    assert runtime["implementations_by_capability"] == {
        "8.0": ["sageattn_dispatch", "qattn_sm80_cuda"],
        "8.6": ["sageattn_dispatch", "qattn_sm80_cuda"],
        "8.9": ["sageattn_dispatch", "qattn_sm80_cuda", "qattn_sm89_cuda"],
        "9.0": [
            "sageattn_dispatch",
            "qattn_sm80_cuda",
            "qattn_sm89_cuda",
            "qattn_sm90_cuda",
        ],
        "12.0": ["sageattn_dispatch", "qattn_sm80_cuda", "qattn_sm89_cuda"],
    }
    capabilities = {
        entry["compute_capability"]
        for entry in MATRIX["validation"]["representative_gpu_capabilities"]
        if entry["required"]
    }
    assert capabilities == {"8.0", "8.6", "8.9", "9.0", "12.0"}

    validator = (ROOT / "scripts" / "validate-wheel.py").read_text(encoding="utf-8")
    for implementation in {
        "sageattn_dispatch",
        "qattn_sm80_cuda",
        "qattn_sm89_cuda",
        "qattn_sm90_cuda",
    }:
        assert f'"{implementation}"' in validator
    assert 'required_implementations = policy["implementations_by_capability"]' in validator
    assert "for implementation_name in required_implementations" in validator
    assert "for causal in case[\"causal_modes\"]" in validator
    assert validator.index("output_shape = list(raw_actual.shape)") < validator.index(
        "actual = raw_actual.float()")
    assert validator.index("raw_actual.dtype == dtype") < validator.index(
        "actual = raw_actual.float()")
    for result_field in {
        "expected_output_dtype",
        "expected_output_shape",
        "output_dtype",
        "output_shape",
    }:
        assert f'"{result_field}"' in validator

    release_workflow = (
        ROOT / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")
    for result_field in {
        "expected_output_dtype",
        "expected_output_shape",
        "output_dtype",
        "output_shape",
    }:
        assert f'"{result_field}": expected_' in release_workflow
    assert "runtime output tensor mismatch" in release_workflow


def test_downstream_patch_scopes_gencodes_and_exact_torch_metadata() -> None:
    patch = (
        ROOT / "patches" / "sageattention" / "2.2.0" / "setup.py.patch"
    ).read_text(encoding="utf-8")
    assert 'nvcc_flags_for("9.0")' in patch
    assert 'nvcc_flags_for("8.9", "9.0", "12.0")' in patch
    assert '"8.0", "8.6", "8.9", "9.0", "12.0"' in patch
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    assert 'code=compute_{num}' not in added_lines
    assert 'torch=={REQUIRED_TORCH_VERSION}' in patch
    assert "version=PACKAGE_VERSION" in patch


def fake_artifact(build: dict, digest: str = "0" * 64) -> dict:
    platform = MATRIX["platform"]
    return {
        "abi_tag": platform["abi_tag"],
        "asset": build["wheel_filename"],
        "build_id": build["id"],
        "cuda_version": build["cuda_version"],
        "platform_tag": platform["platform_tag"],
        "python_tag": platform["python_tag"],
        "sha256": digest,
        "size": 123,
        "supported_compute_capabilities": MATRIX["cuda_policy"]["compute_capabilities"],
        "torch_cuda_version": build["torch_cuda_version"],
        "torch_version": build["torch_version"],
    }


def test_selector_requires_one_exact_runtime_tuple() -> None:
    selector = load_script("select-wheel.py", "test_wheel_selector")
    builds = MATRIX["builds"]
    manifest = {"schema_version": 1, "artifacts": [fake_artifact(build) for build in builds]}
    environment = selector.Environment(
        "cp312", "cp312", "linux_x86_64", "2.10.0+cu128", "12.8")
    assert selector.select_artifact(manifest, environment)["build_id"].endswith("cu128")

    duplicate = {"schema_version": 1, "artifacts": [fake_artifact(builds[0])] * 2}
    with assert_raises(selector.SelectionError, "resolve to one wheel"):
        selector.select_artifact(duplicate, environment)

    mismatch = selector.Environment(
        "cp312", "cp312", "linux_x86_64", "2.10.0+cu128", "13.0")
    with assert_raises(selector.SelectionError, "found 0"):
        selector.select_artifact(manifest, mismatch)

    unsupported_gpu = selector.Environment(
        "cp312", "cp312", "linux_x86_64", "2.10.0+cu128", "12.8", ("10.0",))
    with assert_raises(selector.SelectionError, "does not support visible GPU"):
        selector.select_artifact(manifest, unsupported_gpu)


def test_selector_rejects_simple_index() -> None:
    selector = load_script("select-wheel.py", "test_wheel_selector_simple")
    artifact = fake_artifact(MATRIX["builds"][0])
    with assert_raises(selector.SelectionError, "simple-index"):
        selector.resolve_asset(
            artifact,
            manifest_path=ROOT / "missing-manifest.json",
            base_url="https://example.invalid/simple/sageattention/",
        )


def test_manifest_merge_rehashes_and_rejects_ambiguity(tmp_path: Path) -> None:
    merger = load_script("merge-manifests.py", "test_manifest_merger")
    paths = []
    for build in MATRIX["builds"]:
        directory = tmp_path / build["id"]
        directory.mkdir()
        wheel = directory / build["wheel_filename"]
        wheel.write_bytes(build["id"].encode("ascii"))
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        artifact = fake_artifact(build, digest)
        artifact["size"] = wheel.stat().st_size
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps({"schema_version": 1, "artifacts": [artifact]}),
            encoding="utf-8",
        )
        (directory / "SHA256SUMS").write_text(
            f"{digest}  {wheel.name}\n", encoding="ascii")
        paths.append(manifest)

    merged = merger.merge(paths)
    assert [entry["build_id"] for entry in merged["artifacts"]] == [
        "cp312-torch2.10.0-cu128",
        "cp312-torch2.10.0-cu130",
    ]
    with assert_raises(merger.MergeError, "Duplicate asset|runtime tuple|build ID"):
        merger.merge([paths[0], paths[0]])


def test_docker_bake_pins_both_toolchains() -> None:
    bake = (ROOT / "docker" / "docker-bake.hcl").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "Dockerfile.builder").read_text(encoding="utf-8")
    entrypoint = (ROOT / "docker" / "builder-entrypoint.sh").read_text(encoding="utf-8")
    for build in MATRIX["builds"]:
        assert build["builder_target"] in bake
        assert build["torch_version"] in bake
        assert build["cuda_package_suffix"] in bake
    assert "ubuntu:24.04" in dockerfile
    assert "libcublas-dev-${CUDA_VERSION_DASH}" in dockerfile
    assert "libcusolver-dev-${CUDA_VERSION_DASH}" in dockerfile
    assert "libcusparse-dev-${CUDA_VERSION_DASH}" in dockerfile
    assert "VIRTUAL_ENV=/opt/sageattention-builder-venv" in dockerfile
    assert '/usr/bin/python3.12 -m venv "${VIRTUAL_ENV}"' in dockerfile
    assert '"${VIRTUAL_ENV}/bin/python" -m pip install --upgrade' in dockerfile
    assert "pod_resources.py /usr/local/bin/pod-resources" in dockerfile
    assert "RUNPODCTL_SHA256=908f2210571e8a26a1cba6fb45f09556b34dcad3e1b20dd502df2adf7a57c169" in dockerfile
    assert "RUNPOD_SELF_TERMINATE_SECONDS" in entrypoint
    assert 'runpodctl pod delete "$2"' in entrypoint
    assert "RUNPOD_API_KEY" in entrypoint


def test_workflows_default_to_immutable_image_digests() -> None:
    build_workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8")
    release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8")
    for build in MATRIX["builds"]:
        builder_image = build["builder_image"]
        runtime_image = build["comfyui_runtime_image"]
        assert builder_image.startswith(
            "madiatorlabs/sageattention-wheel-builder@sha256:")
        assert runtime_image.startswith("runpod/comfyui@sha256:")
        for image in (builder_image, runtime_image):
            assert build_workflow.count(f'default: "{image}"') == 2
            assert release_workflow.count(f'default: "{image}"') == 1


def test_workflows_prefill_reviewed_runpod_gpu_ids() -> None:
    build_workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
        encoding="utf-8")
    release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8")
    defaults = (
        "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 PCIe,"
        "NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA H200,"
        "NVIDIA GeForce RTX 5090,NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB",
        "NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 3090",
        "NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,"
        "NVIDIA GeForce RTX 4090,NVIDIA L4",
        "NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA H200",
        "NVIDIA GeForce RTX 5090,NVIDIA RTX PRO 6000 Blackwell Server Edition,"
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition,"
        "NVIDIA RTX PRO 4500 Blackwell Server Edition,NVIDIA RTX PRO 4500 Blackwell",
    )
    for candidates in defaults:
        default = f'default: "{candidates}"'
        assert build_workflow.count(default) == 2
        assert release_workflow.count(default) == 1
    assert 'backend_args+=(--gpu-id "${candidate}")' in build_workflow
    assert 'gpu_candidate_args+=(--gpu-id "${candidate}")' in build_workflow
    assert "allowed_validation_gpu_ids" in build_workflow
    assert "cross capability family" in build_workflow
    assert "NVIDIA RTX A4500" not in build_workflow
    assert "NVIDIA RTX A4500" not in release_workflow
    assert build_workflow.count("default: GPU") == 2
    assert release_workflow.count("default: GPU") == 1
    assert "max-parallel: 1" in build_workflow


def test_build_hides_gpu_before_import_and_records_visibility() -> None:
    build_script = (ROOT / "scripts" / "build-wheel.sh").read_text(
        encoding="utf-8")
    validator = (ROOT / "scripts" / "validate-wheel.py").read_text(
        encoding="utf-8")
    assert build_script.index('export CUDA_VISIBLE_DEVICES=""') < build_script.index(
        "import torch")
    assert '"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")' in build_script
    assert 'build_evidence["cuda_visible_devices"] == ""' in validator
    assert 'os.environ.get("RUNPOD_SELECTED_GPU_ID") or None' in build_script
    assert '"selected_gpu_id": selected_gpu_id' in build_script
    assert 'build_evidence["selected_gpu_id"]' in validator


class MatrixTests(unittest.TestCase):
    def test_exact_build_matrix_matches_comfyui_base(self) -> None:
        test_exact_build_matrix_matches_comfyui_base()

    def test_api_complete_native_sass_matrix(self) -> None:
        test_api_complete_native_sass_matrix()

    def test_resource_and_runtime_release_thresholds_are_conservative(self) -> None:
        test_resource_and_runtime_release_thresholds_are_conservative()

    def test_downstream_patch_scopes_gencodes_and_exact_torch_metadata(self) -> None:
        test_downstream_patch_scopes_gencodes_and_exact_torch_metadata()

    def test_selector_requires_one_exact_runtime_tuple(self) -> None:
        test_selector_requires_one_exact_runtime_tuple()

    def test_selector_rejects_simple_index(self) -> None:
        test_selector_rejects_simple_index()

    def test_manifest_merge_rehashes_and_rejects_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_manifest_merge_rehashes_and_rejects_ambiguity(Path(directory))

    def test_docker_bake_pins_both_toolchains(self) -> None:
        test_docker_bake_pins_both_toolchains()

    def test_workflows_default_to_immutable_image_digests(self) -> None:
        test_workflows_default_to_immutable_image_digests()

    def test_workflows_prefill_reviewed_runpod_gpu_ids(self) -> None:
        test_workflows_prefill_reviewed_runpod_gpu_ids()

    def test_build_hides_gpu_before_import_and_records_visibility(self) -> None:
        test_build_hides_gpu_before_import_and_records_visibility()


if __name__ == "__main__":
    unittest.main()
