from __future__ import annotations

import ast
import io
from dataclasses import replace
from datetime import datetime, timezone
import json
import subprocess
import tarfile
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

from tools.runpod_job import (
    CapacityUnavailableError,
    CommandError,
    Deadline,
    Endpoint,
    JobError,
    JobSpec,
    PodRequest,
    Runpodctl,
    SSHTransport,
    _extract_ssh_endpoint,
    _http_json,
    _make_repo_archive,
    _parser,
    _rfc3339_after,
    _safe_extract,
    run_job,
    verify_pod_assignment,
    wait_for_ssh,
)


class RecordingExecutor:
    def __init__(self, *responses: subprocess.CompletedProcess[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected subprocess call: {argv}")
        return self.responses.pop(0)


class RecordingHttp:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[
            tuple[str, str, dict[str, object] | None, dict[str, str], float]
        ] = []

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append(
            (method, url, dict(payload) if payload is not None else None, dict(headers), timeout)
        )
        return self.response


def completed(stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class RunpodctlTests(unittest.TestCase):
    def test_http_get_uses_no_json_body(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"ok":true}'

        with mock.patch(
            "tools.runpod_job.urllib_request.urlopen",
            return_value=Response(),
        ) as urlopen:
            self.assertEqual(
                _http_json("GET", "https://example.invalid/pod", None, {}, 5),
                {"ok": True},
            )
        request = urlopen.call_args.args[0]
        self.assertIsNone(request.data)

    def test_cpu_create_sets_public_key_and_never_adds_gpu_flags(self) -> None:
        executor = RecordingExecutor()
        http = RecordingHttp({"id": "pod-cpu"})
        ctl = Runpodctl(
            executor=executor,
            http_executor=http,
            env={"RUNPOD_API_KEY": "test-secret"},
        )
        pod_id = ctl.create_pod(
            PodRequest(
                image="registry.invalid/builder@sha256:" + "a" * 64,
                name="cpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="CPU",
            ),
            terminate_after="2026-08-18T05:00:00Z",
            self_terminate_seconds=15_300,
        )

        self.assertEqual(pod_id, "pod-cpu")
        self.assertEqual(executor.calls, [])
        method, url, payload, headers, _ = http.calls[0]
        assert payload is not None
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://rest.runpod.io/v1/pods")
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertEqual(payload["computeType"], "CPU")
        self.assertEqual(payload["containerDiskInGb"], 80)
        self.assertEqual(payload["volumeInGb"], 0)
        self.assertEqual(payload["volumeMountPath"], "/workspace")
        self.assertEqual(payload["cpuFlavorIds"], ["cpu3g"])
        self.assertEqual(payload["vcpuCount"], 16)
        self.assertTrue(payload["supportPublicIp"])
        self.assertEqual(payload["ports"], ["22/tcp"])
        self.assertEqual(
            payload["env"],
            {
                "PUBLIC_KEY": "ssh-ed25519 AAAA test",
                "RUNPOD_SELF_TERMINATE_SECONDS": "15300",
            },
        )
        self.assertNotIn("gpuTypeIds", payload)

    def test_cpu_request_requires_measured_container_disk_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 80 GB container disk"):
            PodRequest(
                image="registry.invalid/builder",
                name="undersized-cpu-build",
                public_key="ssh-ed25519 AAAA test",
                container_disk_gb=79,
            ).validate()

    def test_gpu_create_requires_exact_id_and_forwards_cuda_floor(self) -> None:
        executor = RecordingExecutor(completed('{"pod":{"id":"pod-gpu"}}'))
        ctl = Runpodctl(executor=executor)
        pod_id = ctl.create_pod(
            PodRequest(
                image="registry.invalid/test@sha256:" + "b" * 64,
                name="gpu-test",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id="NVIDIA RTX 4090",
                min_cuda_version="12.8",
            ),
            terminate_after="2026-08-18T05:00:00Z",
            self_terminate_seconds=15_300,
        )

        self.assertEqual(pod_id, "pod-gpu")
        argv = executor.calls[0][0]
        self.assertEqual(argv[argv.index("--gpu-id") + 1], "NVIDIA RTX 4090")
        self.assertEqual(argv[argv.index("--gpu-count") + 1], "1")
        self.assertEqual(argv[argv.index("--min-cuda-version") + 1], "12.8")
        self.assertNotIn("--compute-type", argv)
        self.assertNotIn("--volume-in-gb", argv)
        self.assertIn("--public-ip", argv)
        self.assertEqual(
            argv[argv.index("--terminate-after") + 1],
            "2026-08-18T05:00:00Z",
        )

    def test_gpu_build_create_preserves_deadline_and_disables_volume(self) -> None:
        executor = RecordingExecutor(completed('{"id":"pod-gpu-build"}'))
        ctl = Runpodctl(executor=executor)
        pod_id = ctl.create_pod(
            PodRequest(
                image="registry.invalid/builder@sha256:" + "e" * 64,
                name="gpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id="NVIDIA A100 80GB PCIe",
                min_cuda_version="12.8",
                gpu_workload="BUILD",
                gpu_min_vcpu_count=4,
                gpu_min_memory_gb=32,
                container_disk_gb=80,
            ),
            terminate_after="2026-08-18T05:00:00Z",
            self_terminate_seconds=15_300,
        )

        self.assertEqual(pod_id, "pod-gpu-build")
        argv = executor.calls[0][0]
        self.assertEqual(
            argv[argv.index("--gpu-id") + 1], "NVIDIA A100 80GB PCIe"
        )
        self.assertEqual(argv[argv.index("--container-disk-in-gb") + 1], "80")
        self.assertEqual(argv[argv.index("--volume-in-gb") + 1], "0")
        self.assertEqual(
            argv[argv.index("--terminate-after") + 1],
            "2026-08-18T05:00:00Z",
        )

    def test_gpu_create_classifies_only_explicit_capacity_failure(self) -> None:
        request = PodRequest(
            image="registry.invalid/builder@sha256:" + "e" * 64,
            name="gpu-build",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id="NVIDIA A100 80GB PCIe",
            gpu_workload="BUILD",
        )
        capacity = completed(
            stderr=(
                "Error: There are no longer any instances available with the "
                "requested specifications."
            ),
            code=1,
        )
        with self.assertRaises(CapacityUnavailableError):
            Runpodctl(executor=RecordingExecutor(capacity)).create_pod(
                request,
                terminate_after="2026-08-18T05:00:00Z",
                self_terminate_seconds=15_300,
            )

        unauthorized = completed(stderr="Error: unauthorized", code=1)
        with self.assertRaises(CommandError) as raised:
            Runpodctl(executor=RecordingExecutor(unauthorized)).create_pod(
                request,
                terminate_after="2026-08-18T05:00:00Z",
                self_terminate_seconds=15_300,
            )
        self.assertNotIsInstance(raised.exception, CapacityUnavailableError)

    def test_gpu_build_request_enforces_compilation_minima(self) -> None:
        request = PodRequest(
            image="registry.invalid/builder",
            name="gpu-build",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id="NVIDIA A100 80GB PCIe",
            gpu_workload="BUILD",
        )
        for field, value, message in (
            ("gpu_min_vcpu_count", 3, "at least 4 vCPUs"),
            ("gpu_min_memory_gb", 31, "at least 32 GB system RAM"),
            ("container_disk_gb", 79, "at least 80 GB container disk"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, message
            ):
                replace(request, **{field: value}).validate()

    def test_gpu_build_resource_flags_are_exposed_by_cli(self) -> None:
        args = _parser().parse_args(
            [
                "--image",
                "registry.invalid/builder",
                "--name",
                "gpu-build",
                "--mode",
                "gpu",
                "--gpu-id",
                "NVIDIA A100 80GB PCIe",
                "--gpu-id",
                "NVIDIA H100 PCIe",
                "--gpu-workload",
                "build",
                "--gpu-min-vcpu-count",
                "24",
                "--gpu-min-memory-gb",
                "64",
                "--ssh-key",
                "id_ed25519",
                "--command",
                "true",
            ]
        )
        self.assertEqual(args.gpu_workload, "build")
        self.assertEqual(
            args.gpu_id,
            ["NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe"],
        )
        self.assertEqual(args.gpu_min_vcpu_count, 24)
        self.assertEqual(args.gpu_min_memory_gb, 64)
        parser = _parser()
        self.assertEqual(parser.get_default("gpu_min_vcpu_count"), 4)
        help_text = " ".join(parser.format_help().split())
        self.assertIn("4 required; 16 recommended", help_text)
        self.assertIn("32 GB required; 64 GB recommended", help_text)

    def test_nonzero_runpodctl_result_is_an_error(self) -> None:
        ctl = Runpodctl(executor=RecordingExecutor(completed(stderr="denied", code=1)))
        with self.assertRaisesRegex(CommandError, "denied"):
            ctl.check_auth()

    def test_invalid_json_is_an_error(self) -> None:
        ctl = Runpodctl(executor=RecordingExecutor(completed("not-json")))
        with self.assertRaisesRegex(CommandError, "invalid JSON"):
            ctl.pod_details("pod-1")

    def test_pinned_cli_termination_backstop_uses_rfc3339_datetime(self) -> None:
        now = datetime(2026, 8, 18, 0, 45, tzinfo=timezone.utc)
        self.assertEqual(
            _rfc3339_after(14_400 + 900, now=now),
            "2026-08-18T05:00:00Z",
        )
        with self.assertRaises(ValueError):
            _rfc3339_after(0, now=now)

    def test_cpu_rest_forwards_registry_and_exact_placement_inputs(self) -> None:
        http = RecordingHttp({"id": "pod-cpu"})
        ctl = Runpodctl(
            executor=RecordingExecutor(),
            http_executor=http,
            env={"RUNPOD_API_KEY": "secret"},
        )
        ctl.create_pod(
            PodRequest(
                image="registry.invalid/builder@sha256:" + "d" * 64,
                name="cpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="CPU",
                cloud_type="SECURE",
                registry_auth_id="registry-auth",
                data_center_ids="EU-RO-1, CA-MTL-1",
                cpu_flavor_ids=("cpu3g", "cpu5g"),
                cpu_vcpu_count=16,
            ),
            terminate_after="2026-08-18T05:00:00Z",
            self_terminate_seconds=15_300,
        )
        payload = http.calls[0][2]
        assert payload is not None
        self.assertEqual(payload["containerRegistryAuthId"], "registry-auth")
        self.assertEqual(payload["dataCenterIds"], ["EU-RO-1", "CA-MTL-1"])
        self.assertEqual(payload["cpuFlavorIds"], ["cpu3g", "cpu5g"])
        self.assertFalse(payload["supportPublicIp"])

    def test_assignment_get_has_no_request_body(self) -> None:
        http = RecordingHttp({"id": "pod-1", "image": "example/image:tag"})
        ctl = Runpodctl(
            executor=RecordingExecutor(),
            http_executor=http,
            env={"RUNPOD_API_KEY": "secret"},
        )
        self.assertEqual(ctl.assignment_details("pod-1")["id"], "pod-1")
        method, url, payload, _, _ = http.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(url, "https://rest.runpod.io/v1/pods/pod-1")
        self.assertIsNone(payload)


class EndpointTests(unittest.TestCase):
    def test_extracts_current_ssh_shape(self) -> None:
        endpoint = _extract_ssh_endpoint({"ssh": {"ip": "203.0.113.1", "port": 22001}})
        self.assertEqual(endpoint, Endpoint("203.0.113.1", 22001))

    def test_extracts_runtime_port_shape(self) -> None:
        endpoint = _extract_ssh_endpoint(
            {
                "runtime": {
                    "ports": [
                        {"privatePort": 8888, "publicPort": 1234, "ip": "bad"},
                        {"privatePort": 22, "publicPort": 22002, "ip": "198.51.100.8"},
                    ]
                }
            }
        )
        self.assertEqual(endpoint, Endpoint("198.51.100.8", 22002))

    def test_extracts_official_rest_public_ip_shape(self) -> None:
        endpoint = _extract_ssh_endpoint(
            {"publicIp": "192.0.2.9", "portMappings": {"22": 22123}}
        )
        self.assertEqual(endpoint, Endpoint("192.0.2.9", 22123))

    def test_waits_for_endpoint_and_requires_successful_probe(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.responses = [
                    {"status": "RUNNING", "ssh": {}},
                    {"status": "RUNNING", "ssh": {"ip": "host", "port": 22}},
                ]

            def pod_details(self, pod_id, timeout):
                return self.responses.pop(0)

        class FakeTransport:
            def probe(self, endpoint, timeout):
                return True

        endpoint = wait_for_ssh(
            FakeCtl(),  # type: ignore[arg-type]
            FakeTransport(),  # type: ignore[arg-type]
            "pod-1",
            Deadline(10),
            poll_seconds=0.01,
            sleep=lambda _: None,
        )
        self.assertEqual(endpoint, Endpoint("host", 22))

    def test_terminal_state_stops_waiting(self) -> None:
        class FakeCtl:
            def pod_details(self, pod_id, timeout):
                return {"desiredStatus": "FAILED"}

        with self.assertRaisesRegex(JobError, "terminal state FAILED"):
            wait_for_ssh(
                FakeCtl(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                "pod-1",
                Deadline(10),
                sleep=lambda _: None,
            )


class AssignmentTests(unittest.TestCase):
    def test_cpu_assignment_is_bound_to_image_and_minimum_resources(self) -> None:
        class FakeCtl:
            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "a" * 64,
                    "cpuFlavorId": "cpu3g",
                    "vcpuCount": 16,
                    "memoryInGb": 62,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

        verify_pod_assignment(
            FakeCtl(),  # type: ignore[arg-type]
            "pod-1",
            PodRequest(
                image="registry.invalid/builder@sha256:" + "a" * 64,
                name="cpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="CPU",
            ),
            Deadline(10),
        )

    def test_cpu_assignment_rejects_service_drift(self) -> None:
        class FakeCtl:
            def assignment_details(self, pod_id, timeout):
                return {
                    "imageName": "wrong/image:latest",
                    "cpuFlavorId": "cpu3c",
                    "vcpuCount": 2,
                    "memoryInGb": 4,
                }

        with self.assertRaisesRegex(JobError, "Pod image mismatch"):
            verify_pod_assignment(
                FakeCtl(),  # type: ignore[arg-type]
                "pod-1",
                PodRequest(
                    image="registry.invalid/builder@sha256:" + "a" * 64,
                    name="cpu-build",
                    public_key="ssh-ed25519 AAAA test",
                    compute_type="CPU",
                ),
                Deadline(10),
            )

    def test_cpu_assignment_rejects_disk_or_volume_drift(self) -> None:
        base = {
            "imageName": "registry.invalid/builder@sha256:" + "a" * 64,
            "cpuFlavorId": "cpu3g",
            "vcpuCount": 16,
            "memoryInGb": 62,
            "containerDiskInGb": 80,
            "volumeInGb": 0,
        }

        class FakeCtl:
            def __init__(self, details):
                self.details = details

            def assignment_details(self, pod_id, timeout):
                return self.details

        request = PodRequest(
            image=base["imageName"],
            name="cpu-build",
            public_key="ssh-ed25519 AAAA test",
        )
        for field, value, message in (
            ("containerDiskInGb", 79, "container disk mismatch"),
            ("volumeInGb", 20, "unexpected paid Pod volume"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(JobError, message):
                verify_pod_assignment(
                    FakeCtl({**base, field: value}),  # type: ignore[arg-type]
                    "pod-1",
                    request,
                    Deadline(10),
                )

    def test_gpu_validation_assignment_retains_light_contract(self) -> None:
        class FakeCtl:
            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/runtime@sha256:" + "b" * 64,
                    "gpu": {"id": "NVIDIA RTX 4090"},
                }

        verify_pod_assignment(
            FakeCtl(),  # type: ignore[arg-type]
            "pod-validation",
            PodRequest(
                image="registry.invalid/runtime@sha256:" + "b" * 64,
                name="gpu-validation",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id="NVIDIA RTX 4090",
            ),
            Deadline(10),
        )

    def test_gpu_validation_assignment_requires_exact_gpu_id(self) -> None:
        class FakeCtl:
            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/runtime@sha256:" + "b" * 64,
                    "gpuTypeId": "NVIDIA L40S",
                }

        request = PodRequest(
            image="registry.invalid/runtime@sha256:" + "b" * 64,
            name="gpu-validation",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id="NVIDIA RTX 4090",
        )
        with self.assertRaisesRegex(JobError, "GPU type mismatch"):
            verify_pod_assignment(
                FakeCtl(),  # type: ignore[arg-type]
                "pod-validation",
                request,
                Deadline(10),
            )

    def test_gpu_build_assignment_accepts_both_gpu_id_shapes(self) -> None:
        base = {
            "image": "registry.invalid/builder@sha256:" + "e" * 64,
            "vcpuCount": 4,
            "memoryInGb": 32,
            "containerDiskInGb": 80,
            "volumeInGb": 0,
        }
        request = PodRequest(
            image=base["image"],
            name="gpu-build",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id="NVIDIA A100 80GB PCIe",
            gpu_workload="BUILD",
        )

        class FakeCtl:
            def __init__(self, details):
                self.details = details

            def assignment_details(self, pod_id, timeout):
                return self.details

        for gpu_fields in (
            {"gpuTypeId": "NVIDIA A100 80GB PCIe"},
            {"gpu": {"id": "NVIDIA A100 80GB PCIe"}},
        ):
            with self.subTest(gpu_fields=gpu_fields):
                verify_pod_assignment(
                    FakeCtl({**base, **gpu_fields}),  # type: ignore[arg-type]
                    "pod-gpu-build",
                    request,
                    Deadline(10),
                )

    def test_gpu_build_assignment_fails_closed_on_resource_drift(self) -> None:
        base = {
            "imageName": "registry.invalid/builder@sha256:" + "e" * 64,
            "gpuTypeId": "NVIDIA A100 80GB PCIe",
            "vcpuCount": 4,
            "memoryInGb": 32,
            "containerDiskInGb": 80,
            "volumeInGb": 0,
        }
        request = PodRequest(
            image=base["imageName"],
            name="gpu-build",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id="NVIDIA A100 80GB PCIe",
            gpu_workload="BUILD",
        )

        class FakeCtl:
            def __init__(self, details):
                self.details = details

            def assignment_details(self, pod_id, timeout):
                return self.details

        cases = (
            ({**base, "imageName": "wrong-image"}, "Pod image mismatch"),
            ({**base, "gpuTypeId": "NVIDIA A100-SXM4-80GB"}, "GPU type mismatch"),
            (
                {key: value for key, value in base.items() if key != "gpuTypeId"},
                "GPU type mismatch",
            ),
            ({**base, "vcpuCount": 3}, "GPU build CPU count mismatch"),
            (
                {key: value for key, value in base.items() if key != "vcpuCount"},
                "GPU build CPU count mismatch",
            ),
            ({**base, "memoryInGb": 31}, "GPU build memory mismatch"),
            (
                {key: value for key, value in base.items() if key != "memoryInGb"},
                "GPU build memory mismatch",
            ),
            ({**base, "containerDiskInGb": 79}, "container disk mismatch"),
            (
                {
                    key: value
                    for key, value in base.items()
                    if key != "containerDiskInGb"
                },
                "container disk mismatch",
            ),
            ({**base, "volumeInGb": 20}, "unexpected paid Pod volume"),
            (
                {key: value for key, value in base.items() if key != "volumeInGb"},
                "unexpected paid Pod volume",
            ),
        )
        for details, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                JobError, message
            ):
                verify_pod_assignment(
                    FakeCtl(details),  # type: ignore[arg-type]
                    "pod-gpu-build",
                    request,
                    Deadline(10),
                )


class ArtifactSafetyTests(unittest.TestCase):
    def test_safe_extract_accepts_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "ok.tar.gz"
            data = b"wheel"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("dist/example.whl")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            output = root / "out"
            output.mkdir()
            _safe_extract(archive_path, output)
            self.assertEqual((output / "dist/example.whl").read_bytes(), data)

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "bad.tar.gz"
            data = b"escape"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("../escape.txt")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(JobError, "escapes output"):
                _safe_extract(archive_path, output)
            self.assertFalse((root / "escape.txt").exists())

    def test_repo_archive_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "real.txt").write_text("data", encoding="utf-8")
            try:
                (root / "linked.txt").symlink_to(root / "real.txt")
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(JobError, "unsafe archive member"):
                _make_repo_archive(root)


class SSHSecurityTests(unittest.TestCase):
    def test_transport_uses_isolated_accept_new_known_hosts(self) -> None:
        executor = RecordingExecutor(completed("__runpod_ready__"))
        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp) / "key"
            key.write_text("private", encoding="utf-8")
            transport = SSHTransport(key, executor=executor)
            known_hosts = transport.known_hosts
            try:
                self.assertTrue(transport.probe(Endpoint("host", 22), timeout=2))
                argv = executor.calls[0][0]
                self.assertIn("IdentitiesOnly=yes", argv)
                self.assertIn("StrictHostKeyChecking=accept-new", argv)
                self.assertIn(f"UserKnownHostsFile={known_hosts}", argv)
                self.assertNotIn("UserKnownHostsFile=/dev/null", argv)
            finally:
                transport.close()
            self.assertFalse(known_hosts.exists())

    def test_upload_archive_stays_on_selected_remote_filesystem(self) -> None:
        executor = RecordingExecutor(completed(), completed(), completed())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            key = root / "key"
            key.write_text("private", encoding="utf-8")
            (root / "matrix.json").write_text("{}\n", encoding="utf-8")
            transport = SSHTransport(key, executor=executor)
            try:
                transport.upload_repo(
                    Endpoint("host", 22),
                    root,
                    "/work/sageattention-factory",
                    Deadline(30),
                )
            finally:
                transport.close()

        create_dir, upload, extract = (call[0] for call in executor.calls)
        self.assertEqual(create_dir[0], "ssh")
        self.assertIn("/work/sageattention-factory", create_dir[-1])
        self.assertEqual(upload[0], "scp")
        self.assertIn(
            "root@host:/work/sageattention-factory/.sageattention-source-",
            upload[-1],
        )
        self.assertNotIn("root@host:/tmp/", upload[-1])
        self.assertEqual(extract[0], "ssh")
        self.assertIn("/work/sageattention-factory", extract[-1])


class WorkflowInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.build = (root / ".github/workflows/build.yml").read_text(
            encoding="utf-8"
        )
        cls.release = (root / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )

    def test_cpu_build_uses_sized_container_disk_filesystem(self) -> None:
        self.assertIn("container_disk_gb must be between 80 and 200", self.build)
        self.assertIn("--remote-dir /work/sageattention-factory", self.build)
        self.assertIn("TMPDIR=/work/tmp", self.build)
        self.assertIn(
            "SAGEATTN_WORK_ROOT=/work/sageattention-wheel-builds", self.build
        )
        self.assertEqual(self.build.count("--remote-dir /workspace"), 1)

    def test_workflow_freezes_source_sha_for_every_paid_job(self) -> None:
        self.assertIn(
            "source_ref: ${{ needs.preflight.outputs.source_sha }}", self.release
        )
        self.assertNotIn("source_ref: ${{ inputs.release_tag }}", self.release)
        self.assertEqual(
            self.build.count("ref: ${{ needs.plan.outputs.source_sha }}"), 2
        )
        self.assertIn(
            'git rev-parse --verify "refs/tags/${RELEASE_TAG}^{commit}"',
            self.release,
        )
        self.assertIn('[[ "${RELEASE_TAG}" == -* ]]', self.release)
        self.assertIn(
            '"source_commit": os.environ["RESOLVED_SOURCE_SHA"]', self.release
        )
        self.assertIn(
            '"${current_commit}" != "${RESOLVED_SOURCE_SHA}"', self.release
        )

    def test_embedded_workflow_python_is_syntactically_valid(self) -> None:
        block_count = 0
        for workflow in (self.build, self.release):
            lines = workflow.splitlines()
            index = 0
            while index < len(lines):
                if "python3.12 - <<'PY'" not in lines[index]:
                    index += 1
                    continue
                body: list[str] = []
                index += 1
                while index < len(lines) and lines[index].strip() != "PY":
                    body.append(lines[index])
                    index += 1
                self.assertLess(index, len(lines), "unterminated workflow heredoc")
                ast.parse(textwrap.dedent("\n".join(body)))
                block_count += 1
                index += 1
        self.assertGreaterEqual(block_count, 4)


class RunJobCleanupTests(unittest.TestCase):
    def _spec(self, root: Path, *, allow_paid: bool = True) -> JobSpec:
        key = root / "id_ed25519"
        key.write_text("private-key", encoding="utf-8")
        return JobSpec(
            pod=PodRequest(
                image="registry.invalid/builder@sha256:" + "c" * 64,
                name="bounded-job",
                public_key="ssh-ed25519 AAAA test",
            ),
            repo_root=root,
            ssh_private_key=key,
            remote_dir="/work/sageattention-factory",
            commands=("bash scripts/build-wheel.sh",),
            artifact_paths=("dist/build",),
            artifact_output=root / "artifacts",
            allow_paid_pod=allow_paid,
        )

    def _gpu_spec(
        self,
        root: Path,
        candidates: tuple[str, ...] = (
            "NVIDIA A100 80GB PCIe",
            "NVIDIA H100 PCIe",
        ),
    ) -> JobSpec:
        pod = PodRequest(
            image="registry.invalid/builder@sha256:" + "e" * 64,
            name="gpu-build",
            public_key="ssh-ed25519 AAAA test",
            compute_type="GPU",
            gpu_id=candidates[0],
            gpu_workload="BUILD",
        )
        return replace(
            self._spec(root),
            pod=pod,
            gpu_id_candidates=candidates,
        )

    def test_paid_guard_fails_before_auth_or_create(self) -> None:
        class FakeCtl:
            called = False

            def check_auth(self, timeout):
                self.called = True

        with tempfile.TemporaryDirectory() as temp:
            ctl = FakeCtl()
            with self.assertRaisesRegex(ValueError, "paid Pod creation is disabled"):
                run_job(self._spec(Path(temp), allow_paid=False), ctl=ctl)  # type: ignore[arg-type]
            self.assertFalse(ctl.called)

    def test_cpu_checkout_must_be_below_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            spec = replace(self._spec(Path(temp)), remote_dir="/workspace")
            with self.assertRaisesRegex(ValueError, "directory below /work"):
                spec.validate()

    def test_gpu_build_checkout_must_be_below_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pod = PodRequest(
                image="registry.invalid/builder@sha256:" + "e" * 64,
                name="gpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id="NVIDIA A100 80GB PCIe",
                gpu_workload="BUILD",
            )
            spec = replace(self._spec(root), pod=pod, remote_dir="/workspace")
            with self.assertRaisesRegex(ValueError, "directory below /work"):
                spec.validate()

    def test_gpu_build_assignment_failure_terminates_before_upload(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.terminate_after = terminate_after
                return "pod-gpu-build"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": "NVIDIA A100 80GB PCIe"},
                    "vcpuCount": 3,
                    "memoryInGb": 64,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class NoUploadTransport:
            def upload_repo(self, *args):
                raise AssertionError("resource drift must fail before source upload")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pod = PodRequest(
                image="registry.invalid/builder@sha256:" + "e" * 64,
                name="gpu-build",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id="NVIDIA A100 80GB PCIe",
                gpu_workload="BUILD",
            )
            spec = replace(self._spec(root), pod=pod)
            ctl = FakeCtl()
            with self.assertRaisesRegex(JobError, "GPU build CPU count mismatch"):
                run_job(
                    spec,
                    ctl=ctl,  # type: ignore[arg-type]
                    transport=NoUploadTransport(),  # type: ignore[arg-type]
                    wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
                )
            self.assertRegex(ctl.terminate_after, r"Z$")
            self.assertEqual(ctl.terminated, ["pod-gpu-build"])

    def test_gpu_build_rejects_and_deletes_before_next_candidate(self) -> None:
        first = "NVIDIA A100 80GB PCIe"
        second = "NVIDIA H100 PCIe"
        events: list[tuple[str, str]] = []

        class FakeCtl:
            def __init__(self):
                self.deadlines: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.deadlines.append(terminate_after)
                events.append(("create", request.gpu_id))
                return "pod-first" if request.gpu_id == first else "pod-second"

            def assignment_details(self, pod_id, timeout):
                gpu_id = first if pod_id == "pod-first" else second
                return {
                    "imageName": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpuTypeId": gpu_id,
                    "vcpuCount": 3 if pod_id == "pod-first" else 4,
                    "memoryInGb": 32,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                events.append(("delete", pod_id))

        class FakeTransport:
            def __init__(self):
                self.scripts: list[str] = []

            def upload_repo(self, endpoint, repo_root, remote_dir, deadline):
                events.append(("upload", endpoint.host))

            def run_script(self, endpoint, remote_dir, script, deadline):
                self.scripts.append(script)
                events.append(("command", endpoint.host))

            def download_artifacts(self, *args):
                events.append(("download", "artifacts"))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctl = FakeCtl()
            transport = FakeTransport()
            result = run_job(
                self._gpu_spec(root, (first, second)),
                ctl=ctl,  # type: ignore[arg-type]
                transport=transport,  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        self.assertEqual(result.selected_gpu_id, second)
        self.assertEqual(ctl.deadlines[0], ctl.deadlines[1])
        self.assertRegex(ctl.deadlines[0], r"Z$")
        self.assertLess(
            events.index(("delete", "pod-first")),
            events.index(("create", second)),
        )
        self.assertLess(
            events.index(("create", second)),
            events.index(("upload", "host")),
        )
        self.assertEqual(events[-1], ("delete", "pod-second"))
        self.assertIn(
            "export RUNPOD_SELECTED_GPU_ID='NVIDIA H100 PCIe';",
            transport.scripts[0],
        )

    def test_gpu_validation_exact_assignment_mismatch_uses_next_candidate(self) -> None:
        first = "NVIDIA GeForce RTX 4090"
        second = "NVIDIA L40S"

        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.candidates.append(request.gpu_id)
                return f"pod-{len(self.candidates)}"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/runtime@sha256:" + "f" * 64,
                    "gpuTypeId": "unexpected" if pod_id == "pod-1" else second,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class FakeTransport:
            def upload_repo(self, *args):
                return None

            def run_script(self, *args):
                return None

            def download_artifacts(self, *args):
                return None

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pod = PodRequest(
                image="registry.invalid/runtime@sha256:" + "f" * 64,
                name="gpu-validation",
                public_key="ssh-ed25519 AAAA test",
                compute_type="GPU",
                gpu_id=first,
            )
            spec = replace(
                self._spec(root),
                pod=pod,
                gpu_id_candidates=(first, second),
                remote_dir="/workspace",
            )
            ctl = FakeCtl()
            result = run_job(
                spec,
                ctl=ctl,  # type: ignore[arg-type]
                transport=FakeTransport(),  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        self.assertEqual(ctl.candidates, [first, second])
        self.assertEqual(ctl.terminated, ["pod-1", "pod-2"])
        self.assertEqual(result.selected_gpu_id, second)

    def test_gpu_capacity_fallback_uses_two_bounded_ordered_rounds(self) -> None:
        first = "NVIDIA A100 80GB PCIe"
        second = "NVIDIA H100 PCIe"

        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []
                self.deadlines: list[str] = []
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.candidates.append(request.gpu_id)
                self.deadlines.append(terminate_after)
                if len(self.candidates) <= 2:
                    raise CapacityUnavailableError(
                        "There are no longer any instances available with the "
                        "requested specifications"
                    )
                return "pod-selected"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": first},
                    "vcpuCount": 4,
                    "memoryInGb": 32,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class FakeTransport:
            def upload_repo(self, *args):
                return None

            def run_script(self, *args):
                return None

            def download_artifacts(self, *args):
                return None

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "tools.runpod_job.time.sleep"
        ) as sleep:
            ctl = FakeCtl()
            result = run_job(
                self._gpu_spec(Path(temp), (first, second)),
                ctl=ctl,  # type: ignore[arg-type]
                transport=FakeTransport(),  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        self.assertEqual(ctl.candidates, [first, second, first])
        self.assertEqual(len(set(ctl.deadlines)), 1)
        self.assertRegex(ctl.deadlines[0], r"Z$")
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 5.0)
        self.assertEqual(result.selected_gpu_id, first)
        self.assertEqual(ctl.terminated, ["pod-selected"])

    def test_gpu_candidate_fallback_does_not_retry_configuration_error(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.candidates.append(request.gpu_id)
                raise CommandError("runpodctl pod create failed: unauthorized")

        with tempfile.TemporaryDirectory() as temp:
            ctl = FakeCtl()
            with self.assertRaisesRegex(JobError, "unauthorized"):
                run_job(
                    self._gpu_spec(Path(temp)),
                    ctl=ctl,  # type: ignore[arg-type]
                    transport=object(),  # type: ignore[arg-type]
                )
        self.assertEqual(ctl.candidates, ["NVIDIA A100 80GB PCIe"])

    def test_gpu_candidate_fallback_stops_when_upload_begins(self) -> None:
        first = "NVIDIA A100 80GB PCIe"

        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.candidates.append(request.gpu_id)
                return "pod-first"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpuTypeId": first,
                    "vcpuCount": 4,
                    "memoryInGb": 32,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class UploadFailure:
            def upload_repo(self, *args):
                raise CommandError("source upload failed")

        with tempfile.TemporaryDirectory() as temp:
            ctl = FakeCtl()
            with self.assertRaisesRegex(JobError, "source upload failed"):
                run_job(
                    self._gpu_spec(Path(temp)),
                    ctl=ctl,  # type: ignore[arg-type]
                    transport=UploadFailure(),  # type: ignore[arg-type]
                    wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
                )
        self.assertEqual(ctl.candidates, [first])
        self.assertEqual(ctl.terminated, ["pod-first"])

    def test_command_failure_still_downloads_debug_artifacts_and_terminates(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                self.terminate_after = terminate_after
                return "pod-created"

            def assignment_details(self, pod_id, timeout):
                return {
                    "imageName": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpuFlavorId": "cpu3g",
                    "vcpuCount": 16,
                    "memoryInGb": 62,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class FakeTransport:
            def __init__(self):
                self.downloaded = False

            def upload_repo(self, endpoint, repo_root, remote_dir, deadline):
                return None

            def run_script(self, endpoint, remote_dir, script, deadline):
                raise CommandError("nvcc failed")

            def download_artifacts(
                self, endpoint, remote_dir, artifact_paths, output_dir, deadline
            ):
                self.downloaded = True

        with tempfile.TemporaryDirectory() as temp:
            ctl = FakeCtl()
            transport = FakeTransport()
            with self.assertRaisesRegex(JobError, "nvcc failed"):
                run_job(
                    self._spec(Path(temp)),
                    ctl=ctl,  # type: ignore[arg-type]
                    transport=transport,  # type: ignore[arg-type]
                    wait_fn=lambda *args, **kwargs: Endpoint("host", 22),
                )
            self.assertTrue(transport.downloaded)
            self.assertEqual(ctl.terminated, ["pod-created"])

    def test_success_returns_result_and_terminates(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.terminated = False

            def check_auth(self, timeout):
                pass

            def create_pod(
                self, request, terminate_after, self_terminate_seconds, timeout
            ):
                return "pod-ok"

            def assignment_details(self, pod_id, timeout):
                return {
                    "imageName": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpuFlavorId": "cpu3g",
                    "vcpuCount": 16,
                    "memoryInGb": 62,
                    "containerDiskInGb": 80,
                    "volumeInGb": 0,
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated = True

        class FakeTransport:
            def upload_repo(self, *args):
                pass

            def run_script(self, *args):
                pass

            def download_artifacts(self, *args):
                pass

        with tempfile.TemporaryDirectory() as temp:
            ctl = FakeCtl()
            result = run_job(
                self._spec(Path(temp)),
                ctl=ctl,  # type: ignore[arg-type]
                transport=FakeTransport(),  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )
            self.assertEqual(result.pod_id, "pod-ok")
            self.assertEqual(result.endpoint.port, 2200)
            self.assertTrue(ctl.terminated)


if __name__ == "__main__":
    unittest.main()
