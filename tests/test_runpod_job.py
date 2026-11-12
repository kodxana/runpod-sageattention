from __future__ import annotations

import ast
import io
from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import unittest
from unittest import mock
from pathlib import Path
from urllib import error as urllib_error

from tools.runpod_job import (
    AUTH_PROBE_POD_ID,
    CapacityUnavailableError,
    CommandError,
    Deadline,
    Endpoint,
    JobError,
    JobSpec,
    PodRequest,
    RestApiError,
    RunpodApi,
    SSHTransport,
    VerifiedBuildAssignment,
    _extract_ssh_endpoint,
    _http_json,
    _make_repo_archive,
    _parser,
    _safe_extract,
    _start_placement_round,
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
    """Fake REST transport: returns (or raises) queued responses in order."""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[
            tuple[str, str, dict[str, object] | None, dict[str, str], float]
        ] = []

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append(
            (method, url, json.loads(json.dumps(payload)) if payload is not None else None, dict(headers), timeout)
        )
        if not self.responses:
            raise AssertionError(f"unexpected REST call: {method} {url}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


NO_CAPACITY = (
    "There are no longer any instances available with the requested "
    "specifications. Please refresh and try again."
)
# What an authorized key gets from check_auth's write probe.
NOT_FOUND = RestApiError("POST", 404, "pod not found")


def completed(stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def api(*responses: object, **env: str) -> tuple[RunpodApi, RecordingHttp]:
    http = RecordingHttp(*responses)
    return (
        RunpodApi(http_executor=http, env={"RUNPOD_API_KEY": "test-secret", **env}),
        http,
    )


CPU_REQUEST = PodRequest(
    image="registry.invalid/builder@sha256:" + "a" * 64,
    name="cpu-build",
    public_key="ssh-ed25519 AAAA test",
    compute_type="CPU",
)
GPU_VALIDATION_REQUEST = PodRequest(
    image="registry.invalid/test@sha256:" + "b" * 64,
    name="gpu-test",
    public_key="ssh-ed25519 AAAA test",
    compute_type="GPU",
    gpu_id="NVIDIA GeForce RTX 4090",
    min_cuda_version="12.8",
)
GPU_BUILD_REQUEST = PodRequest(
    image="registry.invalid/builder@sha256:" + "e" * 64,
    name="gpu-build",
    public_key="ssh-ed25519 AAAA test",
    compute_type="GPU",
    gpu_id="NVIDIA A100 80GB PCIe",
    min_cuda_version="12.8",
    gpu_workload="BUILD",
    gpu_min_vcpu_count=16,
    gpu_min_memory_gb=64,
    container_disk_gb=80,
)


class HttpTests(unittest.TestCase):
    @staticmethod
    def _response(body: bytes):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return body

        return Response()

    def test_http_get_uses_no_json_body(self) -> None:
        with mock.patch(
            "tools.runpod_job.urllib_request.urlopen",
            return_value=self._response(b'{"ok":true}'),
        ) as urlopen:
            self.assertEqual(
                _http_json("GET", "https://example.invalid/pod", None, {}, 5),
                {"ok": True},
            )
        self.assertIsNone(urlopen.call_args.args[0].data)

    def test_empty_body_is_no_content(self) -> None:
        # REST v2 answers a successful DELETE with 204 and no body.
        with mock.patch(
            "tools.runpod_job.urllib_request.urlopen",
            return_value=self._response(b""),
        ):
            self.assertIsNone(
                _http_json("DELETE", "https://example.invalid/pod", None, {}, 5)
            )

    def test_http_error_keeps_status_and_detail(self) -> None:
        error = urllib_error.HTTPError(
            "https://example.invalid/pods",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"detail":"no instances"}'),
        )
        with mock.patch(
            "tools.runpod_job.urllib_request.urlopen", side_effect=error
        ), self.assertRaises(RestApiError) as raised:
            _http_json("POST", "https://example.invalid/pods", {}, {}, 5)
        self.assertEqual(raised.exception.status, 400)
        self.assertIn("no instances", raised.exception.detail)

    def test_invalid_json_is_an_error(self) -> None:
        with mock.patch(
            "tools.runpod_job.urllib_request.urlopen",
            return_value=self._response(b"not-json"),
        ), self.assertRaisesRegex(CommandError, "invalid JSON"):
            _http_json("GET", "https://example.invalid/pod", None, {}, 5)


class RunpodApiTests(unittest.TestCase):
    def test_requests_use_v2_bearer_auth_and_explicit_agent(self) -> None:
        ctl, http = api({"items": []}, NOT_FOUND)
        ctl.check_auth()
        method, url, payload, headers, _ = http.calls[0]
        self.assertEqual((method, url), ("GET", "https://api.runpod.io/v2/pods?limit=1"))
        self.assertIsNone(payload)
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        # Cloudflare answers urllib's default agent with 403 error code 1010.
        self.assertNotIn("Python-urllib", headers.get("User-Agent", "Python-urllib"))

    def test_rejected_key_fails_the_auth_check(self) -> None:
        ctl, _ = api(RestApiError("GET", 401, "missing bearer token"))
        with self.assertRaisesRegex(CommandError, "HTTP 401"):
            ctl.check_auth()

    def test_auth_check_proves_write_access_without_side_effects(self) -> None:
        for answer in (
            RestApiError("POST", 404, "resource not found"),  # the v2 spec's body
            NOT_FOUND,
            RestApiError("POST", 400, "invalid pod id"),
        ):
            with self.subTest(status=answer.status, detail=answer.detail):
                ctl, http = api({"items": []}, answer)
                ctl.check_auth()
                method, url, payload, _, _ = http.calls[1]
                self.assertEqual(method, "POST")
                # v2 replaced v1's POST /pods/{id}/stop with /action.
                self.assertEqual(
                    url,
                    "https://api.runpod.io/v2/pods/"
                    f"{AUTH_PROBE_POD_ID}/action",
                )
                self.assertEqual(payload, {"action": "stop"})

    def test_read_only_key_fails_the_auth_check(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status):
                ctl, _ = api({"items": []}, RestApiError("POST", status, "denied"))
                with self.assertRaisesRegex(CommandError, "read/write Pod access"):
                    ctl.check_auth()

    def test_unexplained_probe_answer_fails_the_auth_check(self) -> None:
        # A 404 from an unknown route (or any other status) says nothing about
        # the key, so it must not pass as proof of write access.
        for answer in (
            RestApiError("POST", 404, "404 page not found"),
            RestApiError("POST", 405, "method not allowed"),
            RestApiError("POST", 409, "conflict"),
        ):
            with self.subTest(status=answer.status, detail=answer.detail):
                ctl, _ = api({"items": []}, answer)
                with self.assertRaisesRegex(CommandError, "cannot confirm"):
                    ctl.check_auth()

    def test_list_pods_accepts_a_bare_or_wrapped_list(self) -> None:
        pods = [{"id": "a", "name": "sage-test-x"}]
        for payload in (pods, {"items": pods}):
            with self.subTest(payload=type(payload).__name__):
                ctl, http = api(payload)
                self.assertEqual(ctl.list_pods(), pods)
                self.assertEqual(http.calls[0][:2], ("GET", "https://api.runpod.io/v2/pods"))
        ctl, _ = api({"error": "x"})
        with self.assertRaisesRegex(CommandError, "no list of Pods"):
            ctl.list_pods()

    def test_cpu_create_uses_v2_shape_and_arms_the_watchdog(self) -> None:
        ctl, http = api({"id": "pod-cpu"})
        self.assertEqual(
            ctl.create_pod(CPU_REQUEST, self_terminate_seconds=15_300), "pod-cpu"
        )
        method, url, payload, _, _ = http.calls[0]
        assert payload is not None
        self.assertEqual((method, url), ("POST", "https://api.runpod.io/v2/pods"))
        self.assertEqual(
            payload,
            {
                "name": "cpu-build",
                "image": CPU_REQUEST.image,
                "cloud": "COMMUNITY",
                "disk": 80,
                "ports": ["22/tcp"],
                "env": {
                    "PUBLIC_KEY": "ssh-ed25519 AAAA test",
                    "RUNPOD_SELF_TERMINATE_SECONDS": "15300",
                },
                "cpu": {"id": "cpu3g", "vcpuCount": 16},
            },
        )

    def test_cpu_create_walks_flavors_in_order_on_capacity(self) -> None:
        ctl, http = api(
            RestApiError("POST", 400, NO_CAPACITY),
            {"id": "pod-cpu"},
        )
        request = replace(CPU_REQUEST, cpu_flavor_ids=("cpu3g", "cpu5c"))
        self.assertEqual(
            ctl.create_pod(request, self_terminate_seconds=15_300), "pod-cpu"
        )
        self.assertEqual(
            [call[2]["cpu"]["id"] for call in http.calls], ["cpu3g", "cpu5c"]
        )

    def test_cpu_create_raises_capacity_when_every_flavor_is_full(self) -> None:
        ctl, _ = api(
            RestApiError("POST", 400, "first: " + NO_CAPACITY),
            RestApiError("POST", 400, "last: " + NO_CAPACITY),
        )
        request = replace(CPU_REQUEST, cpu_flavor_ids=("cpu3g", "cpu5c"))
        with self.assertRaisesRegex(CapacityUnavailableError, "last"):
            ctl.create_pod(request, self_terminate_seconds=15_300)

    def test_gpu_validation_create_carries_cuda_floor_without_build_extras(self) -> None:
        ctl, http = api({"id": "pod-gpu"})
        self.assertEqual(
            ctl.create_pod(GPU_VALIDATION_REQUEST, self_terminate_seconds=1_200),
            "pod-gpu",
        )
        payload = http.calls[0][2]
        assert payload is not None
        self.assertEqual(
            payload["gpu"],
            {"id": "NVIDIA GeForce RTX 4090", "count": 1, "minCudaVersion": "12.8"},
        )
        self.assertNotIn("cpu", payload)
        self.assertNotIn("mounts", payload)
        # The ComfyUI validation image has no watchdog to arm.
        self.assertEqual(payload["env"], {"PUBLIC_KEY": "ssh-ed25519 AAAA test"})

    def test_gpu_build_create_filters_placement_and_arms_the_watchdog(self) -> None:
        ctl, http = api({"id": "pod-gpu-build"})
        ctl.create_pod(GPU_BUILD_REQUEST, self_terminate_seconds=15_300)
        payload = http.calls[0][2]
        assert payload is not None
        self.assertEqual(
            payload["gpu"],
            {
                "id": "NVIDIA A100 80GB PCIe",
                "count": 1,
                "minCudaVersion": "12.8",
                "minVcpuCountPerGpu": 16,
                "minRamPerGpu": 64,
            },
        )
        self.assertEqual(payload["disk"], 80)
        self.assertNotIn("mounts", payload)
        self.assertEqual(payload["env"]["RUNPOD_SELF_TERMINATE_SECONDS"], "15300")

    def test_create_forwards_registry_data_centers_and_cloud(self) -> None:
        ctl, http = api({"id": "pod-cpu"})
        ctl.create_pod(
            replace(
                CPU_REQUEST,
                cloud_type="SECURE",
                registry_auth_id="registry-auth",
                data_center_ids="EU-RO-1, CA-MTL-1",
            ),
            self_terminate_seconds=15_300,
        )
        payload = http.calls[0][2]
        assert payload is not None
        self.assertEqual(payload["registry"], "registry-auth")
        self.assertEqual(payload["dataCenterIds"], ["EU-RO-1", "CA-MTL-1"])
        self.assertEqual(payload["cloud"], "SECURE")

    def test_non_production_api_url_reaches_requests_and_watchdog(self) -> None:
        ctl, http = api({"id": "pod-cpu"}, RUNPOD_API_URL="https://api.runpod.dev/v2/")
        ctl.create_pod(CPU_REQUEST, self_terminate_seconds=15_300)
        _, url, payload, _, _ = http.calls[0]
        assert payload is not None
        self.assertEqual(url, "https://api.runpod.dev/v2/pods")
        self.assertEqual(payload["env"]["RUNPOD_API_URL"], "https://api.runpod.dev/v2")

    def test_empty_api_url_variable_means_production(self) -> None:
        ctl, http = api({"items": []}, NOT_FOUND, RUNPOD_API_URL="")
        ctl.check_auth()
        self.assertTrue(http.calls[0][1].startswith("https://api.runpod.io/v2/"))

    def test_create_classifies_only_documented_placement_failures(self) -> None:
        for detail in (NO_CAPACITY, "Insufficient resources"):
            with self.subTest(detail=detail):
                ctl, _ = api(RestApiError("POST", 400, detail))
                with self.assertRaises(CapacityUnavailableError):
                    ctl.create_pod(GPU_BUILD_REQUEST, self_terminate_seconds=15_300)
        # A malformed request and a key that may not create Pods fail on the
        # first candidate instead of sitting through every placement round.
        failures = [(400, "invalid gpu id"), (403, NO_CAPACITY)]
        failures += [(status, "x") for status in (401, 402, 403, 422, 500)]
        for status, detail in failures:
            with self.subTest(status=status, detail=detail):
                ctl, _ = api(RestApiError("POST", status, detail))
                with self.assertRaises(CommandError) as raised:
                    ctl.create_pod(GPU_BUILD_REQUEST, self_terminate_seconds=15_300)
                self.assertNotIsInstance(raised.exception, CapacityUnavailableError)

    def test_create_requires_a_pod_id(self) -> None:
        ctl, _ = api({"name": "no-id"})
        with self.assertRaisesRegex(CommandError, "did not contain a Pod id"):
            ctl.create_pod(CPU_REQUEST, self_terminate_seconds=15_300)

    def test_pod_reads_are_single_pod_gets_without_a_body(self) -> None:
        ctl, http = api({"id": "pod-1"}, {"id": "pod-1"})
        self.assertEqual(ctl.pod_details("pod-1")["id"], "pod-1")
        self.assertEqual(ctl.assignment_details("pod-1")["id"], "pod-1")
        for method, url, payload, _, _ in http.calls:
            self.assertEqual((method, url), ("GET", "https://api.runpod.io/v2/pods/pod-1"))
            self.assertIsNone(payload)

    def test_terminate_deletes_and_treats_missing_pod_as_done(self) -> None:
        ctl, http = api(None, RestApiError("DELETE", 404, "pod not found"))
        ctl.terminate_pod("pod-1")
        ctl.terminate_pod("pod-1")
        self.assertEqual(
            [(c[0], c[1]) for c in http.calls],
            [("DELETE", "https://api.runpod.io/v2/pods/pod-1")] * 2,
        )

    def test_terminate_surfaces_other_failures(self) -> None:
        ctl, _ = api(RestApiError("DELETE", 500, "boom"))
        with self.assertRaisesRegex(CommandError, "HTTP 500"):
            ctl.terminate_pod("pod-1")

    def test_self_terminate_deadline_is_bounded(self) -> None:
        ctl, _ = api()
        for seconds in (599, 21_601):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                ctl.create_pod(CPU_REQUEST, self_terminate_seconds=seconds)


class PodRequestTests(unittest.TestCase):
    def test_cpu_request_requires_measured_container_disk_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 80 GB container disk"):
            PodRequest(
                image="registry.invalid/builder",
                name="undersized-cpu-build",
                public_key="ssh-ed25519 AAAA test",
                container_disk_gb=79,
            ).validate()

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



class EndpointTests(unittest.TestCase):
    def test_extracts_v2_direct_ssh(self) -> None:
        endpoint = _extract_ssh_endpoint(
            {"ssh": {"direct": {"host": "203.0.113.1", "port": 22001, "username": "root"}}}
        )
        self.assertEqual(endpoint, Endpoint("203.0.113.1", 22001))

    def test_ignores_proxy_only_or_incomplete_ssh(self) -> None:
        # The SSH proxy cannot carry the SCP upload and artifact download.
        for payload in (
            {"ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": None}},
            {"ssh": {"direct": {"host": "203.0.113.1"}}},
            {"ssh": {"direct": {"host": "203.0.113.1", "port": "not-a-port"}}},
            {"ssh": None},
            {},
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(_extract_ssh_endpoint(payload))

    def test_waits_for_running_direct_ssh_and_a_successful_probe(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.responses = [
                    {"status": "PROVISIONING", "ssh": {"direct": None}},
                    # v2 can report a mapping before RUNNING; do not probe it.
                    {"status": "STARTING", "ssh": {"direct": {"host": "early", "port": 1}}},
                    {"status": "RUNNING", "ssh": {"direct": {"host": "host", "port": 22}}},
                ]

            def pod_details(self, pod_id, timeout):
                return self.responses.pop(0)

        class FakeTransport:
            def __init__(self):
                self.probed: list[Endpoint] = []

            def probe(self, endpoint, timeout):
                self.probed.append(endpoint)
                return True

        transport = FakeTransport()
        endpoint = wait_for_ssh(
            FakeCtl(),  # type: ignore[arg-type]
            transport,  # type: ignore[arg-type]
            "pod-1",
            Deadline(10),
            poll_seconds=0.01,
            sleep=lambda _: None,
        )
        self.assertEqual(endpoint, Endpoint("host", 22))
        self.assertEqual(transport.probed, [Endpoint("host", 22)])

    def test_terminal_state_stops_waiting(self) -> None:
        for status in ("ERROR", "EXITED", "TERMINATED"):
            class FakeCtl:
                def pod_details(self, pod_id, timeout, status=status):
                    return {"status": status}

            with self.subTest(status=status), self.assertRaisesRegex(
                JobError, f"terminal state {status}"
            ):
                wait_for_ssh(
                    FakeCtl(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    "pod-1",
                    Deadline(10),
                    sleep=lambda _: None,
                )


class FakeAssignmentCtl:
    def __init__(self, details):
        self.details = details

    def assignment_details(self, pod_id, timeout):
        return self.details


CPU_BUILD_IMAGE = "registry.invalid/builder@sha256:" + "a" * 64
GPU_BUILD_IMAGE = "registry.invalid/builder@sha256:" + "e" * 64
RUNTIME_IMAGE = "registry.invalid/runtime@sha256:" + "b" * 64
CPU_POD = {
    "image": CPU_BUILD_IMAGE,
    "cpu": {"id": "cpu3g", "vcpuCount": 16, "memory": 62},
    "disk": 80,
    "mounts": {"persistent": None, "network": [], "global": []},
}
GPU_BUILD_POD = {
    "image": GPU_BUILD_IMAGE,
    "gpu": {"id": "NVIDIA A100 80GB PCIe", "count": 1, "vcpuCount": 4, "memory": 32},
    "disk": 80,
}


def without(mapping, key):
    return {k: v for k, v in mapping.items() if k != key}


class AssignmentTests(unittest.TestCase):
    cpu_request = PodRequest(
        image=CPU_BUILD_IMAGE,
        name="cpu-build",
        public_key="ssh-ed25519 AAAA test",
        compute_type="CPU",
    )
    gpu_build_request = PodRequest(
        image=GPU_BUILD_IMAGE,
        name="gpu-build",
        public_key="ssh-ed25519 AAAA test",
        compute_type="GPU",
        gpu_id="NVIDIA A100 80GB PCIe",
        gpu_workload="BUILD",
    )
    validation_request = PodRequest(
        image=RUNTIME_IMAGE,
        name="gpu-validation",
        public_key="ssh-ed25519 AAAA test",
        compute_type="GPU",
        gpu_id="NVIDIA RTX 4090",
    )

    def verify(self, details, request):
        return verify_pod_assignment(
            FakeAssignmentCtl(details),  # type: ignore[arg-type]
            "pod-1",
            request,
            Deadline(10),
        )

    def test_cpu_assignment_is_bound_to_image_and_minimum_resources(self) -> None:
        assignment = self.verify(CPU_POD, self.cpu_request)
        self.assertEqual(
            assignment,
            VerifiedBuildAssignment(vcpu_count=16, assigned_memory_bytes=62_000_000_000),
        )
        assert assignment is not None
        with self.assertRaises(FrozenInstanceError):
            assignment.vcpu_count = 32  # type: ignore[misc]

    def test_cpu_assignment_without_mounts_has_no_volume(self) -> None:
        # REST v2 attaches no persistent volume unless one was requested.
        self.assertIsNotNone(self.verify(without(CPU_POD, "mounts"), self.cpu_request))

    def test_cpu_assignment_rejects_service_drift(self) -> None:
        cpu = CPU_POD["cpu"]
        for details, message in (
            ({**CPU_POD, "image": "wrong/image:latest"}, "Pod image mismatch"),
            ({**CPU_POD, "cpu": {**cpu, "id": "cpu3c"}}, "CPU flavor mismatch"),
            ({**CPU_POD, "cpu": {**cpu, "vcpuCount": 2}}, "CPU count mismatch"),
            ({**CPU_POD, "cpu": {**cpu, "memory": 4}}, "memory mismatch"),
            ({**CPU_POD, "disk": 79}, "container disk mismatch"),
            (
                {**CPU_POD, "mounts": {"persistent": {"size": 20, "path": "/workspace"}}},
                "unexpected paid Pod volume",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                JobError, message
            ):
                self.verify(details, self.cpu_request)

    def test_gpu_validation_assignment_retains_light_contract(self) -> None:
        details = {"image": RUNTIME_IMAGE, "gpu": {"id": "NVIDIA RTX 4090"}}
        self.assertIsNone(self.verify(details, self.validation_request))

    def test_gpu_validation_assignment_requires_exact_gpu_id(self) -> None:
        for details in (
            {"image": RUNTIME_IMAGE, "gpu": {"id": "NVIDIA L40S"}},
            {"image": RUNTIME_IMAGE},
        ):
            with self.subTest(details=details), self.assertRaisesRegex(
                JobError, "GPU type mismatch"
            ):
                self.verify(details, self.validation_request)

    def test_gpu_build_assignment_returns_verified_resources(self) -> None:
        self.assertEqual(
            self.verify(GPU_BUILD_POD, self.gpu_build_request),
            VerifiedBuildAssignment(4, 32_000_000_000),
        )

    def test_gpu_build_assignment_fails_closed_on_resource_drift(self) -> None:
        gpu = GPU_BUILD_POD["gpu"]
        cases = (
            ({**GPU_BUILD_POD, "image": "wrong-image"}, "Pod image mismatch"),
            (
                {**GPU_BUILD_POD, "gpu": {**gpu, "id": "NVIDIA A100-SXM4-80GB"}},
                "GPU type mismatch",
            ),
            (without(GPU_BUILD_POD, "gpu"), "GPU type mismatch"),
            ({**GPU_BUILD_POD, "gpu": {**gpu, "vcpuCount": 3}}, "GPU build CPU count mismatch"),
            ({**GPU_BUILD_POD, "gpu": {**gpu, "vcpuCount": 4.5}}, "GPU build CPU count mismatch"),
            (
                {**GPU_BUILD_POD, "gpu": {**gpu, "vcpuCount": float("nan")}},
                "GPU build CPU count mismatch",
            ),
            (
                {**GPU_BUILD_POD, "gpu": {**gpu, "vcpuCount": float("inf")}},
                "GPU build CPU count mismatch",
            ),
            (
                {**GPU_BUILD_POD, "gpu": without(gpu, "vcpuCount")},
                "GPU build CPU count mismatch",
            ),
            ({**GPU_BUILD_POD, "gpu": {**gpu, "memory": 31}}, "GPU build memory mismatch"),
            (
                {**GPU_BUILD_POD, "gpu": {**gpu, "memory": float("nan")}},
                "GPU build memory mismatch",
            ),
            (
                {**GPU_BUILD_POD, "gpu": {**gpu, "memory": float("inf")}},
                "GPU build memory mismatch",
            ),
            (
                {**GPU_BUILD_POD, "gpu": without(gpu, "memory")},
                "GPU build memory mismatch",
            ),
            ({**GPU_BUILD_POD, "disk": 79}, "container disk mismatch"),
            ({**GPU_BUILD_POD, "disk": 80.5}, "container disk mismatch"),
            (without(GPU_BUILD_POD, "disk"), "container disk mismatch"),
            (
                {**GPU_BUILD_POD, "mounts": {"persistent": {"size": 20, "path": "/workspace"}}},
                "unexpected paid Pod volume",
            ),
            (
                {**GPU_BUILD_POD, "mounts": {"persistent": {"path": "/workspace"}}},
                "unexpected paid Pod volume",
            ),
        )
        for details, message in cases:
            with self.subTest(message=message, details=details), self.assertRaisesRegex(
                JobError, message
            ):
                self.verify(details, self.gpu_build_request)


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

    def test_long_command_streams_before_exit(self) -> None:
        class SignalingStream(io.StringIO):
            def __init__(self, marker: str, ready: threading.Event) -> None:
                super().__init__()
                self.marker = marker
                self.ready = ready

            def write(self, value: str) -> int:
                written = super().write(value)
                if self.marker in value:
                    self.ready.set()
                return written

        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp) / "key"
            key.write_text("private", encoding="utf-8")
            transport = SSHTransport(key)
            ready = threading.Event()
            stdout = SignalingStream("stream-ready", ready)
            stderr = io.StringIO()
            failures: list[BaseException] = []

            def run() -> None:
                try:
                    with mock.patch("sys.stdout", stdout), mock.patch(
                        "sys.stderr", stderr
                    ):
                        transport._run_streaming(
                            [
                                sys.executable,
                                "-u",
                                "-c",
                                "import time; print('stream-ready'); time.sleep(0.5)",
                            ],
                            timeout=5,
                        )
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(ready.wait(3), "child output was not forwarded live")
            self.assertTrue(worker.is_alive(), "command exited before its output was observed")
            worker.join(5)
            transport.close()

        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertIn("stream-ready", stdout.getvalue())

    def test_long_command_streams_stderr_and_preserves_checked_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp) / "key"
            key.write_text("private", encoding="utf-8")
            transport = SSHTransport(key)
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with mock.patch("sys.stdout", stdout), mock.patch(
                    "sys.stderr", stderr
                ), self.assertRaisesRegex(CommandError, "remote failure"):
                    transport._run_streaming(
                        [
                            sys.executable,
                            "-u",
                            "-c",
                            "import sys; print('remote failure', file=sys.stderr); "
                            "raise SystemExit(7)",
                        ],
                        timeout=5,
                    )
            finally:
                transport.close()

        self.assertIn("remote failure", stderr.getvalue())


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

    def test_cpu_build_exports_verified_resources_for_every_command(self) -> None:
        class FakeCtl:
            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                return "pod-cpu-build"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpu": {"id": "cpu3g", "vcpuCount": "16.0", "memory": "62.5"},
                    "disk": "80",
                }

            def terminate_pod(self, pod_id, timeout):
                return None

        class FakeTransport:
            def __init__(self):
                self.scripts: list[str] = []

            def upload_repo(self, *args):
                return None

            def run_script(self, endpoint, remote_dir, script, deadline):
                self.scripts.append(script)

            def download_artifacts(self, *args):
                return None

        with tempfile.TemporaryDirectory() as temp:
            transport = FakeTransport()
            run_job(
                replace(
                    self._spec(Path(temp)),
                    commands=("first-build-step", "second-build-step"),
                ),
                ctl=FakeCtl(),  # type: ignore[arg-type]
                transport=transport,  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        receipt_prefix = (
            "install -d -m 0755 /run/sageattention; "
            "resource_receipt=$(mktemp "
            "/run/sageattention/.verified-memory-bytes-v1.XXXXXX); "
            "printf '%s\\n' 62500000000 > \"$resource_receipt\"; "
            "chmod 0444 \"$resource_receipt\"; "
            "mv -f -- \"$resource_receipt\" "
            "/run/sageattention/verified-memory-bytes-v1; "
            "unset resource_receipt; "
        )
        export_prefix = (
            "export RUNPOD_CPU_COUNT=16; "
            "export RUNPOD_ASSIGNED_MEMORY_BYTES=62500000000; "
        )
        self.assertEqual(
            transport.scripts,
            [
                receipt_prefix + export_prefix + "first-build-step",
                receipt_prefix + export_prefix + "second-build-step",
            ],
        )
        self.assertIn("chmod 0444", transport.scripts[0])
        self.assertNotIn("RUNPOD_SELECTED_GPU_ID", " ".join(transport.scripts))

    def test_gpu_build_assignment_failure_terminates_before_upload(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.self_terminate_seconds = self_terminate_seconds
                return "pod-gpu-build"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": "NVIDIA A100 80GB PCIe", "vcpuCount": 3, "memory": 64},
                    "disk": 80,
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
            self.assertGreaterEqual(ctl.self_terminate_seconds, 600)
            self.assertEqual(ctl.terminated, ["pod-gpu-build"])

    def test_gpu_build_rejects_and_deletes_before_next_candidate(self) -> None:
        first = "NVIDIA A100 80GB PCIe"
        second = "NVIDIA H100 PCIe"
        events: list[tuple[str, str]] = []

        class FakeCtl:
            def __init__(self):
                self.deadlines: list[int] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.deadlines.append(self_terminate_seconds)
                events.append(("create", request.gpu_id))
                return "pod-first" if request.gpu_id == first else "pod-second"

            def assignment_details(self, pod_id, timeout):
                gpu_id = first if pod_id == "pod-first" else second
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {
                        "id": gpu_id,
                        "vcpuCount": 3 if pod_id == "pod-first" else 4,
                        "memory": 48 if pod_id == "pod-first" else 64,
                    },
                    "disk": 80,
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
        self.assertGreaterEqual(ctl.deadlines[0], 600)
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
        self.assertIn("export RUNPOD_CPU_COUNT=4;", transport.scripts[0])
        self.assertIn(
            "export RUNPOD_ASSIGNED_MEMORY_BYTES=64000000000;",
            transport.scripts[0],
        )
        self.assertNotIn("48000000000", transport.scripts[0])

    def test_gpu_validation_exact_assignment_mismatch_uses_next_candidate(self) -> None:
        first = "NVIDIA GeForce RTX 4090"
        second = "NVIDIA L40S"

        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.candidates.append(request.gpu_id)
                return f"pod-{len(self.candidates)}"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/runtime@sha256:" + "f" * 64,
                    "gpu": {"id": "unexpected" if pod_id == "pod-1" else second},
                }

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        class FakeTransport:
            def __init__(self):
                self.scripts: list[str] = []

            def upload_repo(self, *args):
                return None

            def run_script(self, endpoint, remote_dir, script, deadline):
                self.scripts.append(script)

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
            transport = FakeTransport()
            result = run_job(
                spec,
                ctl=ctl,  # type: ignore[arg-type]
                transport=transport,  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        self.assertEqual(ctl.candidates, [first, second])
        self.assertEqual(ctl.terminated, ["pod-1", "pod-2"])
        self.assertEqual(result.selected_gpu_id, second)
        self.assertIn(
            "export RUNPOD_SELECTED_GPU_ID='NVIDIA L40S';",
            transport.scripts[0],
        )
        self.assertNotIn("RUNPOD_CPU_COUNT", transport.scripts[0])
        self.assertNotIn("RUNPOD_ASSIGNED_MEMORY_BYTES", transport.scripts[0])
        self.assertNotIn("verified-memory-bytes-v1", transport.scripts[0])

    def test_gpu_capacity_fallback_uses_two_bounded_ordered_rounds(self) -> None:
        first = "NVIDIA A100 80GB PCIe"
        second = "NVIDIA H100 PCIe"

        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []
                self.deadlines: list[int] = []
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.candidates.append(request.gpu_id)
                self.deadlines.append(self_terminate_seconds)
                if len(self.candidates) <= 2:
                    raise CapacityUnavailableError(
                        "There are no longer any instances available with the "
                        "requested specifications"
                    )
                return "pod-selected"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": first, "vcpuCount": 4, "memory": 32},
                    "disk": 80,
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
        self.assertGreaterEqual(ctl.deadlines[0], 600)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 300.0)
        self.assertEqual(result.selected_gpu_id, first)
        self.assertEqual(ctl.terminated, ["pod-selected"])

    def test_gpu_attempt_lines_are_flushed_before_each_create(self) -> None:
        first = "NVIDIA A100 80GB PCIe"
        second = "NVIDIA H100 PCIe"
        events: list[tuple[str, object, object]] = []

        class FakeCtl:
            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                events.append(("create", request.gpu_id, None))
                if request.gpu_id == first:
                    raise CapacityUnavailableError("no matching capacity")
                return "pod-selected"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": second, "vcpuCount": 4, "memory": 32},
                    "disk": 80,
                }

            def terminate_pod(self, pod_id, timeout):
                return None

        class FakeTransport:
            def upload_repo(self, *args):
                return None

            def run_script(self, *args):
                return None

            def download_artifacts(self, *args):
                return None

        def record_print(*args, **kwargs):
            message = " ".join(str(value) for value in args)
            events.append(("print", message, kwargs.get("flush")))

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "builtins.print", side_effect=record_print
        ):
            run_job(
                self._gpu_spec(Path(temp), (first, second)),
                ctl=FakeCtl(),  # type: ignore[arg-type]
                transport=FakeTransport(),  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )

        for candidate in (first, second):
            attempt_index = next(
                index
                for index, event in enumerate(events)
                if event[0] == "print"
                and "Trying GPU candidate" in str(event[1])
                and repr(candidate) in str(event[1])
            )
            create_index = events.index(("create", candidate, None))
            self.assertLess(attempt_index, create_index)
            self.assertIs(events[attempt_index][2], True)
        attempt_messages = [
            str(event[1])
            for event in events
            if event[0] == "print" and "Trying GPU candidate" in str(event[1])
        ]
        self.assertIn("Trying GPU candidate 1/2 (round 1/5)", attempt_messages[0])
        self.assertIn("Trying GPU candidate 2/2 (round 1/5)", attempt_messages[1])

    def test_gpu_candidate_fallback_does_not_retry_configuration_error(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.candidates: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.candidates.append(request.gpu_id)
                raise CommandError("Runpod REST POST failed with HTTP 401: unauthorized")

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

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.candidates.append(request.gpu_id)
                return "pod-first"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "e" * 64,
                    "gpu": {"id": first, "vcpuCount": 4, "memory": 32},
                    "disk": 80,
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

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.self_terminate_seconds = self_terminate_seconds
                return "pod-created"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpu": {"id": "cpu3g", "vcpuCount": 16, "memory": 62},
                    "disk": 80,
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

    def test_missing_debug_artifact_does_not_obscure_primary_failure(self) -> None:
        class FakeCtl:
            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                return "pod-created"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpu": {"id": "cpu3g", "vcpuCount": 16, "memory": 62},
                    "disk": 80,
                }

            def terminate_pod(self, pod_id, timeout):
                return None

        class MissingArtifactTransport:
            def upload_repo(self, *args):
                return None

            def run_script(self, *args):
                raise CommandError("Required build command is missing: nvcc")

            def download_artifacts(self, *args):
                raise CommandError(
                    "ssh failed: missing requested artifact: dist/build"
                )

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "sys.stderr", stderr
        ):
            with self.assertRaises(JobError) as raised:
                run_job(
                    self._spec(Path(temp)),
                    ctl=FakeCtl(),  # type: ignore[arg-type]
                    transport=MissingArtifactTransport(),  # type: ignore[arg-type]
                    wait_fn=lambda *args, **kwargs: Endpoint("host", 22),
                )

        self.assertIn("Required build command is missing: nvcc", str(raised.exception))
        self.assertNotIn("artifact download", str(raised.exception))
        self.assertIn("no requested debug artifact was produced", stderr.getvalue())

    def test_success_returns_result_and_terminates(self) -> None:
        class FakeCtl:
            def __init__(self):
                self.terminated = False

            def check_auth(self, timeout):
                pass

            def create_pod(self, request, self_terminate_seconds, timeout):
                return "pod-ok"

            def assignment_details(self, pod_id, timeout):
                return {
                    "image": "registry.invalid/builder@sha256:" + "c" * 64,
                    "cpu": {"id": "cpu3g", "vcpuCount": 16, "memory": 62},
                    "disk": 80,
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
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                result = run_job(
                    self._spec(Path(temp)),
                    ctl=ctl,  # type: ignore[arg-type]
                    transport=FakeTransport(),  # type: ignore[arg-type]
                    wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
                )
            self.assertEqual(result.pod_id, "pod-ok")
            self.assertEqual(result.endpoint.port, 2200)
            self.assertTrue(ctl.terminated)
            messages = output.getvalue()
            self.assertIn(
                "Pod pod-ok: uploading repository to /work/sageattention-factory",
                messages,
            )
            self.assertIn("Pod pod-ok: repository upload complete", messages)
            self.assertIn("Pod pod-ok: starting remote command 1/1", messages)
            self.assertIn("Pod pod-ok: remote command 1/1 complete", messages)
            self.assertIn("Pod pod-ok: downloading requested artifacts", messages)
            self.assertIn("Pod pod-ok: artifact download complete", messages)
            self.assertIn("Pod pod-ok: terminated", messages)



class PlacementRetryTests(unittest.TestCase):
    """Capacity retries: bounded rounds, a backoff between them, fresh budgets."""

    CPU_POD = {
        "image": "registry.invalid/builder@sha256:" + "c" * 64,
        "cpu": {"id": "cpu3g", "vcpuCount": 16, "memory": 64},
        "disk": 80,
    }

    def _spec(self, root: Path, **overrides) -> JobSpec:
        return replace(RunJobCleanupTests._spec(self, root), **overrides)  # type: ignore[arg-type]

    @staticmethod
    def _transport():
        class FakeTransport:
            def upload_repo(self, *args):
                return None

            def run_script(self, *args):
                return None

            def download_artifacts(self, *args):
                return None

        return FakeTransport()

    def test_cpu_placement_retries_capacity_until_a_round_succeeds(self) -> None:
        pod = self.CPU_POD

        class FakeCtl:
            def __init__(self):
                self.creates = 0
                self.terminated: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                self.creates += 1
                if self.creates < 3:
                    raise CapacityUnavailableError("no instances available")
                return "pod-cpu"

            def assignment_details(self, pod_id, timeout):
                return pod

            def terminate_pod(self, pod_id, timeout):
                self.terminated.append(pod_id)

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "tools.runpod_job.time.sleep"
        ) as sleep:
            ctl = FakeCtl()
            result = run_job(
                self._spec(Path(temp)),
                ctl=ctl,  # type: ignore[arg-type]
                transport=self._transport(),  # type: ignore[arg-type]
                wait_fn=lambda *args, **kwargs: Endpoint("host", 2200),
            )
        self.assertEqual(result.pod_id, "pod-cpu")
        self.assertEqual(ctl.creates, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [300.0, 300.0])
        self.assertEqual(ctl.terminated, ["pod-cpu"])

    def test_cpu_placement_gives_up_after_the_configured_rounds(self) -> None:
        class FakeCtl:
            creates = 0

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                FakeCtl.creates += 1
                raise CapacityUnavailableError(f"full #{FakeCtl.creates}")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "tools.runpod_job.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(
                JobError, r"CPU placement exhausted 3 rounds.*full #1.*full #3"
            ):
                run_job(
                    self._spec(
                        Path(temp), placement_rounds=3, placement_backoff_seconds=60
                    ),
                    ctl=FakeCtl(),  # type: ignore[arg-type]
                    transport=object(),  # type: ignore[arg-type]
                )
        self.assertEqual(FakeCtl.creates, 3)
        # A wait between rounds, none after the last one.
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [60, 60])

    def test_gpu_placement_gives_up_after_the_configured_rounds(self) -> None:
        class FakeCtl:
            candidates: list[str] = []

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                FakeCtl.candidates.append(request.gpu_id)
                raise CapacityUnavailableError("no instances available")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "tools.runpod_job.time.sleep"
        ) as sleep:
            spec = replace(
                RunJobCleanupTests._gpu_spec(self, Path(temp)),  # type: ignore[arg-type]
                placement_rounds=4,
            )
            with self.assertRaisesRegex(JobError, "exhausted 4 ordered rounds"):
                run_job(
                    spec,
                    ctl=FakeCtl(),  # type: ignore[arg-type]
                    transport=object(),  # type: ignore[arg-type]
                )
        self.assertEqual(len(FakeCtl.candidates), 4 * 2)
        self.assertEqual(sleep.call_count, 3)

    def test_non_capacity_failure_is_not_retried(self) -> None:
        class FakeCtl:
            creates = 0

            def check_auth(self, timeout):
                return None

            def create_pod(self, request, self_terminate_seconds, timeout):
                FakeCtl.creates += 1
                raise CommandError("Runpod REST POST failed with HTTP 422: bad body")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "tools.runpod_job.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(JobError, "HTTP 422"):
                run_job(
                    self._spec(Path(temp)),
                    ctl=FakeCtl(),  # type: ignore[arg-type]
                    transport=object(),  # type: ignore[arg-type]
                )
        self.assertEqual(FakeCtl.creates, 1)
        sleep.assert_not_called()

    def test_each_round_gets_a_fresh_hard_budget_after_its_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            spec = self._spec(Path(temp), hard_timeout_seconds=900)
            with mock.patch("tools.runpod_job.time.sleep") as sleep:
                first = _start_placement_round(spec, 1, "GPU")
                second = _start_placement_round(spec, 2, "GPU")
        sleep.assert_called_once_with(300.0)
        self.assertIsNot(first, second)
        # The wait is not charged to the budget the next round works with.
        self.assertGreater(second.seconds_left(), 899)

    def test_retry_policy_is_bounded_and_exposed_by_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for overrides, message in (
                ({"placement_rounds": 0}, "between 1 and 10"),
                ({"placement_rounds": 11}, "between 1 and 10"),
                ({"placement_backoff_seconds": -1}, "between 0 and 900"),
                ({"placement_backoff_seconds": 901}, "between 0 and 900"),
            ):
                with self.subTest(overrides=overrides), self.assertRaisesRegex(
                    ValueError, message
                ):
                    self._spec(Path(temp), **overrides).validate()
        parser = _parser()
        self.assertEqual(parser.get_default("placement_rounds"), 5)
        self.assertEqual(parser.get_default("placement_backoff_seconds"), 300.0)


if __name__ == "__main__":
    unittest.main()
