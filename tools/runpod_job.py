#!/usr/bin/env python3
"""Run one bounded build or test job on an ephemeral Runpod Pod.

The module intentionally depends only on Python's standard library, the
Runpod REST API v2, and the ``ssh`` and ``scp`` executables.  GitHub Actions is an
orchestrator only: the checked-out repository is uploaded to the Pod, commands
run there, requested artifacts are copied back, and the Pod is deleted in a
``finally`` path.

Creating a paid Pod requires the explicit ``--allow-paid-pod`` flag.  CI and
pull-request workflows must never pass that flag; manually approved build and
release workflows may.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence, TextIO


class JobError(RuntimeError):
    """Base class for an orchestration failure."""


class JobTimeout(JobError):
    """The hard wall-clock budget was exhausted."""


class CommandError(JobError):
    """A local, SSH, or Runpod API operation failed."""


class CapacityUnavailableError(CommandError):
    """Runpod had no instance matching one exact GPU placement request."""


class GpuPlacementError(JobError):
    """A created GPU Pod failed retryable pre-upload placement checks."""


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int


@dataclass
class Deadline:
    """Monotonic hard deadline shared by every non-cleanup operation."""

    timeout_seconds: float
    clock: Callable[[], float] = time.monotonic
    _ends_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._ends_at = self.clock() + self.timeout_seconds

    def seconds_left(self) -> float:
        return max(0.0, self._ends_at - self.clock())

    def timeout(self, cap: float | None = None) -> float:
        remaining = self.seconds_left()
        if remaining <= 0:
            raise JobTimeout("Runpod job exceeded its hard timeout")
        return max(0.1, min(remaining, cap)) if cap is not None else remaining


@dataclass(frozen=True)
class PodRequest:
    image: str
    name: str
    public_key: str
    compute_type: str = "CPU"
    gpu_id: str | None = None
    cloud_type: str = "COMMUNITY"
    container_disk_gb: int = 80
    min_cuda_version: str | None = None
    gpu_workload: str = "VALIDATION"
    gpu_min_vcpu_count: int = 4
    gpu_min_memory_gb: int = 32
    registry_auth_id: str | None = None
    data_center_ids: str | None = None
    cpu_flavor_ids: tuple[str, ...] = ("cpu3g",)
    cpu_vcpu_count: int = 16
    cpu_min_memory_gb: int = 32

    @property
    def is_gpu_build(self) -> bool:
        return (
            self.compute_type.upper() == "GPU"
            and self.gpu_workload.upper() == "BUILD"
        )

    def validate(self) -> None:
        if not self.image.strip():
            raise ValueError("image is required")
        if not self.name.strip():
            raise ValueError("Pod name is required")
        if not self.public_key.strip():
            raise ValueError("PUBLIC_KEY must not be empty")
        mode = self.compute_type.upper()
        if mode not in {"CPU", "GPU"}:
            raise ValueError("compute_type must be CPU or GPU")
        if mode == "GPU" and not (self.gpu_id or "").strip():
            raise ValueError("an exact gpu_id is required for GPU Pods")
        if mode == "CPU":
            if self.gpu_workload.upper() != "VALIDATION":
                raise ValueError("gpu_workload applies only to GPU Pods")
            if not self.cpu_flavor_ids or any(
                not value.strip() for value in self.cpu_flavor_ids
            ):
                raise ValueError("at least one exact CPU flavor id is required")
            if self.cpu_vcpu_count < 4:
                raise ValueError("CPU build Pods require at least 4 vCPUs")
            if self.cpu_min_memory_gb < 32:
                raise ValueError("CPU build Pods require at least 32 GB RAM")
            if self.container_disk_gb < 80:
                raise ValueError("CPU build Pods require at least 80 GB container disk")
        else:
            workload = self.gpu_workload.upper()
            if workload not in {"VALIDATION", "BUILD"}:
                raise ValueError("gpu_workload must be VALIDATION or BUILD")
            if workload == "BUILD":
                if self.gpu_min_vcpu_count < 4:
                    raise ValueError("GPU build Pods require at least 4 vCPUs")
                if self.gpu_min_memory_gb < 32:
                    raise ValueError("GPU build Pods require at least 32 GB system RAM")
                if self.container_disk_gb < 80:
                    raise ValueError(
                        "GPU build Pods require at least 80 GB container disk"
                    )
            elif self.container_disk_gb < 10:
                raise ValueError("GPU validation Pod container_disk_gb must be at least 10")


@dataclass(frozen=True)
class JobSpec:
    pod: PodRequest
    repo_root: Path
    ssh_private_key: Path
    remote_dir: str
    commands: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    artifact_output: Path
    gpu_id_candidates: tuple[str, ...] = ()
    hard_timeout_seconds: int = 14_400
    terminate_grace_seconds: int = 900
    poll_seconds: float = 5.0
    allow_paid_pod: bool = False
    placement_rounds: int = 5
    placement_backoff_seconds: float = 300.0

    def validate(self) -> None:
        self.pod.validate()
        if self.pod.compute_type.upper() == "GPU":
            candidates = self.gpu_id_candidates or (self.pod.gpu_id or "",)
            if candidates[0] != self.pod.gpu_id:
                raise ValueError(
                    "the first gpu_id candidate must match the Pod request gpu_id"
                )
            if len(candidates) > 8:
                raise ValueError("at most 8 ordered gpu_id candidates are allowed")
            if any(
                not candidate
                or candidate != candidate.strip()
                or "\n" in candidate
                or "\r" in candidate
                or "," in candidate
                for candidate in candidates
            ):
                raise ValueError("gpu_id candidates must be exact non-empty single lines")
            if len(set(candidates)) != len(candidates):
                raise ValueError("gpu_id candidates must not contain duplicates")
        if not self.allow_paid_pod:
            raise ValueError(
                "paid Pod creation is disabled; pass --allow-paid-pod only "
                "from an approved workflow_dispatch/release job"
            )
        if not self.repo_root.is_dir():
            raise ValueError(f"repo root does not exist: {self.repo_root}")
        if not self.ssh_private_key.is_file():
            raise ValueError(
                f"SSH private key does not exist: {self.ssh_private_key}"
            )
        _validate_remote_dir(self.remote_dir)
        if self.pod.compute_type.upper() == "CPU" or self.pod.is_gpu_build:
            remote_parts = PurePosixPath(self.remote_dir).parts
            if len(remote_parts) < 3 or remote_parts[:2] != ("/", "work"):
                raise ValueError("build remote_dir must be a directory below /work")
        if not self.commands:
            raise ValueError("at least one remote command is required")
        for path in self.artifact_paths:
            _validate_relative_remote_path(path)
        if self.hard_timeout_seconds <= 0:
            raise ValueError("hard timeout must be positive")
        if self.terminate_grace_seconds < 60:
            raise ValueError("terminate grace must be at least 60 seconds")
        if not 1 <= self.placement_rounds <= 10:
            raise ValueError("placement rounds must be between 1 and 10")
        if not 0 <= self.placement_backoff_seconds <= 900:
            raise ValueError("placement backoff must be between 0 and 900 seconds")


@dataclass(frozen=True)
class JobResult:
    pod_id: str
    endpoint: Endpoint
    artifact_output: Path
    selected_gpu_id: str | None = None


@dataclass(frozen=True)
class VerifiedBuildAssignment:
    """Trusted build resources read back from the selected Runpod Pod."""

    vcpu_count: int
    assigned_memory_bytes: int


DEFAULT_API_URL = "https://api.runpod.io/v2"
# Cloudflare in front of the Runpod API rejects urllib's default
# Python-urllib/x.y agent with 403 "error code: 1010".
USER_AGENT = "runpod-sageattention-factory/1"
# Never a real Pod id (ids are lowercase alphanumerics), so the write probe in
# check_auth cannot touch anything.
AUTH_PROBE_POD_ID = "sageattention-write-access-probe"
Executor = Callable[..., subprocess.CompletedProcess[str]]
HttpExecutor = Callable[
    [str, str, Mapping[str, object] | None, Mapping[str, str], float], object
]


def _completed_text(proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip() for part in (proc.stdout or "", proc.stderr or "") if part.strip()
    )


class RestApiError(CommandError):
    """The Runpod REST API answered with a non-2xx status."""

    def __init__(self, method: str, status: int, detail: str) -> None:
        super().__init__(
            f"Runpod REST {method} failed with HTTP {status}: {detail}"
        )
        self.status = status
        self.detail = detail


# Messages Runpod uses for "no host can take this right now".  REST v2 sends
# them with HTTP 400, the same status as a malformed request, so the text is
# the only thing that tells capacity apart from a request that can never work.
_CAPACITY_MESSAGES = (
    "there are no longer any instances available with the requested "
    "specifications",
    "there are no instances available with the requested specifications",
    "this machine does not have the resources to deploy your pod. "
    "please try a different machine",
    "insufficient resources",
    "no gpus available",
    "not enough free gpus",
)


def _is_capacity_unavailable_error(error: CommandError) -> bool:
    """Classify a create failure as worth trying the next candidate.

    Only a 400 whose message names missing capacity qualifies.  Any other 400
    is a request the API will reject every time, and a 403 means the key or
    account may not do this at all; retrying either only burns placement
    rounds before failing with a misleading capacity message.
    """

    if isinstance(error, RestApiError):
        if error.status != 400:
            return False
        normalized = " ".join(error.detail.casefold().split())
    else:
        normalized = " ".join(str(error).casefold().split())
    return any(message in normalized for message in _CAPACITY_MESSAGES)


def _is_missing_resource(detail: str) -> bool:
    """A 404 body that names a missing resource, not an unknown route."""

    normalized = " ".join(detail.casefold().split())
    return "resource not found" in normalized or "pod not found" in normalized


def _http_json(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> object:
    """Issue one bounded JSON request using only the standard library."""

    body = (
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib_request.Request(
        url,
        data=body,
        headers=dict(headers),
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RestApiError(method, exc.code, detail) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise CommandError(f"Runpod REST {method} failed: {exc}") from exc
    if not raw.strip():
        return None  # 204 No Content, e.g. a successful DELETE
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"Runpod REST returned invalid JSON: {raw[:200]!r}"
        ) from exc


class RunpodApi:
    """Minimal Runpod REST v2 client for the Pod lifecycle this job needs."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        http_executor: HttpExecutor = _http_json,
        api_url: str | None = None,
    ) -> None:
        self._env = dict(env) if env is not None else None
        self._http_executor = http_executor
        environment = self._env if self._env is not None else os.environ
        api_url = (
            api_url
            or environment.get("RUNPOD_API_URL", "").strip()
            or DEFAULT_API_URL
        )
        self._api_url = api_url.rstrip("/")

    def _api_key(self) -> str:
        environment = self._env if self._env is not None else os.environ
        api_key = environment.get("RUNPOD_API_KEY", "").strip()
        if not api_key:
            raise CommandError("RUNPOD_API_KEY is required for the Runpod REST API")
        return api_key

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        timeout: float,
    ) -> object:
        return self._http_executor(
            method,
            f"{self._api_url}{path}",
            payload,
            {
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout,
        )

    def check_auth(self, *, timeout: float = 30) -> None:
        """Fail before any create when the key cannot manage Pods.

        Reading ``/pods`` only proves read access; a read-only key passes it
        and then gets 403 on every create.  A ``stop`` action
        (``POST /pods/{id}/action``, scope ``pods:write``) on a Pod id that
        cannot exist is a write the API must authorize first, and it changes
        nothing: a read-only key gets 401 or 403, an authorized key gets the
        spec's 404 "resource not found" (or a 400 for the id).  Any other 404,
        such as an unknown route, proves nothing and fails the check rather
        than letting every key through.
        """

        self._request("GET", "/pods?limit=1", timeout=timeout)
        try:
            self._request(
                "POST",
                f"/pods/{AUTH_PROBE_POD_ID}/action",
                {"action": "stop"},
                timeout=timeout,
            )
        except RestApiError as exc:
            if exc.status in {401, 403}:
                raise CommandError(
                    "RUNPOD_API_KEY can read Pods but not create or delete them "
                    f"(HTTP {exc.status} on a write probe); use a key with "
                    "read/write Pod access"
                ) from exc
            if exc.status == 400 or (
                exc.status == 404 and _is_missing_resource(exc.detail)
            ):
                return
            raise CommandError(
                f"write probe got an unexpected HTTP {exc.status} "
                f"({exc.detail!r}); cannot confirm the key may manage Pods"
            ) from exc
        raise CommandError(
            f"write probe unexpectedly stopped a Pod named {AUTH_PROBE_POD_ID!r}"
        )

    def list_pods(self, *, timeout: float = 60) -> list[dict[str, object]]:
        """Every Pod on the account, as REST v2 returns them."""

        payload = self._request("GET", "/pods", timeout=timeout)
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("pods"))
        if not isinstance(payload, list):
            raise CommandError("Runpod REST Pod list returned no list of Pods")
        return [pod for pod in payload if isinstance(pod, dict)]

    def create_pod(
        self,
        request: PodRequest,
        *,
        self_terminate_seconds: int,
        timeout: float = 120,
    ) -> str:
        request.validate()
        if self_terminate_seconds < 600 or self_terminate_seconds > 21_600:
            raise ValueError(
                "self-termination deadline must be between 600 and 21600 seconds"
            )
        env = {"PUBLIC_KEY": request.public_key.strip()}
        is_build = request.compute_type.upper() == "CPU" or request.is_gpu_build
        if is_build:
            # REST v2 has no platform-side terminate-after, so the builder
            # image's in-Pod watchdog is the backstop if the runner disappears.
            env["RUNPOD_SELF_TERMINATE_SECONDS"] = str(self_terminate_seconds)
            if self._api_url != DEFAULT_API_URL:
                # The watchdog calls the API too; keep it on the Pod's environment.
                env["RUNPOD_API_URL"] = self._api_url
        payload: dict[str, object] = {
            "name": request.name,
            "image": request.image,
            "cloud": request.cloud_type.upper(),
            # Compilation stays on the sized ephemeral container disk; REST v2
            # attaches no persistent volume unless `mounts` asks for one.
            "disk": request.container_disk_gb,
            "ports": ["22/tcp"],
            "env": env,
        }
        if request.data_center_ids:
            payload["dataCenterIds"] = [
                value.strip()
                for value in request.data_center_ids.split(",")
                if value.strip()
            ]
        if request.registry_auth_id:
            payload["registry"] = request.registry_auth_id

        if request.compute_type.upper() == "GPU":
            gpu: dict[str, object] = {"id": request.gpu_id or "", "count": 1}
            if request.min_cuda_version:
                gpu["minCudaVersion"] = request.min_cuda_version
            if request.is_gpu_build:
                # Placement filters, so an undersized host is never created.
                gpu["minVcpuCountPerGpu"] = request.gpu_min_vcpu_count
                gpu["minRamPerGpu"] = request.gpu_min_memory_gb
            return self._create(dict(payload, gpu=gpu), timeout=timeout)

        # REST v2 places one CPU flavor per create; walk the ordered list.
        last_error: CapacityUnavailableError | None = None
        for flavor in request.cpu_flavor_ids:
            cpu = {"id": flavor, "vcpuCount": request.cpu_vcpu_count}
            try:
                return self._create(dict(payload, cpu=cpu), timeout=timeout)
            except CapacityUnavailableError as exc:
                print(
                    f"CPU flavor {flavor!r} has no matching capacity: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                last_error = exc
        assert last_error is not None  # validate() requires at least one flavor
        raise last_error

    def _create(self, payload: Mapping[str, object], *, timeout: float) -> str:
        try:
            response = self._request("POST", "/pods", payload, timeout=timeout)
        except CommandError as exc:
            if _is_capacity_unavailable_error(exc):
                raise CapacityUnavailableError(str(exc)) from exc
            raise
        if not isinstance(response, dict):
            raise CommandError("Runpod REST Pod create returned a non-object response")
        pod_id = response.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise CommandError("Runpod REST Pod create response did not contain a Pod id")
        return pod_id

    def pod_details(self, pod_id: str, *, timeout: float = 30) -> dict[str, object]:
        payload = self._request("GET", f"/pods/{pod_id}", timeout=timeout)
        if not isinstance(payload, dict):
            raise CommandError("Runpod REST Pod get returned a non-object response")
        return payload

    # The single-Pod read carries both readiness and the assigned resources.
    assignment_details = pod_details

    def terminate_pod(self, pod_id: str, *, timeout: float = 30) -> None:
        try:
            self._request("DELETE", f"/pods/{pod_id}", timeout=timeout)
        except RestApiError as exc:
            if exc.status != 404:  # already gone is the goal of cleanup
                raise


def _validate_remote_dir(path: str) -> None:
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or parsed == PurePosixPath("/") or ".." in parsed.parts:
        raise ValueError("remote_dir must be a safe absolute directory below /")


def _validate_relative_remote_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts or path in {".", "./"}:
        raise ValueError(f"unsafe relative artifact path: {path!r}")


def _extract_ssh_endpoint(payload: Mapping[str, object]) -> Endpoint | None:
    """Return the Pod's direct SSH mapping from a REST v2 Pod read.

    Only ``ssh.direct`` is usable: the SSH proxy carries an interactive shell
    but not the SCP this job uses for upload and artifact download.
    """

    ssh = payload.get("ssh")
    direct = ssh.get("direct") if isinstance(ssh, Mapping) else None
    if not isinstance(direct, Mapping):
        return None
    host = direct.get("host")
    port = direct.get("port")
    if not host or not port:
        return None
    try:
        return Endpoint(str(host), int(port))
    except (TypeError, ValueError):
        return None


def _pod_status(payload: Mapping[str, object]) -> str:
    return str(payload.get("status") or "UNKNOWN").upper()


class SSHTransport:
    """Upload source, execute commands, and retrieve artifacts over SSH."""

    def __init__(
        self,
        private_key: Path,
        *,
        executor: Executor = subprocess.run,
    ) -> None:
        self.private_key = private_key
        self._executor = executor
        fd, known_hosts_name = tempfile.mkstemp(prefix="sageattention-known-hosts-")
        os.close(fd)
        self.known_hosts = Path(known_hosts_name)
        self.known_hosts.chmod(0o600)

    def close(self) -> None:
        self.known_hosts.unlink(missing_ok=True)

    def _ssh_prefix(self, endpoint: Endpoint) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.private_key),
            "-p",
            str(endpoint.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            f"root@{endpoint.host}",
        ]

    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            proc = self._executor(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CommandError(f"required executable is missing: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandError(f"{argv[0]} timed out after {timeout:.0f}s") from exc
        if check and proc.returncode != 0:
            detail = _completed_text(proc) or f"exit code {proc.returncode}"
            raise CommandError(f"{argv[0]} failed: {detail}")
        return proc

    def probe(self, endpoint: Endpoint, *, timeout: float) -> bool:
        command = [*self._ssh_prefix(endpoint), "printf '__runpod_ready__'"]
        proc = self._run(command, timeout=timeout, check=False)
        return proc.returncode == 0 and "__runpod_ready__" in (proc.stdout or "")

    def _run_remote(
        self,
        endpoint: Endpoint,
        script: str,
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = "bash -lc " + shlex.quote(script)
        return self._run(
            [*self._ssh_prefix(endpoint), command],
            timeout=timeout,
            check=check,
        )

    @staticmethod
    def _forward_stream(
        source: TextIO,
        destination: TextIO,
        captured: list[str],
    ) -> None:
        """Forward one subprocess text stream while retaining checked-error detail."""

        try:
            for chunk in iter(source.readline, ""):
                captured.append(chunk)
                destination.write(chunk)
                destination.flush()
        finally:
            source.close()

    def _run_streaming(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a long command with live output and the same checked semantics."""

        try:
            process = subprocess.Popen(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise CommandError(f"required executable is missing: {argv[0]}") from exc

        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise CommandError(f"{argv[0]} did not provide captured output streams")

        stdout: list[str] = []
        stderr: list[str] = []
        readers = (
            threading.Thread(
                target=self._forward_stream,
                args=(process.stdout, sys.stdout, stdout),
                daemon=True,
            ),
            threading.Thread(
                target=self._forward_stream,
                args=(process.stderr, sys.stderr, stderr),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            for reader in readers:
                reader.join()

        completed = subprocess.CompletedProcess(
            list(argv),
            returncode,
            "".join(stdout),
            "".join(stderr),
        )
        if timed_out:
            raise CommandError(f"{argv[0]} timed out after {timeout:.0f}s")
        if check and completed.returncode != 0:
            detail = _completed_text(completed) or f"exit code {completed.returncode}"
            raise CommandError(f"{argv[0]} failed: {detail}")
        return completed

    def upload_repo(
        self,
        endpoint: Endpoint,
        repo_root: Path,
        remote_dir: str,
        deadline: Deadline,
    ) -> None:
        _validate_remote_dir(remote_dir)
        archive_path = _make_repo_archive(repo_root)
        remote_archive = (
            f"{remote_dir.rstrip('/')}/.sageattention-source-"
            f"{uuid.uuid4().hex}.tar.gz"
        )
        try:
            # Place both the transfer archive and extracted checkout on the
            # caller-selected filesystem. Build jobs select /work, which is
            # the explicitly sized ephemeral container disk rather than a
            # small image-default /workspace volume.
            self._run_remote(
                endpoint,
                f"install -d -m 0700 {shlex.quote(remote_dir)}",
                timeout=deadline.timeout(60),
            )
            self._run(
                [
                    "scp",
                    "-i",
                    str(self.private_key),
                    "-P",
                    str(endpoint.port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    f"UserKnownHostsFile={self.known_hosts}",
                    str(archive_path),
                    f"root@{endpoint.host}:{remote_archive}",
                ],
                timeout=deadline.timeout(900),
            )
            script = (
                "set -euo pipefail; "
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_dir)}; "
                f"rm -f {shlex.quote(remote_archive)}"
            )
            self._run_remote(
                endpoint,
                script,
                timeout=deadline.timeout(300),
            )
        finally:
            archive_path.unlink(missing_ok=True)

    def run_script(
        self,
        endpoint: Endpoint,
        remote_dir: str,
        script: str,
        deadline: Deadline,
    ) -> None:
        wrapped = (
            "set -euo pipefail; "
            f"cd {shlex.quote(remote_dir)}; "
            f"{script}"
        )
        command = "bash -lc " + shlex.quote(wrapped)
        self._run_streaming(
            [*self._ssh_prefix(endpoint), command],
            timeout=deadline.timeout(),
        )

    def download_artifacts(
        self,
        endpoint: Endpoint,
        remote_dir: str,
        artifact_paths: Iterable[str],
        output_dir: Path,
        deadline: Deadline,
    ) -> None:
        paths = tuple(artifact_paths)
        if not paths:
            return
        for path in paths:
            _validate_relative_remote_path(path)
        remote_archive = (
            f"{remote_dir.rstrip('/')}/.sageattention-artifacts-"
            f"{uuid.uuid4().hex}.tar.gz"
        )
        checks = " ".join(shlex.quote(path) for path in paths)
        script = (
            "set -euo pipefail; "
            f"cd {shlex.quote(remote_dir)}; "
            f"for item in {checks}; do test -e \"$item\" || "
            "{ echo \"missing requested artifact: $item\" >&2; exit 20; }; done; "
            f"tar -czf {shlex.quote(remote_archive)} -- {checks}"
        )
        self._run_remote(endpoint, script, timeout=deadline.timeout(600))

        output_dir.mkdir(parents=True, exist_ok=True)
        fd, local_name = tempfile.mkstemp(prefix="sageattention-artifacts-", suffix=".tar.gz")
        os.close(fd)
        local_archive = Path(local_name)
        try:
            self._run(
                [
                    "scp",
                    "-i",
                    str(self.private_key),
                    "-P",
                    str(endpoint.port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    f"UserKnownHostsFile={self.known_hosts}",
                    f"root@{endpoint.host}:{remote_archive}",
                    str(local_archive),
                ],
                timeout=deadline.timeout(900),
            )
            _safe_extract(local_archive, output_dir)
        finally:
            local_archive.unlink(missing_ok=True)
            try:
                self._run_remote(
                    endpoint,
                    f"rm -f {shlex.quote(remote_archive)}",
                    timeout=min(30.0, max(1.0, deadline.seconds_left())),
                    check=False,
                )
            except JobError:
                pass


def _make_repo_archive(repo_root: Path) -> Path:
    root = repo_root.resolve()
    fd, name = tempfile.mkstemp(prefix="sageattention-source-", suffix=".tar.gz")
    os.close(fd)
    archive_path = Path(name)
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

    def filter_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if any(part in excluded for part in PurePosixPath(info.name).parts):
            return None
        if info.issym() or info.islnk() or info.isdev():
            raise JobError(
                f"repository contains an unsafe archive member: {info.name!r}"
            )
        return info

    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.dereference = False
            archive.add(root, arcname=".", filter=filter_member)
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a trusted Pod result while rejecting traversal and links."""

    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise JobError(f"unsafe artifact archive member: {member.name!r}")
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise JobError(
                    f"artifact archive escapes output directory: {member.name!r}"
                ) from exc
        archive.extractall(destination)


_TERMINAL_STATES = {"ERROR", "EXITED", "TERMINATED"}


def wait_for_ssh(
    ctl: RunpodApi,
    transport: SSHTransport,
    pod_id: str,
    deadline: Deadline,
    *,
    poll_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Endpoint:
    """Wait for assignment and confirm that the SSH service accepts a key."""

    last_status: str | None = None
    while True:
        details = ctl.pod_details(pod_id, timeout=deadline.timeout(30))
        status = _pod_status(details)
        if status != last_status:
            print(f"Pod {pod_id}: status={status}", flush=True)
            last_status = status
        if status in _TERMINAL_STATES:
            raise JobError(f"Pod {pod_id} entered terminal state {status}")
        endpoint = _extract_ssh_endpoint(details) if status == "RUNNING" else None
        if endpoint and transport.probe(endpoint, timeout=deadline.timeout(15)):
            print(
                f"Pod {pod_id}: SSH ready at {endpoint.host}:{endpoint.port}",
                flush=True,
            )
            return endpoint
        sleep(min(max(0.1, poll_seconds), deadline.timeout()))


def _compute(details: Mapping[str, object], kind: str) -> Mapping[str, object]:
    """Return the REST v2 ``gpu`` or ``cpu`` block, or an empty mapping."""

    block = details.get(kind)
    return block if isinstance(block, Mapping) else {}


def _persistent_volume_gb(details: Mapping[str, object]) -> object:
    """Return the Pod volume size, or 0 when no persistent mount exists."""

    mounts = details.get("mounts")
    persistent = mounts.get("persistent") if isinstance(mounts, Mapping) else None
    if persistent is None:
        return 0
    return persistent.get("size") if isinstance(persistent, Mapping) else persistent


def _finite_decimal(details: Mapping[str, object], name: str) -> Decimal | None:
    value = details.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _integer(details: Mapping[str, object], name: str) -> int | None:
    parsed = _finite_decimal(details, name)
    if parsed is None or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _memory_bytes(details: Mapping[str, object], name: str) -> tuple[Decimal, int] | None:
    memory_gb = _finite_decimal(details, name)
    if memory_gb is None or memory_gb <= 0:
        return None
    memory_bytes = memory_gb * Decimal(1_000_000_000)
    if memory_bytes != memory_bytes.to_integral_value():
        return None
    return memory_gb, int(memory_bytes)


def verify_pod_assignment(
    ctl: RunpodApi,
    pod_id: str,
    request: PodRequest,
    deadline: Deadline,
) -> VerifiedBuildAssignment | None:
    """Refuse image, placement, or build-resource drift before source upload."""

    details = ctl.assignment_details(pod_id, timeout=deadline.timeout(30))
    actual_image = details.get("image")
    if actual_image != request.image:
        raise JobError(
            f"Pod image mismatch: expected {request.image!r}, got {actual_image!r}"
        )
    is_gpu = request.compute_type.upper() == "GPU"
    # Drift on a GPU candidate is retryable on the next candidate; on the
    # single CPU placement it is fatal.
    drift_error: type[JobError] = GpuPlacementError if is_gpu else JobError
    if is_gpu:
        gpu = _compute(details, "gpu")
        gpu_type_id = gpu.get("id")
        if gpu_type_id != request.gpu_id:
            raise GpuPlacementError(
                f"GPU type mismatch: expected exact gpuId {request.gpu_id!r}, "
                f"got {gpu_type_id!r}"
            )
        if not request.is_gpu_build:
            print(
                f"Pod {pod_id}: verified image and gpu_id={gpu_type_id}",
                flush=True,
            )
            return None
        compute = gpu
        minimum_vcpus = request.gpu_min_vcpu_count
        minimum_memory_gb = request.gpu_min_memory_gb
        placement = f"gpu_id={gpu_type_id}"
        label = "GPU build "
    else:
        compute = _compute(details, "cpu")
        flavor = compute.get("id")
        if flavor not in request.cpu_flavor_ids:
            raise JobError(
                "CPU flavor mismatch: expected one of "
                f"{request.cpu_flavor_ids!r}, got {flavor!r}"
            )
        minimum_vcpus = request.cpu_vcpu_count
        minimum_memory_gb = request.cpu_min_memory_gb
        placement = f"flavor={flavor}"
        label = ""

    vcpus = _integer(compute, "vcpuCount")
    if vcpus is None or vcpus < minimum_vcpus:
        raise drift_error(
            f"{label}CPU count mismatch: required at least {minimum_vcpus}, "
            f"got {compute.get('vcpuCount')!r}"
        )
    memory = _memory_bytes(compute, "memory")
    if memory is None or memory[0] < minimum_memory_gb:
        raise drift_error(
            f"{label}memory mismatch: required at least {minimum_memory_gb} GB, "
            f"got {compute.get('memory')!r}"
        )
    memory_gb, assigned_memory_bytes = memory
    container_disk = _integer(details, "disk")
    if container_disk is None or container_disk < request.container_disk_gb:
        raise drift_error(
            f"container disk mismatch: requested at least "
            f"{request.container_disk_gb} GB, got {details.get('disk')!r}"
        )
    pod_volume = _persistent_volume_gb(details)
    if pod_volume != 0:
        raise drift_error(
            f"unexpected paid Pod volume: requested 0 GB, got {pod_volume!r}"
        )
    print(
        f"Pod {pod_id}: verified image, {placement}, "
        f"vcpus={vcpus}, memory={memory_gb:g} GB, "
        f"container_disk={container_disk} GB, pod_volume=0 GB",
        flush=True,
    )
    return VerifiedBuildAssignment(vcpus, assigned_memory_bytes)


def _terminate_strictly(ctl: RunpodApi, pod_id: str) -> Exception | None:
    """Try deletion repeatedly within a small cleanup-only grace window."""

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            ctl.terminate_pod(pod_id, timeout=20)
            print(f"Pod {pod_id}: terminated", flush=True)
            return None
        except Exception as exc:  # cleanup must retain and report every failure
            last_error = exc
            print(
                f"Pod {pod_id}: cleanup attempt {attempt}/3 failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < 3:
                time.sleep(2)
    return last_error


RESOURCE_RECEIPT_PATH = "/run/sageattention/verified-memory-bytes-v1"


def _resource_receipt_prefix(assigned_memory_bytes: int) -> str:
    """Atomically publish the verified presentation budget inside one Pod."""

    if assigned_memory_bytes <= 0:
        raise ValueError("assigned_memory_bytes must be positive")
    return (
        "install -d -m 0755 /run/sageattention; "
        "resource_receipt=$(mktemp "
        "/run/sageattention/.verified-memory-bytes-v1.XXXXXX); "
        f"printf '%s\\n' {assigned_memory_bytes} > \"$resource_receipt\"; "
        "chmod 0444 \"$resource_receipt\"; "
        f"mv -f -- \"$resource_receipt\" {RESOURCE_RECEIPT_PATH}; "
        "unset resource_receipt; "
    )


def _start_placement_round(spec: JobSpec, round_number: int, kind: str) -> Deadline:
    """Wait out the backoff before a retry and start that round's deadline.

    Runpod capacity comes and goes within minutes, and waiting for it costs
    nothing because no Pod exists yet. The hard timeout therefore bounds the
    work on a Pod, not the wait for one: each round starts it afresh.
    """

    if round_number > 1:
        print(
            f"No {kind} capacity in placement round {round_number - 1}/"
            f"{spec.placement_rounds}; retrying in "
            f"{spec.placement_backoff_seconds:g}s",
            flush=True,
        )
        time.sleep(spec.placement_backoff_seconds)
    return Deadline(spec.hard_timeout_seconds)


def _gpu_candidates(spec: JobSpec) -> tuple[str, ...]:
    if spec.pod.compute_type.upper() != "GPU":
        return ()
    return spec.gpu_id_candidates or (spec.pod.gpu_id or "",)


def run_job(
    spec: JobSpec,
    *,
    ctl: RunpodApi | None = None,
    transport: SSHTransport | None = None,
    wait_fn: Callable[..., Endpoint] = wait_for_ssh,
) -> JobResult:
    """Run a job and make Pod deletion mandatory on every post-create path."""

    spec.validate()
    is_build_job = spec.pod.compute_type.upper() == "CPU" or spec.pod.is_gpu_build
    deadline = Deadline(spec.hard_timeout_seconds)
    ctl = ctl or RunpodApi()
    transport = transport or SSHTransport(spec.ssh_private_key)
    pod_id: str | None = None
    endpoint: Endpoint | None = None
    selected_gpu_id: str | None = None
    verified_build_assignment: VerifiedBuildAssignment | None = None
    repo_uploaded = False
    primary_error: Exception | None = None
    artifact_error: Exception | None = None

    try:
        ctl.check_auth(timeout=deadline.timeout(30))
        self_terminate_seconds = (
            spec.hard_timeout_seconds + spec.terminate_grace_seconds
        )
        if spec.pod.compute_type.upper() == "CPU":
            cpu_failures: list[str] = []
            for round_number in range(1, spec.placement_rounds + 1):
                deadline = _start_placement_round(spec, round_number, "CPU")
                print(
                    f"CPU placement round {round_number}/{spec.placement_rounds}: "
                    f"requesting Pod {spec.pod.name!r}; "
                    f"self-delete after {self_terminate_seconds}s",
                    flush=True,
                )
                try:
                    pod_id = ctl.create_pod(
                        spec.pod,
                        self_terminate_seconds=self_terminate_seconds,
                        timeout=deadline.timeout(120),
                    )
                except CapacityUnavailableError as exc:
                    cpu_failures.append(f"round {round_number}: {exc}")
                    print(
                        f"CPU placement round {round_number} has no matching capacity",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                break
            if pod_id is None:
                raise JobError(
                    f"CPU placement exhausted {spec.placement_rounds} rounds "
                    "before upload: " + "; ".join(cpu_failures)
                )
            print(
                f"Created CPU Pod {pod_id}; in-Pod self-delete after "
                f"{self_terminate_seconds}s",
                flush=True,
            )
            endpoint = wait_fn(
                ctl,
                transport,
                pod_id,
                deadline,
                poll_seconds=spec.poll_seconds,
            )
            verified_build_assignment = verify_pod_assignment(
                ctl,
                pod_id,
                spec.pod,
                deadline,
            )
        else:
            candidates = _gpu_candidates(spec)
            placement_failures: list[str] = []
            for round_number in range(1, spec.placement_rounds + 1):
                deadline = _start_placement_round(spec, round_number, "GPU")
                round_only_capacity_failures = True
                for candidate_number, candidate in enumerate(candidates, start=1):
                    candidate_request = replace(spec.pod, gpu_id=candidate)
                    pod_id = None
                    endpoint = None
                    print(
                        f"Trying GPU candidate {candidate_number}/{len(candidates)} "
                        f"(round {round_number}/{spec.placement_rounds}): "
                        f"requesting exact gpuId {candidate!r}",
                        flush=True,
                    )
                    try:
                        pod_id = ctl.create_pod(
                            candidate_request,
                            self_terminate_seconds=self_terminate_seconds,
                            timeout=deadline.timeout(120),
                        )
                    except CapacityUnavailableError as exc:
                        placement_failures.append(
                            f"round {round_number} {candidate!r}: {exc}"
                        )
                        print(
                            f"GPU candidate {candidate_number}/{len(candidates)} "
                            f"{candidate!r} has no matching capacity",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    print(
                        f"Created GPU Pod {pod_id} for candidate {candidate!r}",
                        flush=True,
                    )
                    endpoint = wait_fn(
                        ctl,
                        transport,
                        pod_id,
                        deadline,
                        poll_seconds=spec.poll_seconds,
                    )
                    try:
                        candidate_assignment = verify_pod_assignment(
                            ctl,
                            pod_id,
                            candidate_request,
                            deadline,
                        )
                    except GpuPlacementError as exc:
                        round_only_capacity_failures = False
                        placement_failures.append(
                            f"round {round_number} {candidate!r}: {exc}"
                        )
                        print(
                            f"GPU candidate {candidate!r} was rejected "
                            f"before upload: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        rejected_cleanup_error = _terminate_strictly(ctl, pod_id)
                        if rejected_cleanup_error is not None:
                            raise JobError(
                                "refusing GPU candidate fallback because rejected "
                                f"Pod {pod_id} could not be deleted: "
                                f"{rejected_cleanup_error}"
                            ) from exc
                        pod_id = None
                        endpoint = None
                        continue

                    selected_gpu_id = candidate
                    verified_build_assignment = candidate_assignment
                    print(
                        f"Selected GPU candidate {candidate!r} for Pod {pod_id} "
                        f"(round {round_number})",
                        flush=True,
                    )
                    break
                if selected_gpu_id is not None:
                    break
                if not round_only_capacity_failures:
                    break

            if selected_gpu_id is None:
                detail = "; ".join(placement_failures)
                raise JobError(
                    f"GPU placement exhausted {spec.placement_rounds} ordered "
                    "rounds before upload"
                    + (f": {detail}" if detail else "")
                )

        assert pod_id is not None and endpoint is not None
        if is_build_job and verified_build_assignment is None:
            raise JobError("build Pod assignment did not provide verified CPU/RAM")
        print(
            f"Pod {pod_id}: uploading repository to {spec.remote_dir}",
            flush=True,
        )
        transport.upload_repo(
            endpoint,
            spec.repo_root,
            spec.remote_dir,
            deadline,
        )
        repo_uploaded = True
        print(f"Pod {pod_id}: repository upload complete", flush=True)
        command_count = len(spec.commands)
        for command_number, command in enumerate(spec.commands, start=1):
            print(
                f"Pod {pod_id}: starting remote command "
                f"{command_number}/{command_count}",
                flush=True,
            )
            remote_exports: list[tuple[str, str]] = []
            receipt_prefix = ""
            if is_build_job:
                assert verified_build_assignment is not None
                receipt_prefix = _resource_receipt_prefix(
                    verified_build_assignment.assigned_memory_bytes
                )
                remote_exports.extend(
                    (
                        (
                            "RUNPOD_CPU_COUNT",
                            str(verified_build_assignment.vcpu_count),
                        ),
                        (
                            "RUNPOD_ASSIGNED_MEMORY_BYTES",
                            str(verified_build_assignment.assigned_memory_bytes),
                        ),
                    )
                )
            if selected_gpu_id is not None:
                remote_exports.append(("RUNPOD_SELECTED_GPU_ID", selected_gpu_id))
            remote_command = command
            if remote_exports:
                export_prefix = "; ".join(
                    f"export {name}={shlex.quote(value)}"
                    for name, value in remote_exports
                )
                remote_command = f"{receipt_prefix}{export_prefix}; {command}"
            transport.run_script(
                endpoint,
                spec.remote_dir,
                remote_command,
                deadline,
            )
            print(
                f"Pod {pod_id}: remote command "
                f"{command_number}/{command_count} complete",
                flush=True,
            )
    except Exception as exc:
        primary_error = exc
    finally:
        if (
            endpoint
            and repo_uploaded
            and spec.artifact_paths
            and deadline.seconds_left() > 1
        ):
            try:
                print(
                    f"Pod {pod_id}: downloading requested artifacts",
                    flush=True,
                )
                transport.download_artifacts(
                    endpoint,
                    spec.remote_dir,
                    spec.artifact_paths,
                    spec.artifact_output,
                    deadline,
                )
                print(f"Pod {pod_id}: artifact download complete", flush=True)
            except Exception as exc:
                artifact_error = exc
        cleanup_error = _terminate_strictly(ctl, pod_id) if pod_id else None
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            close_transport()

    if (
        primary_error is not None
        and artifact_error is not None
        and "missing requested artifact:" in str(artifact_error)
    ):
        print(
            "WARNING: no requested debug artifact was produced after the job "
            "failed; preserving the primary job error",
            file=sys.stderr,
            flush=True,
        )
        artifact_error = None

    failures = [
        ("job", primary_error),
        ("artifact download", artifact_error),
        ("Pod cleanup", cleanup_error),
    ]
    present = [(label, error) for label, error in failures if error is not None]
    if present:
        detail = "; ".join(f"{label}: {error}" for label, error in present)
        raise JobError(detail) from primary_error
    assert pod_id is not None and endpoint is not None
    return JobResult(
        pod_id,
        endpoint,
        spec.artifact_output,
        selected_gpu_id=selected_gpu_id,
    )


def _read_public_key(args: argparse.Namespace) -> str:
    if args.public_key_file:
        return Path(args.public_key_file).read_text(encoding="utf-8").strip()
    return os.environ.get(args.public_key_env, "").strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="immutable builder/test image ref")
    parser.add_argument("--name", required=True, help="short unique Pod name")
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--gpu-id",
        action="append",
        metavar="GPU_ID",
        help=(
            "exact Runpod gpuId; required in GPU mode and repeatable up to 8 "
            "times in ordered fallback priority"
        ),
    )
    parser.add_argument("--min-cuda-version")
    parser.add_argument(
        "--gpu-workload",
        choices=("validation", "build"),
        default="validation",
        help="GPU Pod contract; build enables strict compilation resource checks",
    )
    parser.add_argument(
        "--gpu-min-vcpu-count",
        type=int,
        default=4,
        help="GPU build minimum vCPUs (4 required; 16 recommended)",
    )
    parser.add_argument(
        "--gpu-min-memory-gb",
        type=int,
        default=32,
        help="GPU build minimum system RAM (32 GB required; 64 GB recommended)",
    )
    parser.add_argument("--cloud-type", choices=("COMMUNITY", "SECURE"), default="COMMUNITY")
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument(
        "--cpu-flavor-ids",
        default="cpu3g",
        help="comma-separated exact CPU flavor ids, in priority order",
    )
    parser.add_argument(
        "--cpu-vcpu-count",
        type=int,
        default=16,
        help="CPU Pod vCPUs; cpu3g/16 targets the recommended 64 GB RAM",
    )
    parser.add_argument("--cpu-min-memory-gb", type=int, default=32)
    parser.add_argument("--registry-auth-id")
    parser.add_argument("--data-center-ids")
    parser.add_argument("--public-key-env", default="RUNPOD_SSH_PUBLIC_KEY")
    parser.add_argument("--public-key-file")
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument(
        "--remote-dir",
        help=(
            "remote checkout directory; defaults to /work/sageattention-factory "
            "for CPU/GPU builds and /workspace for GPU validation"
        ),
    )
    parser.add_argument(
        "--command",
        action="append",
        required=True,
        help="remote bash fragment; repeat to run phases in order",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="relative remote path to retrieve; repeat as needed",
    )
    parser.add_argument("--artifact-output", default="artifacts", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--terminate-grace-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--placement-rounds",
        type=int,
        default=5,
        help="placement attempts when every candidate lacks capacity (1-10)",
    )
    parser.add_argument(
        "--placement-backoff-seconds",
        type=float,
        default=300.0,
        help="wait between placement rounds; not counted in --timeout-seconds",
    )
    parser.add_argument(
        "--allow-paid-pod",
        action="store_true",
        help="mandatory acknowledgement; approved workflows only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    previous_handlers: dict[signal.Signals, object] = {}

    def request_shutdown(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        raise JobError(f"received {name}; terminating the active Pod")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    try:
        gpu_id_candidates = tuple(args.gpu_id or ())
        request = PodRequest(
            image=args.image,
            name=args.name,
            public_key=_read_public_key(args),
            compute_type=args.mode.upper(),
            gpu_id=gpu_id_candidates[0] if gpu_id_candidates else None,
            cloud_type=args.cloud_type,
            container_disk_gb=args.container_disk_gb,
            min_cuda_version=args.min_cuda_version,
            gpu_workload=args.gpu_workload.upper(),
            gpu_min_vcpu_count=args.gpu_min_vcpu_count,
            gpu_min_memory_gb=args.gpu_min_memory_gb,
            registry_auth_id=args.registry_auth_id,
            data_center_ids=args.data_center_ids,
            cpu_flavor_ids=tuple(
                value.strip()
                for value in args.cpu_flavor_ids.split(",")
                if value.strip()
            ),
            cpu_vcpu_count=args.cpu_vcpu_count,
            cpu_min_memory_gb=args.cpu_min_memory_gb,
        )
        spec = JobSpec(
            pod=request,
            repo_root=args.repo.resolve(),
            ssh_private_key=args.ssh_key.resolve(),
            remote_dir=args.remote_dir
            or (
                "/work/sageattention-factory"
                if args.mode == "cpu" or args.gpu_workload == "build"
                else "/workspace"
            ),
            commands=tuple(args.command),
            artifact_paths=tuple(args.artifact),
            artifact_output=args.artifact_output.resolve(),
            gpu_id_candidates=(
                gpu_id_candidates if args.mode == "gpu" else ()
            ),
            hard_timeout_seconds=args.timeout_seconds,
            terminate_grace_seconds=args.terminate_grace_seconds,
            poll_seconds=args.poll_seconds,
            allow_paid_pod=args.allow_paid_pod,
            placement_rounds=args.placement_rounds,
            placement_backoff_seconds=args.placement_backoff_seconds,
        )
        result = run_job(spec)
    except (JobError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    result_payload = {
        "pod_id": result.pod_id,
        "ssh_host": result.endpoint.host,
        "ssh_port": result.endpoint.port,
        "artifact_output": str(result.artifact_output),
    }
    if result.selected_gpu_id is not None:
        result_payload["selected_gpu_id"] = result.selected_gpu_id
    print(json.dumps(result_payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
