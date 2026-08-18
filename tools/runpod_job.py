#!/usr/bin/env python3
"""Run one bounded build or test job on an ephemeral Runpod Pod.

The module intentionally depends only on Python's standard library and the
``runpodctl``, ``ssh`` and ``scp`` executables.  GitHub Actions is an
orchestrator only: the checked-out repository is uploaded to the Pod, commands
run there, requested artifacts are copied back, and the Pod is deleted in a
``finally`` path.

Creating a paid Pod requires the explicit ``--allow-paid-pod`` flag.  CI and
pull-request workflows must never pass that flag; manually approved build and
release workflows may.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence


class JobError(RuntimeError):
    """Base class for an orchestration failure."""


class JobTimeout(JobError):
    """The hard wall-clock budget was exhausted."""


class CommandError(JobError):
    """A local, SSH, or runpodctl command failed."""


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


@dataclass(frozen=True)
class JobResult:
    pod_id: str
    endpoint: Endpoint
    artifact_output: Path
    selected_gpu_id: str | None = None


Executor = Callable[..., subprocess.CompletedProcess[str]]
HttpExecutor = Callable[
    [str, str, Mapping[str, object] | None, Mapping[str, str], float], object
]


def _completed_text(proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip() for part in (proc.stdout or "", proc.stderr or "") if part.strip()
    )


def _is_capacity_unavailable_error(error: CommandError) -> bool:
    """Match only Runpod's explicit no-instance placement response."""

    normalized = " ".join(str(error).casefold().split())
    return any(
        message in normalized
        for message in (
            "there are no longer any instances available with the requested "
            "specifications",
            "there are no instances available with the requested specifications",
        )
    )


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
        raise CommandError(
            f"Runpod REST {method} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise CommandError(f"Runpod REST {method} failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"Runpod REST returned invalid JSON: {raw[:200]!r}"
        ) from exc


class Runpodctl:
    """Small checked subprocess wrapper around ``runpodctl``."""

    def __init__(
        self,
        binary: str = "runpodctl",
        *,
        executor: Executor = subprocess.run,
        env: Mapping[str, str] | None = None,
        http_executor: HttpExecutor = _http_json,
        rest_api_url: str = "https://rest.runpod.io/v1",
    ) -> None:
        self.binary = binary
        self._executor = executor
        self._env = dict(env) if env is not None else None
        self._http_executor = http_executor
        self._rest_api_url = rest_api_url.rstrip("/")

    def call(
        self,
        *args: str,
        timeout: float = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = [self.binary, *args]
        try:
            proc = self._executor(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=self._env,
            )
        except FileNotFoundError as exc:
            raise CommandError(f"{self.binary} is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandError(
                f"runpodctl command timed out after {timeout:.0f}s"
            ) from exc
        if check and proc.returncode != 0:
            detail = _completed_text(proc) or f"exit code {proc.returncode}"
            raise CommandError(f"runpodctl {' '.join(args[:2])} failed: {detail}")
        return proc

    def call_json(self, *args: str, timeout: float = 60) -> object:
        proc = self.call(*args, "-o", "json", timeout=timeout)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise CommandError(
                f"runpodctl returned invalid JSON: {(proc.stdout or '')[:200]}"
            ) from exc

    def check_auth(self, *, timeout: float = 30) -> None:
        self.call("user", timeout=timeout)

    def _api_key(self) -> str:
        environment = self._env if self._env is not None else os.environ
        api_key = environment.get("RUNPOD_API_KEY", "").strip()
        if not api_key:
            raise CommandError("RUNPOD_API_KEY is required for the Runpod REST API")
        return api_key

    def _rest_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    def create_pod(
        self,
        request: PodRequest,
        *,
        terminate_after: str,
        self_terminate_seconds: int,
        timeout: float = 120,
    ) -> str:
        request.validate()
        if self_terminate_seconds < 600 or self_terminate_seconds > 21_600:
            raise ValueError(
                "self-termination deadline must be between 600 and 21600 seconds"
            )
        if request.compute_type.upper() == "CPU":
            return self._create_cpu_pod_rest(
                request,
                self_terminate_seconds=self_terminate_seconds,
                timeout=timeout,
            )

        args = [
            "pod",
            "create",
            "--image",
            request.image,
            "--name",
            request.name,
            "--cloud-type",
            request.cloud_type,
            "--container-disk-in-gb",
            str(request.container_disk_gb),
            "--ports",
            "22/tcp",
            "--terminate-after",
            terminate_after,
            "--env",
            json.dumps(
                {"PUBLIC_KEY": request.public_key.strip()},
                separators=(",", ":"),
            ),
        ]
        if request.is_gpu_build:
            # Keep compilation on the explicitly sized ephemeral container
            # disk.  The pinned CLI path is retained so --terminate-after is
            # still a platform-side backstop even if the runner disappears.
            args.extend(["--volume-in-gb", "0"])
        if request.data_center_ids:
            args.extend(["--data-center-ids", request.data_center_ids])
        if request.registry_auth_id:
            args.extend(["--registry-auth-id", request.registry_auth_id])
        if request.cloud_type.upper() == "COMMUNITY":
            args.append("--public-ip")
        args.extend(["--gpu-id", request.gpu_id or "", "--gpu-count", "1"])
        if request.min_cuda_version:
            args.extend(["--min-cuda-version", request.min_cuda_version])

        try:
            payload = self.call_json(*args, timeout=timeout)
        except CommandError as exc:
            if _is_capacity_unavailable_error(exc):
                raise CapacityUnavailableError(str(exc)) from exc
            raise
        if not isinstance(payload, dict):
            raise CommandError("runpodctl pod create returned a non-object response")
        nested = payload.get("pod")
        pod_id = payload.get("id") or (
            nested.get("id") if isinstance(nested, dict) else None
        )
        if not isinstance(pod_id, str) or not pod_id:
            raise CommandError("runpodctl pod create response did not contain a Pod id")
        return pod_id

    def _create_cpu_pod_rest(
        self,
        request: PodRequest,
        *,
        self_terminate_seconds: int,
        timeout: float,
    ) -> str:
        """Create a sized CPU Pod, which runpodctl v2.3.0 cannot express."""

        payload: dict[str, object] = {
            "name": request.name,
            "imageName": request.image,
            "computeType": "CPU",
            "cloudType": request.cloud_type.upper(),
            "containerDiskInGb": request.container_disk_gb,
            # The REST service otherwise provisions its default /workspace
            # Pod volume. CPU compilation is intentionally ephemeral under
            # /work on containerDiskInGb, so explicitly disable that volume.
            "volumeInGb": 0,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": request.cloud_type.upper() == "COMMUNITY",
            "cpuFlavorIds": list(request.cpu_flavor_ids),
            "cpuFlavorPriority": "custom",
            "vcpuCount": request.cpu_vcpu_count,
            "env": {
                "PUBLIC_KEY": request.public_key.strip(),
                "RUNPOD_SELF_TERMINATE_SECONDS": str(self_terminate_seconds),
            },
        }
        if request.data_center_ids:
            payload["dataCenterIds"] = [
                value.strip()
                for value in request.data_center_ids.split(",")
                if value.strip()
            ]
            payload["dataCenterPriority"] = "custom"
        if request.registry_auth_id:
            payload["containerRegistryAuthId"] = request.registry_auth_id

        response = self._http_executor(
            "POST",
            f"{self._rest_api_url}/pods",
            payload,
            self._rest_headers(),
            timeout,
        )
        if not isinstance(response, dict):
            raise CommandError("Runpod REST Pod create returned a non-object response")
        pod_id = response.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise CommandError("Runpod REST Pod create response did not contain a Pod id")
        return pod_id

    def assignment_details(
        self,
        pod_id: str,
        *,
        timeout: float = 30,
    ) -> dict[str, object]:
        payload = self._http_executor(
            "GET",
            f"{self._rest_api_url}/pods/{pod_id}?includeMachine=true",
            None,
            self._rest_headers(),
            timeout,
        )
        if not isinstance(payload, dict):
            raise CommandError("Runpod REST Pod get returned a non-object response")
        return payload

    def pod_details(self, pod_id: str, *, timeout: float = 30) -> dict[str, object]:
        payload = self.call_json("pod", "get", pod_id, timeout=timeout)
        if not isinstance(payload, dict):
            raise CommandError("runpodctl pod get returned a non-object response")
        return payload

    def terminate_pod(self, pod_id: str, *, timeout: float = 30) -> None:
        self.call("pod", "delete", pod_id, timeout=timeout)


def _validate_remote_dir(path: str) -> None:
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or parsed == PurePosixPath("/") or ".." in parsed.parts:
        raise ValueError("remote_dir must be a safe absolute directory below /")


def _validate_relative_remote_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts or path in {".", "./"}:
        raise ValueError(f"unsafe relative artifact path: {path!r}")


def _extract_ssh_endpoint(payload: Mapping[str, object]) -> Endpoint | None:
    """Accept both current runpodctl SSH output and older runtime port maps."""

    ssh = payload.get("ssh")
    if isinstance(ssh, Mapping):
        host = ssh.get("ip") or ssh.get("host")
        port = ssh.get("port")
        if host and port:
            try:
                return Endpoint(str(host), int(port))
            except (TypeError, ValueError):
                pass

    runtime = payload.get("runtime")
    if isinstance(runtime, Mapping):
        ports = runtime.get("ports")
        if isinstance(ports, list):
            for item in ports:
                if not isinstance(item, Mapping):
                    continue
                private_port = item.get("privatePort") or item.get("containerPort")
                if str(private_port) != "22":
                    continue
                host = item.get("ip") or item.get("host")
                port = item.get("publicPort") or item.get("port")
                if host and port:
                    try:
                        return Endpoint(str(host), int(port))
                    except (TypeError, ValueError):
                        continue
    public_ip = payload.get("publicIp")
    mappings = payload.get("portMappings")
    if public_ip and isinstance(mappings, Mapping):
        public_port = mappings.get("22") or mappings.get(22)
        if public_port:
            try:
                return Endpoint(str(public_ip), int(public_port))
            except (TypeError, ValueError):
                pass
    return None


def _pod_status(payload: Mapping[str, object]) -> str:
    status = payload.get("status") or payload.get("desiredStatus")
    if status is None and isinstance(payload.get("pod"), Mapping):
        status = payload["pod"].get("status")  # type: ignore[index]
    return str(status or "UNKNOWN").upper()


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
        proc = self._run_remote(
            endpoint,
            wrapped,
            timeout=deadline.timeout(),
        )
        if proc.stdout:
            print(
                proc.stdout,
                end="" if proc.stdout.endswith("\n") else "\n",
                flush=True,
            )
        if proc.stderr:
            print(
                proc.stderr,
                file=sys.stderr,
                end="" if proc.stderr.endswith("\n") else "\n",
                flush=True,
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


_TERMINAL_STATES = {"DEAD", "ERROR", "EXITED", "FAILED", "TERMINATED"}


def wait_for_ssh(
    ctl: Runpodctl,
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
        endpoint = _extract_ssh_endpoint(details)
        if endpoint and transport.probe(endpoint, timeout=deadline.timeout(15)):
            print(
                f"Pod {pod_id}: SSH ready at {endpoint.host}:{endpoint.port}",
                flush=True,
            )
            return endpoint
        sleep(min(max(0.1, poll_seconds), deadline.timeout()))


def _assignment_payload(details: Mapping[str, object]) -> Mapping[str, object]:
    nested = details.get("pod")
    return nested if isinstance(nested, Mapping) else details


def _number(details: Mapping[str, object], name: str) -> float | None:
    value = details.get(name)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gpu_type_ids(details: Mapping[str, object]) -> tuple[str, ...]:
    """Collect exact GPU ids from documented current and legacy shapes."""

    found: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in found:
            found.append(value)

    add(details.get("gpuTypeId"))
    gpu = details.get("gpu")
    if isinstance(gpu, Mapping):
        add(gpu.get("id"))
    machine = details.get("machine")
    if isinstance(machine, Mapping):
        add(machine.get("gpuTypeId"))
        machine_gpu = machine.get("gpuType")
        if isinstance(machine_gpu, Mapping):
            add(machine_gpu.get("id"))
    return tuple(found)


def verify_pod_assignment(
    ctl: Runpodctl,
    pod_id: str,
    request: PodRequest,
    deadline: Deadline,
) -> None:
    """Refuse image, placement, or build-resource drift before source upload."""

    details = _assignment_payload(
        ctl.assignment_details(pod_id, timeout=deadline.timeout(30))
    )
    actual_image = details.get("imageName") or details.get("image")
    if actual_image != request.image:
        raise JobError(
            f"Pod image mismatch: expected {request.image!r}, got {actual_image!r}"
        )
    if request.compute_type.upper() == "GPU":
        gpu_type_ids = _gpu_type_ids(details)
        if not gpu_type_ids or any(
            gpu_type_id != request.gpu_id for gpu_type_id in gpu_type_ids
        ):
            raise GpuPlacementError(
                f"GPU type mismatch: expected exact gpuId {request.gpu_id!r}, "
                f"got {gpu_type_ids or None!r}"
            )
        gpu_type_id = gpu_type_ids[0]
        if not request.is_gpu_build:
            print(
                f"Pod {pod_id}: verified image and gpu_id={gpu_type_id}",
                flush=True,
            )
            return

        vcpus = _number(details, "vcpuCount")
        if vcpus is None or vcpus < request.gpu_min_vcpu_count:
            raise GpuPlacementError(
                f"GPU build CPU count mismatch: required at least "
                f"{request.gpu_min_vcpu_count}, got {details.get('vcpuCount')!r}"
            )
        memory = _number(details, "memoryInGb")
        if memory is None or memory < request.gpu_min_memory_gb:
            raise GpuPlacementError(
                f"GPU build memory mismatch: required at least "
                f"{request.gpu_min_memory_gb} GB, "
                f"got {details.get('memoryInGb')!r}"
            )
        container_disk = _number(details, "containerDiskInGb")
        if container_disk is None or container_disk < request.container_disk_gb:
            raise GpuPlacementError(
                f"container disk mismatch: requested at least "
                f"{request.container_disk_gb} GB, "
                f"got {details.get('containerDiskInGb')!r}"
            )
        pod_volume = _number(details, "volumeInGb")
        if pod_volume != 0:
            raise GpuPlacementError(
                "unexpected paid Pod volume: requested 0 GB, "
                f"got {details.get('volumeInGb')!r}"
            )
        print(
            f"Pod {pod_id}: verified image, gpu_id={gpu_type_id}, "
            f"vcpus={vcpus:g}, memory={memory:g} GB, "
            f"container_disk={container_disk:g} GB, pod_volume=0 GB",
            flush=True,
        )
        return

    flavor = details.get("cpuFlavorId")
    if flavor not in request.cpu_flavor_ids:
        raise JobError(
            "CPU flavor mismatch: expected one of "
            f"{request.cpu_flavor_ids!r}, got {flavor!r}"
        )
    vcpus = _number(details, "vcpuCount")
    if vcpus is None or vcpus < request.cpu_vcpu_count:
        raise JobError(
            f"CPU count mismatch: requested at least {request.cpu_vcpu_count}, "
            f"got {details.get('vcpuCount')!r}"
        )
    memory = _number(details, "memoryInGb")
    if memory is None or memory < request.cpu_min_memory_gb:
        raise JobError(
            f"memory mismatch: required at least {request.cpu_min_memory_gb} GB, "
            f"got {details.get('memoryInGb')!r}"
        )
    container_disk = _number(details, "containerDiskInGb")
    if container_disk is None or container_disk < request.container_disk_gb:
        raise JobError(
            f"container disk mismatch: requested at least "
            f"{request.container_disk_gb} GB, "
            f"got {details.get('containerDiskInGb')!r}"
        )
    pod_volume = _number(details, "volumeInGb")
    if pod_volume != 0:
        raise JobError(
            "unexpected paid Pod volume: requested 0 GB, "
            f"got {details.get('volumeInGb')!r}"
        )
    print(
        f"Pod {pod_id}: verified image, flavor={flavor}, "
        f"vcpus={vcpus:g}, memory={memory:g} GB, "
        f"container_disk={container_disk:g} GB, pod_volume=0 GB",
        flush=True,
    )


def _terminate_strictly(ctl: Runpodctl, pod_id: str) -> Exception | None:
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


def _rfc3339_after(
    seconds: int,
    *,
    now: datetime | None = None,
) -> str:
    """Return the RFC3339 datetime required by pinned runpodctl v2.3.0."""

    if seconds <= 0:
        raise ValueError("termination interval must be positive")
    base = now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    target = (base.astimezone(timezone.utc) + timedelta(seconds=seconds)).replace(
        microsecond=0
    )
    return target.isoformat().replace("+00:00", "Z")


GPU_PLACEMENT_ROUNDS = 2
GPU_PLACEMENT_BACKOFF_SECONDS = 5.0


def _gpu_candidates(spec: JobSpec) -> tuple[str, ...]:
    if spec.pod.compute_type.upper() != "GPU":
        return ()
    return spec.gpu_id_candidates or (spec.pod.gpu_id or "",)


def run_job(
    spec: JobSpec,
    *,
    ctl: Runpodctl | None = None,
    transport: SSHTransport | None = None,
    wait_fn: Callable[..., Endpoint] = wait_for_ssh,
) -> JobResult:
    """Run a job and make Pod deletion mandatory on every post-create path."""

    spec.validate()
    deadline = Deadline(spec.hard_timeout_seconds)
    ctl = ctl or Runpodctl()
    transport = transport or SSHTransport(spec.ssh_private_key)
    pod_id: str | None = None
    endpoint: Endpoint | None = None
    selected_gpu_id: str | None = None
    repo_uploaded = False
    primary_error: Exception | None = None
    artifact_error: Exception | None = None

    try:
        ctl.check_auth(timeout=deadline.timeout(30))
        self_terminate_seconds = (
            spec.hard_timeout_seconds + spec.terminate_grace_seconds
        )
        terminate_after = _rfc3339_after(self_terminate_seconds)
        if spec.pod.compute_type.upper() == "CPU":
            print(
                "CPU placement attempt 1/1: requesting Pod "
                f"{spec.pod.name!r}; terminate-after={terminate_after}",
                flush=True,
            )
            pod_id = ctl.create_pod(
                spec.pod,
                terminate_after=terminate_after,
                self_terminate_seconds=self_terminate_seconds,
                timeout=deadline.timeout(120),
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
            verify_pod_assignment(ctl, pod_id, spec.pod, deadline)
        else:
            candidates = _gpu_candidates(spec)
            placement_failures: list[str] = []
            for round_number in range(1, GPU_PLACEMENT_ROUNDS + 1):
                if round_number > 1:
                    backoff = deadline.timeout(GPU_PLACEMENT_BACKOFF_SECONDS)
                    print(
                        "All GPU candidates rejected in placement round "
                        f"{round_number - 1}; retrying once after {backoff:g}s",
                        flush=True,
                    )
                    time.sleep(backoff)
                round_only_capacity_failures = True
                for candidate_number, candidate in enumerate(candidates, start=1):
                    candidate_request = replace(spec.pod, gpu_id=candidate)
                    pod_id = None
                    endpoint = None
                    print(
                        f"Trying GPU candidate {candidate_number}/{len(candidates)} "
                        f"(round {round_number}/{GPU_PLACEMENT_ROUNDS}): "
                        f"requesting exact gpuId {candidate!r}; "
                        f"terminate-after={terminate_after}",
                        flush=True,
                    )
                    try:
                        pod_id = ctl.create_pod(
                            candidate_request,
                            terminate_after=terminate_after,
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
                        f"Created GPU Pod {pod_id} for candidate {candidate!r}; "
                        f"terminate-after={terminate_after}",
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
                        verify_pod_assignment(
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
                    "GPU placement exhausted two ordered rounds before upload"
                    + (f": {detail}" if detail else "")
                )

        assert pod_id is not None and endpoint is not None
        transport.upload_repo(
            endpoint,
            spec.repo_root,
            spec.remote_dir,
            deadline,
        )
        repo_uploaded = True
        for command in spec.commands:
            remote_command = command
            if selected_gpu_id is not None:
                remote_command = (
                    "export RUNPOD_SELECTED_GPU_ID="
                    f"{shlex.quote(selected_gpu_id)}; {command}"
                )
            transport.run_script(
                endpoint,
                spec.remote_dir,
                remote_command,
                deadline,
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
                transport.download_artifacts(
                    endpoint,
                    spec.remote_dir,
                    spec.artifact_paths,
                    spec.artifact_output,
                    deadline,
                )
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
