#!/usr/bin/env python3
"""Select, verify, and optionally install one exact SageAttention wheel asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SelectionError(RuntimeError):
    """No unique, exact, verifiable wheel can be selected."""


@dataclass(frozen=True)
class Environment:
    python_tag: str
    abi_tag: str
    platform_tag: str
    torch_version: str
    torch_cuda_version: str
    gpu_capabilities: tuple[str, ...] = ()


def detect_environment() -> Environment:
    if sys.implementation.name != "cpython":
        raise SelectionError(f"Only CPython is supported, found {sys.implementation.name}")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SelectionError(
            f"Only Linux x86_64 is supported, found {platform.system()} {platform.machine()}")
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on host environment
        raise SelectionError(f"Cannot detect installed torch: {error}") from error
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        raise SelectionError("Installed torch is not a CUDA build")
    gpu_capabilities = ()
    if torch.cuda.is_available():
        gpu_capabilities = tuple(sorted({
            f"{major}.{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        }))
    return Environment(
        python_tag=python_tag,
        abi_tag=python_tag,
        platform_tag=platform_tag,
        torch_version=str(torch.__version__),
        torch_cuda_version=str(cuda_version),
        gpu_capabilities=gpu_capabilities,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise SelectionError("Unsupported manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SelectionError("Manifest contains no artifacts")
    return manifest


def select_artifact(manifest: dict[str, Any], environment: Environment) -> dict[str, Any]:
    fields = {
        "python_tag": environment.python_tag,
        "abi_tag": environment.abi_tag,
        "platform_tag": environment.platform_tag,
        "torch_version": environment.torch_version,
        "torch_cuda_version": environment.torch_cuda_version,
        "cuda_version": environment.torch_cuda_version,
    }
    matches = [
        artifact
        for artifact in manifest["artifacts"]
        if all(artifact.get(field) == expected for field, expected in fields.items())
    ]
    if len(matches) != 1:
        candidates = [
            {
                "asset": artifact.get("asset"),
                **{field: artifact.get(field) for field in fields},
            }
            for artifact in manifest["artifacts"]
        ]
        raise SelectionError(
            "Exact environment match must resolve to one wheel; "
            f"found {len(matches)} for {fields}. Candidates: {candidates}")

    selected = matches[0]
    supported_capabilities = selected.get("supported_compute_capabilities", [])
    unsupported_capabilities = sorted(
        set(environment.gpu_capabilities) - set(supported_capabilities))
    if unsupported_capabilities:
        raise SelectionError(
            "Selected wheel does not support visible GPU capabilities: "
            f"{unsupported_capabilities}")
    digest = selected.get("sha256", "")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise SelectionError("Selected artifact has no valid SHA-256")
    asset = selected.get("asset")
    if (
        not isinstance(asset, str)
        or "/" in asset
        or "\\" in asset
        or PurePosixPath(asset).name != asset
        or not asset.endswith(".whl")
    ):
        raise SelectionError(f"Unsafe or invalid wheel asset name: {asset!r}")
    return selected


def _reject_simple_index(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    path_parts = {part.lower() for part in parsed.path.split("/") if part}
    if "simple" in path_parts:
        raise SelectionError(
            "A package simple-index URL is forbidden; provide a direct release asset base URL")


def resolve_asset(
    artifact: dict[str, Any],
    *,
    manifest_path: Path,
    base_url: str | None,
) -> str | Path:
    asset = artifact["asset"]
    explicit_url = artifact.get("url")
    if explicit_url is not None:
        if not isinstance(explicit_url, str):
            raise SelectionError("Artifact URL must be a string")
        _reject_simple_index(explicit_url)
        parsed = urllib.parse.urlparse(explicit_url)
        if parsed.scheme != "https":
            raise SelectionError("Remote wheel URLs must use HTTPS")
        if Path(urllib.parse.unquote(parsed.path)).name != asset:
            raise SelectionError("Artifact URL basename does not match its signed asset name")
        return explicit_url

    local_asset = manifest_path.parent / asset
    if local_asset.is_file():
        return local_asset
    if not base_url:
        raise SelectionError("Artifact is not local and no direct release asset base URL was supplied")

    _reject_simple_index(base_url)
    parsed_base = urllib.parse.urlparse(base_url)
    if parsed_base.scheme != "https":
        raise SelectionError("Remote release base URL must use HTTPS")
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(asset)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_asset(source: str | Path, artifact: dict[str, Any], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / artifact["asset"]
    temporary = destination / f".{artifact['asset']}.part"
    if isinstance(source, Path) and source.resolve() == final_path.resolve():
        actual_digest = sha256_file(source)
        if actual_digest != artifact["sha256"]:
            raise SelectionError(
                f"SHA-256 mismatch for {artifact['asset']}: "
                f"expected {artifact['sha256']}, got {actual_digest}")
        return source
    if temporary.exists():
        temporary.unlink()

    if isinstance(source, Path):
        shutil.copyfile(source, temporary)
    else:
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "sageattention-wheel-selector/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise SelectionError("Wheel download redirected away from HTTPS")
            shutil.copyfileobj(response, output)

    actual_digest = sha256_file(temporary)
    if actual_digest != artifact["sha256"]:
        temporary.unlink(missing_ok=True)
        raise SelectionError(
            f"SHA-256 mismatch for {artifact['asset']}: "
            f"expected {artifact['sha256']}, got {actual_digest}")
    os.replace(temporary, final_path)
    return final_path


def install_wheel(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        help="HTTPS directory containing direct wheel assets; simple-index URLs are refused",
    )
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print selected manifest entry as JSON")
    parser.add_argument("--python-tag", help=argparse.SUPPRESS)
    parser.add_argument("--abi-tag", help=argparse.SUPPRESS)
    parser.add_argument("--platform-tag", help=argparse.SUPPRESS)
    parser.add_argument("--torch-version", help=argparse.SUPPRESS)
    parser.add_argument("--torch-cuda-version", help=argparse.SUPPRESS)
    return parser.parse_args()


def environment_from_args(args: argparse.Namespace) -> Environment:
    overrides = [
        args.python_tag,
        args.abi_tag,
        args.platform_tag,
        args.torch_version,
        args.torch_cuda_version,
    ]
    if not any(overrides):
        return detect_environment()
    if not all(overrides):
        raise SelectionError("Environment overrides are test-only and must all be supplied together")
    return Environment(*overrides)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    environment = environment_from_args(args)
    artifact = select_artifact(load_manifest(manifest_path), environment)
    source = resolve_asset(artifact, manifest_path=manifest_path, base_url=args.base_url)

    if args.json:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    elif not args.download_dir and not args.install:
        print(source)

    if args.download_dir:
        wheel_path = materialize_asset(source, artifact, args.download_dir.resolve())
        print(wheel_path)
        if args.install:
            install_wheel(wheel_path)
    elif args.install:
        with tempfile.TemporaryDirectory(prefix="sageattention-install-") as temp_dir:
            wheel_path = materialize_asset(source, artifact, Path(temp_dir))
            install_wheel(wheel_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelectionError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"wheel selection failed: {error}") from error
