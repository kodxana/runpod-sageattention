#!/usr/bin/env python3
"""Create a deterministic compatibility manifest for one built wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any


ARCH_RE = re.compile(r"(?:^|[^a-z0-9])(sm_[0-9]+a?)(?:[^a-z0-9]|$)", re.IGNORECASE)
PTX_RE = re.compile(r"(?:compute_[0-9]+a?|\.ptx(?:\b|$))", re.IGNORECASE)
EXTENSION_BASENAMES = {
    "_qattn_sm80": "sageattention._qattn_sm80",
    "_qattn_sm89": "sageattention._qattn_sm89",
    "_qattn_sm90": "sageattention._qattn_sm90",
    "_fused": "sageattention._fused",
}


class InspectionError(RuntimeError):
    """The wheel cannot be inspected conclusively."""


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    if matrix.get("schema_version") != 1:
        raise InspectionError(f"Unsupported matrix schema in {path}")
    return matrix


def find_build(matrix: dict[str, Any], build_id: str) -> dict[str, Any]:
    matches = [entry for entry in matrix.get("builds", []) if entry.get("id") == build_id]
    if len(matches) != 1:
        raise InspectionError(
            f"Expected exactly one matrix build named {build_id!r}, found {len(matches)}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_cuobjdump(cuobjdump: str, option: str, binary: Path) -> str:
    completed = subprocess.run(
        [cuobjdump, option, str(binary)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise InspectionError(
            f"{cuobjdump} {option} failed for {binary.name}:\n{completed.stdout}")
    return completed.stdout


def normalize_cubin_architecture(value: str) -> str:
    # `compute_90a,code=sm_90a` is an architecture-specific compile target,
    # but cuobjdump identifies the resulting ELF image as sm_90.
    normalized = value.lower()
    return "sm_90" if normalized == "sm_90a" else normalized


def inspect_device_code(binary: Path, cuobjdump: str) -> tuple[list[str], bool]:
    elf_listing = _run_cuobjdump(cuobjdump, "--list-elf", binary)
    architectures = sorted(
        {normalize_cubin_architecture(match) for match in ARCH_RE.findall(elf_listing)},
        key=lambda value: (int(re.search(r"[0-9]+", value).group()), value),
    )
    if not architectures:
        # CUDA releases have varied the terse list format. The verbose output
        # consistently includes an `arch = sm_*` field.
        elf_listing = _run_cuobjdump(cuobjdump, "--dump-elf", binary)
        architectures = sorted(
            {normalize_cubin_architecture(match) for match in ARCH_RE.findall(elf_listing)},
            key=lambda value: (int(re.search(r"[0-9]+", value).group()), value),
        )
    if not architectures:
        raise InspectionError(f"No embedded cubin architecture found in {binary.name}")

    ptx_listing = _run_cuobjdump(cuobjdump, "--list-ptx", binary)
    has_ptx = PTX_RE.search(ptx_listing) is not None
    return architectures, has_ptx


def read_wheel(
    wheel: Path,
    *,
    cuobjdump: str | None = None,
    inspect_binaries: bool = True,
) -> dict[str, Any]:
    if not wheel.is_file():
        raise InspectionError(f"Wheel does not exist: {wheel}")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise InspectionError("Wheel must contain exactly one METADATA and one WHEEL file")

        metadata = BytesParser(policy=compat32).parsebytes(archive.read(metadata_names[0]))
        wheel_metadata = BytesParser(policy=compat32).parsebytes(archive.read(wheel_names[0]))
        extension_members: dict[str, str] = {}
        for member in names:
            if not member.endswith(".so"):
                continue
            stem = Path(member).name.split(".", 1)[0]
            extension_name = EXTENSION_BASENAMES.get(stem)
            if extension_name is not None:
                if extension_name in extension_members:
                    raise InspectionError(f"Duplicate extension {extension_name} in wheel")
                extension_members[extension_name] = member

        architectures: dict[str, list[str]] = {}
        ptx_extensions: list[str] = []
        if inspect_binaries:
            resolved_cuobjdump = cuobjdump or shutil.which("cuobjdump")
            if not resolved_cuobjdump:
                raise InspectionError("cuobjdump is required for native-code inspection")
            with tempfile.TemporaryDirectory(prefix="sageattention-wheel-") as temp_dir:
                temp_root = Path(temp_dir)
                for extension_name, member in sorted(extension_members.items()):
                    extracted = temp_root / Path(member).name
                    extracted.write_bytes(archive.read(member))
                    cubins, has_ptx = inspect_device_code(extracted, resolved_cuobjdump)
                    architectures[extension_name] = cubins
                    if has_ptx:
                        ptx_extensions.append(extension_name)

    return {
        "distribution": metadata.get("Name"),
        "version": metadata.get("Version"),
        "requires_dist": metadata.get_all("Requires-Dist") or [],
        "requires_python": metadata.get("Requires-Python"),
        "wheel_tags": wheel_metadata.get_all("Tag") or [],
        "extension_members": extension_members,
        "extension_architectures": architectures,
        "ptx_extensions": ptx_extensions,
    }


def build_manifest(
    matrix: dict[str, Any],
    build_id: str,
    wheel: Path,
    wheel_info: dict[str, Any],
    build_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = find_build(matrix, build_id)
    package = matrix["package"]
    platform = matrix["platform"]
    policy = matrix["cuda_policy"]
    artifact = {
        "abi_tag": platform["abi_tag"],
        "asset": wheel.name,
        "build_id": build_id,
        "cuda_version": build["cuda_version"],
        "distribution": wheel_info["distribution"],
        "extension_architectures": wheel_info["extension_architectures"],
        "extension_compile_targets": policy["extension_compile_targets"],
        "native_sass_only": not bool(wheel_info["ptx_extensions"]),
        "platform_tag": platform["platform_tag"],
        "python_tag": platform["python_tag"],
        "requires_dist": sorted(wheel_info["requires_dist"]),
        "sha256": sha256_file(wheel),
        "size": wheel.stat().st_size,
        "source_commit": package["source_commit"],
        "source_url": package["source_url"],
        "supported_compute_capabilities": policy["compute_capabilities"],
        "torch_cuda_version": build["torch_cuda_version"],
        "torch_version": build["torch_version"],
        "version": wheel_info["version"],
        "wheel_tags": sorted(wheel_info["wheel_tags"]),
    }
    if build_evidence is not None:
        artifact["build_evidence"] = build_evidence
    return {"artifacts": [artifact], "schema_version": 1}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_checksum(path: Path, wheel: Path, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{digest}  {wheel.name}\n", encoding="ascii", newline="\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--cuobjdump", default=None)
    parser.add_argument("--evidence", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.matrix.resolve())
    wheel = args.wheel.resolve()
    wheel_info = read_wheel(wheel, cuobjdump=args.cuobjdump)
    evidence = None
    if args.evidence:
        with args.evidence.resolve().open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
    manifest = build_manifest(matrix, args.build_id, wheel, wheel_info, evidence)
    atomic_json(args.manifest.resolve(), manifest)
    artifact = manifest["artifacts"][0]
    atomic_checksum(args.checksums.resolve(), wheel, artifact["sha256"])
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InspectionError as error:
        raise SystemExit(f"wheel inspection failed: {error}") from error
