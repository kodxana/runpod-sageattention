#!/usr/bin/env python3
"""Merge per-build manifests deterministically after revalidating local wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class MergeError(RuntimeError):
    """Input manifests are unsafe, inconsistent, or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_and_verify(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise MergeError(f"Unsupported schema in {path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise MergeError(f"No artifacts in {path}")

    expected_checksum_lines = []
    for artifact in artifacts:
        asset = artifact.get("asset")
        if (
            not isinstance(asset, str)
            or "/" in asset
            or "\\" in asset
            or Path(asset).name != asset
            or not asset.endswith(".whl")
        ):
            raise MergeError(f"Unsafe asset name in {path}: {asset!r}")
        wheel = path.parent / asset
        if not wheel.is_file():
            raise MergeError(f"Manifest asset is not beside {path}: {asset}")
        digest = sha256_file(wheel)
        if artifact.get("sha256") != digest:
            raise MergeError(f"SHA-256 mismatch for {wheel}")
        if artifact.get("size") != wheel.stat().st_size:
            raise MergeError(f"Size mismatch for {wheel}")
        expected_checksum_lines.append(f"{digest}  {asset}")

    sibling_checksums = path.parent / "SHA256SUMS"
    if sibling_checksums.is_file():
        actual_lines = sibling_checksums.read_text(encoding="ascii").splitlines()
        if actual_lines != sorted(expected_checksum_lines):
            raise MergeError(f"Checksum file does not match {path}: {sibling_checksums}")
    return artifacts


def runtime_tuple(artifact: dict[str, Any]) -> tuple[str, ...]:
    keys = (
        "python_tag",
        "abi_tag",
        "platform_tag",
        "torch_version",
        "torch_cuda_version",
    )
    values = tuple(artifact.get(key) for key in keys)
    if not all(isinstance(value, str) and value for value in values):
        raise MergeError(f"Artifact has an incomplete compatibility tuple: {artifact.get('asset')}")
    return values


def merge(paths: list[Path]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    assets: set[str] = set()
    runtime_tuples: set[tuple[str, ...]] = set()
    build_ids: set[str] = set()
    for path in paths:
        for artifact in load_and_verify(path):
            asset = artifact["asset"]
            compatibility = runtime_tuple(artifact)
            build_id = artifact.get("build_id")
            if asset in assets:
                raise MergeError(f"Duplicate asset: {asset}")
            if compatibility in runtime_tuples:
                raise MergeError(f"Ambiguous duplicate runtime tuple: {compatibility}")
            if not isinstance(build_id, str) or not build_id or build_id in build_ids:
                raise MergeError(f"Missing or duplicate build ID: {build_id!r}")
            assets.add(asset)
            runtime_tuples.add(compatibility)
            build_ids.add(build_id)
            artifacts.append(artifact)

    artifacts.sort(key=lambda artifact: (*runtime_tuple(artifact), artifact["asset"]))
    return {"artifacts": artifacts, "schema_version": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [path.resolve() for path in args.manifests]
    merged = merge(paths)
    payload = json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    checksum_payload = "".join(
        f"{artifact['sha256']}  {artifact['asset']}\n"
        for artifact in merged["artifacts"]
    )
    atomic_write(args.output.resolve(), payload)
    atomic_write(args.checksums.resolve(), checksum_payload)
    print(f"merged {len(merged['artifacts'])} wheel manifests")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MergeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"manifest merge failed: {error}") from error
