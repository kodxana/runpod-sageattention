# SageAttention wheels for Runpod ComfyUI

This repository builds reproducible Linux x86-64 SageAttention wheels for the
exact Python, PyTorch, and CUDA stacks used by the Runpod ComfyUI base images.
Compilation runs on CPU Pods; representative NVIDIA GPU Pods validate the
finished wheels before a release can be published.

The factory is intentionally strict. A wheel is selected by the complete
runtime tuple—not merely by its Python wheel tag:

```text
SageAttention source + Python ABI + PyTorch version + CUDA variant + GPU cubins
```

## Initial compatibility matrix

| Variant | Python | PyTorch | Toolkit | Native GPU targets |
|---|---|---|---|---|
| `cu128` | CPython 3.12 | `2.10.0+cu128` | CUDA 12.8 | SM 80, 86, 89, 90a, 120 |
| `cu130` | CPython 3.12 | `2.10.0+cu130` | CUDA 13.0 | SM 80, 86, 89, 90a, 120 |

SageAttention 2.2.0 does not implement SM 100 dispatch. B100/B200 support is
therefore deliberately excluded rather than advertised based on an untested
PTX fallback.

## Pipeline

```mermaid
flowchart LR
    M["Pinned matrix.json"] --> I["Dedicated builder images"]
    I --> C12["Runpod CPU Pod: cu128"]
    I --> C13["Runpod CPU Pod: cu130"]
    C12 --> W["Wheels + manifests + checksums"]
    C13 --> W
    W --> G["Representative Runpod GPU Pods"]
    G --> V["Import, cubin, and numerical validation"]
    V --> R["Immutable GitHub Release"]
```

GitHub Actions coordinates the work and retains the evidence. It does not run
NVCC and it does not promote an artifact before GPU validation succeeds.

## Why a downstream patch is necessary

Upstream SageAttention 2.2.0 applies one global CUDA architecture list to all
extensions. Its Hopper TMA/WGMMA source then gets compiled for older GPU
targets, which fails in `ptxas`. This repository gives each extension its own
safe native-SASS target list:

| Extension | Targets |
|---|---|
| `_qattn_sm80` | 80, 86, 89, 90a, 120 |
| `_qattn_sm89` | 89, 90a, 120 |
| `_qattn_sm90` | 90a only |
| `_fused` | 80, 86, 89, 90a, 120 |

## Build resources

A GPU is not required and does not accelerate the compilation. The validated
CUDA 12.8 build took 11 minutes 29 seconds with two concurrent extensions and
two Ninja jobs per extension, peaking at 29.65 GB of cgroup-accounted memory.

Recommended CPU Pod:

- 16 vCPUs;
- 64 GB RAM recommended, 32 GB minimum, for the reviewed
  `MAX_JOBS=2` × `EXT_PARALLEL=1` default;
- 80 GB container storage (the automated factory minimum and default); the
  build preflight requires at least 20 GiB free on both the work and output
  filesystems.

Smaller Pods can use serialized compilation. The build entrypoint obtains its
limits from cgroups and lowers concurrency rather than trusting host-wide
`/proc/meminfo` or CPU counts.

## Safety properties

- Source tags and commits are both pinned and verified.
- CUDA 12.8 and CUDA 13.0 produce distinct downstream versions and manifests.
- The exact PyTorch dependency is present in wheel metadata.
- The installer refuses a CUDA/PyTorch mismatch or an ambiguous wheelhouse.
- CUDA driver stubs are link-time-only builder inputs.
- GPU jobs receive an RFC3339 platform termination deadline. CPU builders arm
  an in-Pod self-delete watchdog, and every job also has `finally` cleanup.
- Paid Pod workflows require an explicit dispatch/release gate.
- Release creation depends on all configured GPU tests passing.
- No global `LD_PRELOAD` is used by the resource-reporting shim.

## Repository map

- `matrix.json` — authoritative build and test matrix.
- `docker/` — dedicated CPU builder images and scoped resource shim.
- `patches/` — reviewed downstream SageAttention build patches.
- `scripts/` — build, inspect, manifest, selection, and GPU validation tools.
- `tools/` — Runpod lifecycle and cgroup resource helpers.
- `.github/workflows/` — local CI, explicit builds, and gated releases.
- `docs/` — architecture, matrix, Runpod, and operations details.

## Local validation

The repository's non-GPU checks use only Python's standard library:

```bash
python3.12 -m unittest discover -s tests -v
python3.12 -m compileall scripts tools tests
```

Building a wheel requires one of the pinned builder images. Launching Pods and
publishing releases additionally requires the GitHub secrets documented in
`docs/runpod-orchestration.md`.

## Distribution rule

Do not combine all local CUDA/PyTorch variants into one unconstrained simple
package index. Standard wheel tags do not describe CUDA, PyTorch, or compute
capability. Consumers should use the selector with a release manifest or an
explicit variant-specific asset URL.

## Status

The CUDA 12.8 fat-wheel patch and RTX 4090 runtime path have been proven on a
live Runpod Pod; the measurements and limitations are recorded in
[`docs/cuda128-proof.md`](docs/cuda128-proof.md). Before the first public
release, the CUDA 13.0 variant and the remaining representative GPU families
must pass the same gated workflow.
