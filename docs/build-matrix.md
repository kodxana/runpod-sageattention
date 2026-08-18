# SageAttention wheel build matrix

This repository produces two deliberately separate SageAttention 2.2.0 wheels for the two PyTorch stacks used by `runpod/comfyui`. They are not interchangeable, even though both wheels contain the same GPU architecture matrix.

| Build ID | Builder toolkit | Exact PyTorch | Wheel version | Matching ComfyUI line |
| --- | --- | --- | --- | --- |
| `cp312-torch2.10.0-cu128` | CUDA 12.8 | `2.10.0+cu128` | `2.2.0+torch2.10.0.cu128` | `runpod/comfyui:cuda12.8` |
| `cp312-torch2.10.0-cu130` | CUDA 13.0 | `2.10.0+cu130` | `2.2.0+torch2.10.0.cu130` | `runpod/comfyui:cuda13.0` |

The table shows the human-readable ComfyUI release lines. `matrix.json` and the
workflow defaults pin the currently reviewed OCI index digests so a mutable tag
cannot change the runtime used by a build or release run.

Both target Ubuntu 24.04, Linux x86-64, and CPython 3.12 (`cp312-cp312-linux_x86_64`). The source is pinned to SageAttention commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`. `matrix.json` is the machine-readable source of truth for these values, output names, resources, scheduler requirements, validation thresholds, and representative GPUs.

## Native-code policy

The wheel contains native SASS only. No PTX fallback is included. The requested NVCC targets and the labels observed by `cuobjdump` differ for Hopper: NVCC receives `compute_90a,code=sm_90a`, while `cuobjdump --list-elf` reports that cubin as `sm_90`. The matrix records both forms separately.

| Extension | NVCC compile targets | Runtime role |
| --- | --- | --- |
| `sageattention._qattn_sm80` | `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_120` | INT8 QK and FP16 PV implementation usable from SM80 upward |
| `sageattention._qattn_sm89` | `sm_89`, `sm_90a`, `sm_120` | FP8 implementation; lower architectures would compile trap fallbacks |
| `sageattention._qattn_sm90` | `sm_90a` only | Hopper TMA/WGMMA implementation; it must not receive any other gencode |
| `sageattention._fused` | `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_120` | Quantization, conversion, and fused helper kernels |

The upstream dispatcher uses the SM80 CUDA path on SM80, Triton on SM86, the SM89 CUDA path on SM89 and SM120, and the dedicated SM90 path on SM90. The wider matrix above intentionally preserves explicitly callable public CUDA APIs as well as the automatic dispatcher.

Runtime validation is prefilled with exact Runpod `gpuId` values: `NVIDIA A100 80GB PCIe` (SM80), `NVIDIA GeForce RTX 3090` (SM86), `NVIDIA GeForce RTX 4090` (SM89), `NVIDIA H100 PCIe` (SM90), and `NVIDIA GeForce RTX 5090` (SM120). B200 (SM100) and B300 (SM103) are excluded because SageAttention 2.2.0 has no SM10x cubins or dispatcher path; an SM120 result is not compatible with either data-center architecture.

## Builder images

Build both images with:

```bash
docker buildx bake -f docker/docker-bake.hcl builder-cu128 builder-cu130
```

The images install Ubuntu 24.04 build tools, Python 3.12 headers, CUDA compiler/runtime development packages, cuBLAS, cuSOLVER, cuSPARSE, `cuobjdump`, PyTorch, and the wheel inspection tools. The CUDA keyring package is verified against its pinned SHA-256 before installation. CUDA driver stubs are available through `LIBRARY_PATH` solely for linking the Hopper extension; they are neither placed on runtime `LD_LIBRARY_PATH` nor packaged in a wheel.

The image is SSH-ready and accepts newline-separated public keys through `SSH_AUTHORIZED_KEYS` (or `PUBLIC_KEY`). Password login is disabled. Its default command is foreground `sshd`; an explicit container command runs normally. Setting `START_SSHD=1` starts SSH alongside that command. Factory builders launch this image on a GPU-backed Pod, although the build itself does not use the accelerator.

Every default GPU-backed builder receives an absolute RFC3339 platform termination deadline and is deleted from a retried `finally` path. The explicit CPU fallback has no verified platform deadline, so it must enable `RUNPOD_SELF_TERMINATE_SECONDS` in addition to the same mandatory cleanup. When enabled, the entrypoint accepts only 600–21600 seconds, requires a valid `RUNPOD_POD_ID`, Pod-scoped `RUNPOD_API_KEY`, and the checksum-pinned `runpodctl` bundled in the image, then arms a detached deletion watchdog. Missing prerequisites fail startup, and credentials are never printed.

`pod-resources` provides the authoritative cgroup-aware CPU and memory view. Scoped wrappers make `free`, `top`, and `htop` report Pod resources without setting `LD_PRELOAD` globally.

## GPU-backed build

The factory schedules compilation on a GPU Pod by default, but `build-wheel.sh` exports `CUDA_VISIBLE_DEVICES=""`; the attached accelerator and its VRAM do not compile the wheel. Host-system resources are authoritative. Build policy requires at least 4 effective vCPUs, a 32 GB system-RAM assignment, and an 80 GB container disk; 16 vCPUs and 64 GB system RAM are recommended. The backend-neutral in-Pod preflight independently requires a finite cgroup limit of at least 32 GiB, enough current headroom for the compiler reserve plus one job, and 20 GiB free on both the work and output filesystems. The reviewed default is two compiler jobs and one extension at a time. The live 2-by-2 proof peaked near 29.65 GiB, so higher parallelism should only be enabled with new peak evidence on a larger assignment.

The matrix field `resources.gpu_required=false` describes the build command's compute behavior, not the default Runpod provisioning backend. It remains false because compilation never launches a GPU kernel.

The cu128 and cu130 builders run sequentially. `NVIDIA A100 80GB PCIe` is the prefilled Runpod build `gpuId`; its architecture and VRAM are irrelevant to compilation because the GPU is hidden. The host driver must still satisfy the matching CUDA scheduler floor so the container can start. Before source upload, orchestration verifies the exact GPU assignment, the matrix floor of 4 effective vCPUs, 32 GB system RAM, and an 80 GB container disk; 16 vCPUs and 64 GB system RAM remain recommended. The in-Pod cgroup preflight then independently verifies usable resources.

Run one matrix entry inside its matching builder:

```bash
bash /workspace/scripts/build-wheel.sh \
  --build-id cp312-torch2.10.0-cu128 \
  --output-dir /workspace/dist/cp312-torch2.10.0-cu128
```

The output directory must initially be empty. It receives exactly:

- the exact wheel named by `matrix.json`;
- `manifest.json` with compatibility, cubin coverage, hashes, tool versions, matrix and patch hashes, image identity, resource snapshots, selected parallelism, elapsed time, and cgroup peak evidence;
- `SHA256SUMS` containing exactly that wheel.

The system-resource preflight refuses an unlimited or undersized container. `ALLOW_LOW_RESOURCES=1` is an explicit escape hatch for diagnostics, not release builds. The build script is the sole authority for its cgroup-aware `MAX_JOBS` default; the SSH entrypoint does not precompute it. Values above the reviewed `MAX_JOBS=2` and `EXT_PARALLEL=1` caps are rejected unless `ALLOW_UNSAFE_PARALLELISM=1` is explicitly set on a larger Pod. Set `BUILDER_IMAGE_REF` and `BUILDER_IMAGE_DIGEST` in release jobs so the immutable builder identity is captured.

## Static and GPU validation

Static validation does not launch the attached GPU and checks the exact filename, local version, `Requires-Dist: torch==...` metadata, Python/ABI/platform tag, four compiled modules, per-extension cubins, absence of PTX, absence of driver stubs, artifact size, manifest, and checksum:

```bash
python3.12 scripts/validate-wheel.py \
  --matrix matrix.json \
  --build-id cp312-torch2.10.0-cu128 \
  --wheel dist/cp312-torch2.10.0-cu128/sageattention-2.2.0+torch2.10.0.cu128-cp312-cp312-linux_x86_64.whl \
  --manifest dist/cp312-torch2.10.0-cu128/manifest.json \
  --checksums dist/cp312-torch2.10.0-cu128/SHA256SUMS
```

GPU validation adds `--runtime`, imports all four compiled modules explicitly, and runs the canonical matrix case in causal and noncausal modes. It launches normal `sageattn` dispatch plus every native public API valid on that family: SM80 on all five families, SM89 FP8 on SM89/SM90/SM120, and the Hopper API on SM90. This catches a native cubin that imports correctly but fails when launched, including the SM86 cubin otherwise bypassed by Triton dispatch. Release thresholds are cosine similarity at least `0.995` and relative L2 at most `0.10` against PyTorch math SDPA. Example:

```bash
RUNTIME_IMAGE_REF='runpod/comfyui@sha256:<digest>' \
python3.12 scripts/validate-wheel.py \
  --matrix matrix.json \
  --build-id cp312-torch2.10.0-cu128 \
  --wheel /workspace/artifacts/<wheel>.whl \
  --manifest /workspace/artifacts/manifest.json \
  --checksums /workspace/artifacts/SHA256SUMS \
  --runtime \
  --expected-capability 8.9 \
  --runtime-report runtime-results/cp312-torch2.10.0-cu128/sm89.json
```

Release validation covers every required matrix representative: SM80, SM86, SM89, SM90, and SM120. Before converting an output to FP32 for numerical comparison, the validator requires it to be a CUDA tensor with the canonical shape and dtype. Every result records `output_shape`, `output_dtype`, `expected_output_shape`, and `expected_output_dtype`; the release gate requires exact equality for all four fields. A report uses status `pass` and binds the wheel asset/hash, expected and actual capability, exact runtime versions, GPU, immutable runtime image, loaded module paths, and the implementation name and metrics for every required causal/noncausal launch.

## Release manifest and safe installation

Each build emits a one-artifact manifest. Merge both only with the checked merger:

```bash
python3.12 scripts/merge-manifests.py \
  --output release/manifest.json \
  --checksums release/SHA256SUMS \
  dist/cp312-torch2.10.0-cu128/manifest.json \
  dist/cp312-torch2.10.0-cu130/manifest.json
```

The merger rehashes adjacent wheels and rejects duplicate assets, build IDs, or runtime tuples. The selector then detects the exact CPython, ABI, platform, PyTorch local version, and PyTorch CUDA version. It refuses zero matches, multiple matches, mismatches, unsafe asset names, bad hashes, insecure URLs, and package simple-index URLs.

```bash
python3.12 scripts/select-wheel.py \
  --manifest release/manifest.json \
  --base-url 'https://github.com/<owner>/<repo>/releases/download/<tag>' \
  --install
```

Installation always downloads one direct asset, verifies its SHA-256, and gives that local file to pip with `--no-deps`. Do not publish the cu128 and cu130 wheels behind one unconstrained simple-index project page: standard pip version ordering does not know which CUDA runtime the Pod needs.

## Portability boundary

These are intentionally ComfyUI-base wheels, not generic manylinux wheels. Building on Ubuntu 24.04 can bind them to that glibc generation. The Python ABI, PyTorch C++ ABI, exact PyTorch local build, CUDA toolkit line, Linux architecture, and embedded GPU code are all part of compatibility. Produce a separate matrix entry and wheel whenever any of those changes.
