# CUDA 12.8 fat-wheel proof

This records the disposable-Pod experiment that established the initial
builder settings and downstream architecture patch. The prototype wheel itself
is deliberately not committed to this repository.

## Environment

- SageAttention: 2.2.0 at
  `eb615cf6cf4d221338033340ee2de1c37fbdba4a`
- Ubuntu: 24.04
- CPython: 3.12.3
- PyTorch: `2.10.0+cu128`
- CUDA toolkit/NVCC: 12.8.93
- Validation GPU: NVIDIA GeForce RTX 4090, SM 89

The Pod exposed about 62 GB through cgroup v1 and a 54.4-core CPU quota while
host-scoped `/proc` files misleadingly showed 251 GiB and 256 CPUs.

## Upstream failure

The clean upstream build was requested with:

```text
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;12.0
EXT_PARALLEL=2
MAX_JOBS=2
```

It failed after 381 seconds and peaked at roughly 37.3 GB. The shared global
`-gencode` list caused Hopper WGMMA/TMA instructions to be assembled for older
targets. `ptxas` rejected instructions including `mbarrier`, TMA
`cp.async.bulk.tensor`, and `wgmma`.

## Patched build

The per-extension target patch completed a clean build with the same requested
architecture list:

- elapsed: 689 seconds (11 minutes 29 seconds);
- peak cgroup usage: 29,652,967,424 bytes;
- wheel size: 38,182,423 bytes;
- prototype SHA-256:
  `384697a72229a5acf95c7342ca90714ddf82575ca1cb176ead44b05f08c7de29`.

Binary inspection observed:

| Extension | `cuobjdump` cubins |
|---|---|
| `_qattn_sm80` | `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_120` |
| `_qattn_sm89` | `sm_89`, `sm_90`, `sm_120` |
| `_qattn_sm90` | `sm_90` |
| `_fused` | `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_120` |

NVCC uses the architecture-specific `compute_90a`/`sm_90a` target during
compilation; `cuobjdump --list-elf` labels the resulting cubin `sm_90`.

## Runtime result

The wheel was installed with `--no-deps` into a fresh Python 3.12 virtual
environment inheriting the ComfyUI-base PyTorch packages. All four extension
modules imported successfully.

| Input | Cosine similarity vs Torch SDPA | Relative L2 | Finite |
|---|---:|---:|---|
| Non-causal | 0.9993014 | 0.0373795 | Yes |
| Causal | 0.9994161 | 0.0341675 | Yes |

## What this proof does not establish

Static cubin inspection proves that code objects exist for the requested
targets, not that every target executes correctly. Public release still
requires representative SM 80, 86, 89, 90, and 120 runtime tests for each
CUDA/PyTorch variant. CUDA 13.0 also needs a separate build and a host driver
that can load CUDA 13 containers.
