# Architecture and release contract

## Separation of responsibilities

The system has three independent trust boundaries:

1. **GitHub orchestration** resolves a pinned matrix, starts ephemeral Pods,
   moves source and artifacts, records evidence, and performs release gating.
2. **CPU builder Pods** contain compilers, CUDA development libraries, and the
   exact PyTorch stack. They produce wheels but never decide whether a wheel is
   releasable.
3. **GPU validation Pods** use the actual ComfyUI-compatible runtime image.
   They install one exact wheel, import every compiled module, execute the
   architecture-selected kernel, and compare it with PyTorch SDPA.

The builder and runtime images are deliberately different. The builder should
not carry ComfyUI, Jupyter, or model assets; the validation Pod should resemble
the consumer environment rather than the compiler environment.

## Artifact identity

Every build result consists of:

- exactly one wheel;
- a SHA-256 checksum;
- a machine-readable manifest containing all source/toolchain pins;
- the applied patch checksums;
- the requested and observed cubin matrix;
- build duration, effective CPU count, memory limit, and peak memory;
- the builder image reference and, in release jobs, immutable digest;
- build and validation logs.

The release key is the tuple:

```text
(sage commit, downstream patch set, cp312 ABI, torch version, CUDA toolkit)
```

GPU targets are recorded inside the manifest and verified from the compiled
objects. They are not represented as separate Docker platforms.

## Builder image policy

Both builders use Ubuntu 24.04 and Python 3.12 to match `comfyui-base`. Each
image contains only the matching CUDA toolkit, CUDA development libraries,
PyTorch distribution, compiler toolchain, build frontend, SSH service, and
resource helper.

The builder image may link `_qattn_sm90` against `libcuda.so` from the toolkit
stub directory when no NVIDIA driver is present. The stub directory is scoped
to the link command and is never copied to a runtime image or placed on a
consumer's library path.

## Architecture policy

Published wheels use exact native SASS and no PTX fallback for the initial
matrix. This makes the supported set explicit and prevents an architecture-
specific Hopper program from being treated as forward-compatible code.

SageAttention's extensions do not all support the same targets. The patch must
continue to generate independent `-gencode` lists. Any SageAttention upgrade is
treated as a new source audit; carrying the patch forward mechanically is not
enough.

## Test policy

Binary inspection verifies every compiled extension's cubins even when the
current Pod cannot execute those cubins. Runtime validation then uses one
representative GPU per supported compute-capability family, rather than paying
to test every product SKU with the same capability.

Routine releases require the representatives configured by the matrix. A
separate manual compatibility audit may cover the wider Runpod catalog.

Numerical tests require finite outputs, expected shapes and dtypes, a minimum
cosine similarity, and a bounded relative L2 error against PyTorch SDPA for
both causal and non-causal inputs. Thresholds live in the matrix and may not be
weakened automatically after a failure.

## Promotion policy

Build artifacts are uploaded under immutable workflow-run identities first.
The release job downloads those exact artifacts only after validation jobs have
passed. It verifies checksums again, creates an immutable GitHub Release, and
attaches wheels, manifests, checksums, and concise test evidence.

No mutable `latest`, stable index, or release asset is updated during the build
stage. A failed or skipped required GPU test blocks promotion.
