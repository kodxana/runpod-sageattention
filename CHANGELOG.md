# Changelog

## Unreleased

- Add the initial standalone SageAttention wheel factory.
- Pin ComfyUI-compatible CUDA 12.8 and CUDA 13.0 PyTorch stacks.
- Add safe per-extension CUDA architecture compilation for SageAttention 2.2.0.
- Add sequential GPU-backed Runpod builds with a sized CPU fallback,
  representative-GPU validation, and gated releases.
- Add cgroup-aware resource discovery for build throttling and terminal tools.
- Restore the pinned CUDA and Python builder paths for OpenSSH sessions, add
  GPU-free image/toolchain validation, and make GPU fallback logs deterministic.
