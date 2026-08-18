# Changelog

## Unreleased

- Backport upstream SageAttention commit
  `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4` so the SM90 fake implementation
  no longer overwrites the real eager custom-op binding, and version corrected
  wheels with the downstream `sm90fix1` local suffix.
- Recognize Runpod's explicit per-machine resource placement failure as a
  capacity miss and continue through the reviewed GPU candidate list.
- Preserve per-kernel numerical and launch diagnostics for failed GPU tests,
  and require schema-complete, contradiction-free reports at promotion.
- Add the initial standalone SageAttention wheel factory.
- Pin ComfyUI-compatible CUDA 12.8 and CUDA 13.0 PyTorch stacks.
- Add safe per-extension CUDA architecture compilation for SageAttention 2.2.0.
- Add sequential GPU-backed Runpod builds with a sized CPU fallback,
  representative-GPU validation, and gated releases.
- Add cgroup-aware resource discovery for build throttling and terminal tools.
- Bind release-build memory capacity to the exact verified Runpod assignment
  receipt, retain smaller cgroup limits, and fail closed on unsafe peak evidence.
- Permit genuinely missing cgroup counters only through a serialized,
  64-GiB-or-larger process-group RSS evidence mode with a dedicated build
  supervisor, bounded signal cleanup, and explicit accounting limitations.
- Restore the pinned CUDA and Python builder paths for OpenSSH sessions, add
  GPU-free image/toolchain validation, and make GPU fallback logs deterministic.
