# Changelog

## Unreleased

- Add an hourly reaper (`.github/workflows/reap-pods.yml`,
  `tools/reap_pods.py`) that deletes `sage-build-*` / `sage-test-*` Pods whose
  GitHub run attempt has completed or no longer exists, or is older than 360
  minutes. Validation Pods had no backstop if the runner died: REST v2 has no
  platform termination deadline and the watchdog only runs in builder images.
- Treat only a REST `400` that reports missing capacity as a capacity miss. A
  `403` or any other `400` now fails on the first candidate instead of waiting
  through every placement round, and the auth check proves write access with a
  no-op v2 `stop` action (`POST /pods/{id}/action`) on a Pod id that
  cannot exist; only 401/403 (read-only key), 400, or a 404 naming a missing
  resource are understood, and anything else fails the check.
- Check in CI that builder and runtime image pins in `build.yml` (dispatch and
  call) and `release.yml` match `matrix.json`.
- Validate the cu128 and cu130 wheels on `runpod/comfyui` 1.7.0, the image the
  cu132 wheel already uses, instead of 1.4.6.
- Add a CUDA 13.2 / PyTorch 2.13.0 variant matching the `runpod/comfyui`
  `-cuda13.2` image.
- Publish the builder images as `runpod/sageattention-wheel-builder` and pin
  their rebuilt digests.
- Move all Pod orchestration to the Runpod REST API v2 and drop `runpodctl`
  from the runner and the builder image. REST v1 stops serving on November 15,
  2026 and GraphQL in early 2027; the API key now needs only REST Pods
  read/write. GPU builds pass their vCPU and RAM minimums as placement filters,
  and builder Pods arm the in-Pod self-delete watchdog, which now calls REST v2
  directly, because v2 has no platform-side termination deadline.
- Retry placement when every candidate lacks capacity: up to 5 rounds with a
  300-second wait between them (`--placement-rounds`,
  `--placement-backoff-seconds`), for CPU and GPU Pods. The wait does not count
  against the job's hard timeout; the GitHub job timeouts grow to cover it.
- Send an explicit User-Agent on Runpod REST calls; Cloudflare rejects the
  default `Python-urllib` agent with 403 "error code: 1010".
- Allow targeting a non-production Runpod API through the `RUNPOD_API_URL`
  repository variable (a REST v2 base URL); unset means production.
- Restore the executable bit on repository scripts.
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
