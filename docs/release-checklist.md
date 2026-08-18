# Release checklist

## Source and matrix

- [ ] SageAttention tag resolves to the pinned full commit.
- [ ] Python, PyTorch, CUDA toolkit, and ComfyUI runtime image pins are exact.
- [ ] Builder image references resolve to recorded immutable digests.
- [ ] Patch files apply cleanly to a pristine checkout.
- [ ] Patch SHA-256 values are recorded in each build manifest.

## Wheel builds

- [ ] The `cu128` build records the verified Runpod CPU/RAM assignment, matching
      root-owned receipt, selected capacity source, and bounded cgroup peak.
- [ ] The `cu130` build starts only after the `cu128` builder has finished.
- [ ] GPU-backed builds record the exact requested build GPU and use an absolute
      platform termination deadline; CPU fallback builds arm the in-Pod watchdog.
- [ ] Resource preflight selects safe parallelism or refuses an undersized Pod.
- [ ] Build evidence records an empty `CUDA_VISIBLE_DEVICES` value.
- [ ] Each variant produces exactly one expected CPython 3.12 Linux x86-64 wheel.
- [ ] Wheel metadata contains the downstream local version and exact PyTorch pin.
- [ ] Wheel and build manifest checksums verify after GitHub artifact transfer.

## Binary inspection

- [ ] `_qattn_sm80` contains SM 80, 86, 89, 90, and 120 cubins.
- [ ] `_qattn_sm89` contains SM 89, 90, and 120 cubins only.
- [ ] `_qattn_sm90` contains an SM 90 cubin only.
- [ ] `_fused` contains SM 80, 86, 89, 90, and 120 cubins.
- [ ] No PTX fallback is present in the initial release matrix.
- [ ] CUDA driver stubs are absent from the wheel.

## GPU validation

- [ ] Every required representative GPU test ran; none was silently skipped.
- [ ] The wheel was installed into the matching ComfyUI-compatible runtime.
- [ ] All compiled extension modules imported.
- [ ] Causal and non-causal kernel launches returned finite values.
- [ ] Raw outputs matched the canonical shape and dtype before FP32 conversion.
- [ ] Reports record matching actual/expected shape and dtype for every launch.
- [ ] Cosine-similarity and relative-L2 checks passed.
- [ ] CUDA 13 tests ran on a host driver compatible with CUDA 13.

## Promotion

- [ ] Release inputs identify one immutable GitHub workflow run.
- [ ] Checksums were reverified after downloading build/test artifacts.
- [ ] Both CUDA variants have separate, unambiguous assets and manifests.
- [ ] Release notes state supported Python, PyTorch, CUDA, and SM targets.
- [ ] No shared unconstrained PEP 503 page can select the wrong variant.
- [ ] All Runpod Pods from the workflow are terminated.
