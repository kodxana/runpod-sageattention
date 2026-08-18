# Contributing

Changes to this repository can produce native code that imports successfully
but fails only when a particular GPU kernel launches. Treat build-matrix,
toolchain, patch, and release-workflow changes as high risk.

## Before opening a pull request

Run the repository tests and syntax checks:

```bash
python3.12 -m unittest discover -s tests -v
python3.12 -m compileall scripts tools tests
bash -n scripts/*.sh docker/*.sh docker/resource-shim/*.sh
```

If Docker is available, also resolve both Bake targets. A full wheel build is
not required for documentation-only changes.

## Updating SageAttention

1. Pin both the upstream tag and its full commit in `matrix.json`.
2. Read the new setup/build logic and CUDA dispatcher before changing patches.
3. Audit every CUDA translation unit for architecture-specific instructions.
4. Regenerate patches against a pristine checkout and run `git apply --check`.
5. Build from an empty source/build cache for both CUDA variants.
6. Inspect the embedded cubins in every extension.
7. Run the complete representative-GPU matrix.
8. Record behavioral or compatibility changes in `CHANGELOG.md`.

Do not assume an old per-extension architecture matrix remains correct for a
new SageAttention release.

## Updating PyTorch or CUDA

PyTorch C++ extensions use non-stable LibTorch APIs. Any PyTorch or CUDA pin
change requires a rebuild, even if the SageAttention source is unchanged.

Update the matrix and builder together, then validate the resulting wheel in
the corresponding ComfyUI runtime image. Never relabel an existing wheel for a
new stack.

## Runpod costs

Pull-request CI must remain free of automatic paid Pod creation. Build and GPU
workflows are explicit dispatches. Verify the auto-termination deadline and
the always-run cleanup path when changing lifecycle code.

