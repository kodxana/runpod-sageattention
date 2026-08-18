## Summary

Describe the change and the compatibility tuple it affects.

## Validation

- [ ] Standard-library unit tests pass.
- [ ] Python and shell syntax checks pass.
- [ ] Matrix/schema checks pass.
- [ ] No paid Runpod workflow was triggered unintentionally.

For build, patch, CUDA, PyTorch, or release changes:

- [ ] Clean CPU build completed.
- [ ] Cubin inspection completed.
- [ ] Required representative GPU tests completed.
- [ ] Build/test Pods were confirmed terminated.

## Release safety

- [ ] The change cannot place incompatible CUDA/PyTorch wheels in one
      ambiguous installer path.
- [ ] Release promotion still depends on all required validation jobs.
