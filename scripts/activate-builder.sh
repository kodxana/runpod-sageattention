#!/usr/bin/env bash

# OpenSSH constructs its own restricted PATH for remote commands instead of
# retaining the image's Docker ENV. Source this file before using the builder
# toolchain so both SSH-driven builds and ordinary containers resolve the same
# pinned CUDA and Python installations.

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/sageattention-builder-venv}"

DEFAULT_SYSTEM_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CURRENT_PATH="${PATH:-${DEFAULT_SYSTEM_PATH}}"
export PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${CURRENT_PATH}"

CUDA_TARGET_ROOT="${CUDA_HOME}/targets/x86_64-linux"
CUDA_RUNTIME_PATH="${CUDA_HOME}/lib64:${CUDA_TARGET_ROOT}/lib"
CUDA_LINK_PATH="${CUDA_HOME}/lib64/stubs:${CUDA_TARGET_ROOT}/lib/stubs:${CUDA_RUNTIME_PATH}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_PATH}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_PATH}"
fi
if [[ -n "${LIBRARY_PATH:-}" ]]; then
    export LIBRARY_PATH="${CUDA_LINK_PATH}:${LIBRARY_PATH}"
else
    export LIBRARY_PATH="${CUDA_LINK_PATH}"
fi

unset DEFAULT_SYSTEM_PATH CURRENT_PATH CUDA_TARGET_ROOT CUDA_RUNTIME_PATH CUDA_LINK_PATH
