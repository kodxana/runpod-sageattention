#!/usr/bin/env bash
set -euo pipefail

# This probe deliberately never asks for a GPU. It verifies that a builder
# image contains the complete, internally consistent host-side toolchain before
# a paid Runpod Pod is launched.
VALIDATOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "${VALIDATOR_DIR}/activate-builder.sh" ]]; then
    # Repository checkout or bind-mounted CI probe.
    source "${VALIDATOR_DIR}/activate-builder.sh"
elif [[ -f /usr/local/lib/sageattention-builder/activate-builder.sh ]]; then
    # Helper installed into the builder image.
    source /usr/local/lib/sageattention-builder/activate-builder.sh
else
    echo "Builder activation helper is missing" >&2
    exit 69
fi
export CUDA_VISIBLE_DEVICES=""

usage() {
    cat <<'EOF'
Usage: validate-builder-image.sh [options]

Required options (or matching BUILDER_* environment variables):
  --cuda-version VERSION       CUDA toolkit release, for example 12.8
  --torch-version VERSION      Exact PyTorch version, for example 2.10.0+cu128
  --torch-cuda-version VERSION CUDA version reported by PyTorch
  --python-version VERSION     Python major.minor version, for example 3.12
  --nvcc-targets LIST          Semicolon-delimited native targets, for example
                               sm_80;sm_86;sm_89;sm_90a;sm_120
Optional as a complete set (defaults to matching BUILDER_* metadata):
  --build-version VERSION      Exact build frontend version
  --packaging-version VERSION  Exact packaging version
  --setuptools-version VERSION Exact setuptools version
  --wheel-version VERSION      Exact wheel version
EOF
}

EXPECTED_CUDA_VERSION="${BUILDER_CUDA_VERSION:-}"
EXPECTED_TORCH_VERSION="${BUILDER_TORCH_VERSION:-}"
EXPECTED_TORCH_CUDA_VERSION="${BUILDER_TORCH_CUDA_VERSION:-}"
EXPECTED_PYTHON_VERSION="${BUILDER_PYTHON_VERSION:-}"
NVCC_TARGETS="${BUILDER_NVCC_TARGETS:-}"
EXPECTED_BUILD_VERSION="${BUILDER_BUILD_VERSION:-}"
EXPECTED_PACKAGING_VERSION="${BUILDER_PACKAGING_VERSION:-}"
EXPECTED_SETUPTOOLS_VERSION="${BUILDER_SETUPTOOLS_VERSION:-}"
EXPECTED_WHEEL_VERSION="${BUILDER_WHEEL_VERSION:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-version)
            EXPECTED_CUDA_VERSION="${2:?missing value for --cuda-version}"
            shift 2
            ;;
        --torch-version)
            EXPECTED_TORCH_VERSION="${2:?missing value for --torch-version}"
            shift 2
            ;;
        --torch-cuda-version)
            EXPECTED_TORCH_CUDA_VERSION="${2:?missing value for --torch-cuda-version}"
            shift 2
            ;;
        --python-version)
            EXPECTED_PYTHON_VERSION="${2:?missing value for --python-version}"
            shift 2
            ;;
        --nvcc-targets)
            NVCC_TARGETS="${2:?missing value for --nvcc-targets}"
            shift 2
            ;;
        --build-version)
            EXPECTED_BUILD_VERSION="${2:?missing value for --build-version}"
            shift 2
            ;;
        --packaging-version)
            EXPECTED_PACKAGING_VERSION="${2:?missing value for --packaging-version}"
            shift 2
            ;;
        --setuptools-version)
            EXPECTED_SETUPTOOLS_VERSION="${2:?missing value for --setuptools-version}"
            shift 2
            ;;
        --wheel-version)
            EXPECTED_WHEEL_VERSION="${2:?missing value for --wheel-version}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

for variable in \
    EXPECTED_CUDA_VERSION \
    EXPECTED_TORCH_VERSION \
    EXPECTED_TORCH_CUDA_VERSION \
    EXPECTED_PYTHON_VERSION \
    NVCC_TARGETS; do
    if [[ -z "${!variable}" ]]; then
        echo "${variable} is required" >&2
        exit 64
    fi
done

[[ "${EXPECTED_CUDA_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]] || {
    echo "Invalid CUDA version: ${EXPECTED_CUDA_VERSION}" >&2
    exit 64
}
[[ "${EXPECTED_TORCH_CUDA_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]] || {
    echo "Invalid PyTorch CUDA version: ${EXPECTED_TORCH_CUDA_VERSION}" >&2
    exit 64
}
[[ "${EXPECTED_PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]] || {
    echo "Invalid Python version: ${EXPECTED_PYTHON_VERSION}" >&2
    exit 64
}

BUILD_FRONTEND_EXPECTATIONS=(
    "${EXPECTED_BUILD_VERSION}"
    "${EXPECTED_PACKAGING_VERSION}"
    "${EXPECTED_SETUPTOOLS_VERSION}"
    "${EXPECTED_WHEEL_VERSION}"
)
BUILD_FRONTEND_EXPECTATION_COUNT=0
for expectation in "${BUILD_FRONTEND_EXPECTATIONS[@]}"; do
    [[ -z "${expectation}" ]] || ((BUILD_FRONTEND_EXPECTATION_COUNT += 1))
done
if (( BUILD_FRONTEND_EXPECTATION_COUNT != 0 && BUILD_FRONTEND_EXPECTATION_COUNT != 4 )); then
    echo "Build frontend versions must be provided together as one exact set" >&2
    exit 64
fi

for required_command in git gcc g++ ninja patch nvcc ptxas cuobjdump python3.12; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "Builder image is missing required command: ${required_command}" >&2
        exit 69
    }
done

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ "${CUDA_HOME}" == /* ]] || {
    echo "CUDA_HOME must be absolute, got: ${CUDA_HOME}" >&2
    exit 65
}
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || {
    echo "Builder image is missing ${CUDA_HOME}/bin/nvcc" >&2
    exit 69
}

NVCC_PATH="$(command -v nvcc)"
if [[ "$(readlink -f -- "${NVCC_PATH}")" != "$(readlink -f -- "${CUDA_HOME}/bin/nvcc")" ]]; then
    echo "nvcc in PATH does not belong to CUDA_HOME: ${NVCC_PATH}" >&2
    exit 65
fi

NVCC_OUTPUT="$(nvcc --version)"
if [[ ! "${NVCC_OUTPUT}" =~ release[[:space:]]+([0-9]+\.[0-9]+) ]]; then
    echo "Could not parse nvcc release from: ${NVCC_OUTPUT}" >&2
    exit 65
fi
if [[ "${BASH_REMATCH[1]}" != "${EXPECTED_CUDA_VERSION}" ]]; then
    echo "nvcc mismatch: expected ${EXPECTED_CUDA_VERSION}, got ${BASH_REMATCH[1]}" >&2
    exit 65
fi

for metadata in \
    "BUILDER_CUDA_VERSION:${EXPECTED_CUDA_VERSION}" \
    "BUILDER_TORCH_VERSION:${EXPECTED_TORCH_VERSION}" \
    "BUILDER_TORCH_CUDA_VERSION:${EXPECTED_TORCH_CUDA_VERSION}" \
    "BUILDER_PYTHON_VERSION:${EXPECTED_PYTHON_VERSION}" \
    "BUILDER_NVCC_TARGETS:${NVCC_TARGETS}" \
    "BUILDER_BUILD_VERSION:${EXPECTED_BUILD_VERSION}" \
    "BUILDER_PACKAGING_VERSION:${EXPECTED_PACKAGING_VERSION}" \
    "BUILDER_SETUPTOOLS_VERSION:${EXPECTED_SETUPTOOLS_VERSION}" \
    "BUILDER_WHEEL_VERSION:${EXPECTED_WHEEL_VERSION}"; do
    name="${metadata%%:*}"
    expected="${metadata#*:}"
    actual="${!name:-}"
    # These metadata fields were expanded after the first published digests.
    # Their absence is acceptable because the independent runtime checks below
    # prove the tuple; a declared contradictory value is not.
    if [[ -n "${actual}" && "${actual}" != "${expected}" ]]; then
        echo "Builder image metadata mismatch for ${name}: expected ${expected}, got ${actual}" >&2
        exit 65
    fi
done

EXPECTED_PYTHON_VERSION="${EXPECTED_PYTHON_VERSION}" \
EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION}" \
EXPECTED_TORCH_CUDA_VERSION="${EXPECTED_TORCH_CUDA_VERSION}" \
EXPECTED_CUDA_HOME="$(readlink -f -- "${CUDA_HOME}")" \
EXPECTED_BUILD_VERSION="${EXPECTED_BUILD_VERSION}" \
EXPECTED_PACKAGING_VERSION="${EXPECTED_PACKAGING_VERSION}" \
EXPECTED_SETUPTOOLS_VERSION="${EXPECTED_SETUPTOOLS_VERSION}" \
EXPECTED_WHEEL_VERSION="${EXPECTED_WHEEL_VERSION}" \
python3.12 - <<'PY'
import importlib.metadata
import os
import pathlib
import sys

expected_python = tuple(
    int(part) for part in os.environ["EXPECTED_PYTHON_VERSION"].split(".")
)
actual_python = sys.version_info[:2]
if actual_python != expected_python:
    raise SystemExit(
        f"Python mismatch: expected {expected_python[0]}.{expected_python[1]}, "
        f"got {actual_python[0]}.{actual_python[1]}"
    )

import torch
from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home

expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
if str(torch.__version__) != expected_torch:
    raise SystemExit(
        f"PyTorch mismatch: expected {expected_torch}, got {torch.__version__}"
    )
expected_torch_cuda = os.environ["EXPECTED_TORCH_CUDA_VERSION"]
if str(torch.version.cuda) != expected_torch_cuda:
    raise SystemExit(
        f"PyTorch CUDA mismatch: expected {expected_torch_cuda}, "
        f"got {torch.version.cuda}"
    )
if torch_cuda_home is None:
    raise SystemExit("torch.utils.cpp_extension could not locate CUDA_HOME")
actual_cuda_home = pathlib.Path(torch_cuda_home).resolve()
expected_cuda_home = pathlib.Path(os.environ["EXPECTED_CUDA_HOME"])
if actual_cuda_home != expected_cuda_home:
    raise SystemExit(
        f"PyTorch CUDA_HOME mismatch: expected {expected_cuda_home}, "
        f"got {actual_cuda_home}"
    )

expected_build_frontend = {
    "build": os.environ["EXPECTED_BUILD_VERSION"],
    "packaging": os.environ["EXPECTED_PACKAGING_VERSION"],
    "setuptools": os.environ["EXPECTED_SETUPTOOLS_VERSION"],
    "wheel": os.environ["EXPECTED_WHEEL_VERSION"],
}
for distribution, expected_version in expected_build_frontend.items():
    if not expected_version:
        continue
    try:
        actual_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"{distribution} is not installed") from None
    if actual_version != expected_version:
        raise SystemExit(
            f"{distribution} mismatch: expected {expected_version}, "
            f"got {actual_version}"
        )
PY

require_cuda_file() {
    local description="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -e "${candidate}" ]]; then
            return
        fi
    done
    echo "Builder image is missing ${description}" >&2
    printf 'Checked: %s\n' "$*" >&2
    exit 69
}

CUDA_TARGET_ROOT="${CUDA_HOME}/targets/x86_64-linux"
require_cuda_file "CUDA runtime header" \
    "${CUDA_HOME}/include/cuda_runtime.h" \
    "${CUDA_TARGET_ROOT}/include/cuda_runtime.h"
require_cuda_file "cuBLAS header" \
    "${CUDA_HOME}/include/cublas_v2.h" \
    "${CUDA_TARGET_ROOT}/include/cublas_v2.h"
require_cuda_file "cuSOLVER header" \
    "${CUDA_HOME}/include/cusolverDn.h" \
    "${CUDA_TARGET_ROOT}/include/cusolverDn.h"
require_cuda_file "cuSPARSE header" \
    "${CUDA_HOME}/include/cusparse.h" \
    "${CUDA_TARGET_ROOT}/include/cusparse.h"
require_cuda_file "CUDA driver link stub" \
    "${CUDA_HOME}/lib64/stubs/libcuda.so" \
    "${CUDA_TARGET_ROOT}/lib/stubs/libcuda.so"
for library in libcudart libcublas libcusolver libcusparse; do
    require_cuda_file "${library} development library" \
        "${CUDA_HOME}/lib64/${library}.so" \
        "${CUDA_TARGET_ROOT}/lib/${library}.so"
done

IFS=';' read -r -a TARGETS <<< "${NVCC_TARGETS}"
if (( ${#TARGETS[@]} == 0 )); then
    echo "At least one NVCC target is required" >&2
    exit 64
fi

declare -A SEEN_TARGETS=()
for target in "${TARGETS[@]}"; do
    [[ "${target}" =~ ^sm_([0-9]+a?)$ ]] || {
        echo "Invalid NVCC target: ${target}" >&2
        exit 64
    }
    if [[ -n "${SEEN_TARGETS[${target}]:-}" ]]; then
        echo "Duplicate NVCC target: ${target}" >&2
        exit 64
    fi
    SEEN_TARGETS["${target}"]=1
done

PROBE_DIR="$(mktemp -d)"
trap 'rm -rf -- "${PROBE_DIR}"' EXIT
printf '%s\n' \
    'extern "C" __global__ void sageattention_builder_probe(int *value) {' \
    '    if (threadIdx.x == 0) *value = 1;' \
    '}' \
    > "${PROBE_DIR}/probe.cu"

for target in "${TARGETS[@]}"; do
    architecture="${target#sm_}"
    output="${PROBE_DIR}/probe-${target}.cubin"
    nvcc \
        --cubin \
        --std=c++17 \
        "--generate-code=arch=compute_${architecture},code=${target}" \
        "${PROBE_DIR}/probe.cu" \
        --output-file "${output}"
    [[ -s "${output}" ]] || {
        echo "nvcc produced no cubin for ${target}" >&2
        exit 70
    }
    # --dump-elf is the documented standalone-cubin probe; --list-elf is for
    # cubins embedded in a host fatbinary.
    cuobjdump --dump-elf "${output}" >/dev/null
done

printf '%s\n' \
    "Builder image validation passed without a GPU" \
    "  Python: ${EXPECTED_PYTHON_VERSION}" \
    "  PyTorch: ${EXPECTED_TORCH_VERSION}" \
    "  PyTorch CUDA: ${EXPECTED_TORCH_CUDA_VERSION}" \
    "  CUDA toolkit: ${EXPECTED_CUDA_VERSION}" \
    "  NVCC targets: ${NVCC_TARGETS}"
if (( BUILD_FRONTEND_EXPECTATION_COUNT == 4 )); then
    printf '%s\n' \
        "  build: ${EXPECTED_BUILD_VERSION}" \
        "  packaging: ${EXPECTED_PACKAGING_VERSION}" \
        "  setuptools: ${EXPECTED_SETUPTOOLS_VERSION}" \
        "  wheel: ${EXPECTED_WHEEL_VERSION}"
fi
