#!/usr/bin/env bash
set -euo pipefail

# Runpod's SSH server supplies a system-only PATH for remote commands. Restore
# the pinned virtual environment and CUDA toolkit paths from the builder image
# before testing for or invoking any build command.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/activate-builder.sh"

# Compilation is deliberately GPU-independent even when the container is
# scheduled on a GPU-backed Pod. Set this before any Python or PyTorch process.
export CUDA_VISIBLE_DEVICES=""

usage() {
    cat <<'EOF'
Usage: build-wheel.sh --build-id ID --output-dir DIRECTORY [options]

Options:
  --matrix PATH       Build matrix (default: <repo>/matrix.json)
  --source-dir PATH   Existing SageAttention git checkout at the pinned commit
  --keep-work         Keep the temporary build directory for diagnostics
EOF
}

REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MATRIX_PATH="${REPO_ROOT}/matrix.json"
BUILD_ID=""
OUTPUT_DIR=""
SOURCE_DIR="${SAGEATTN_SOURCE_DIR:-}"
KEEP_WORK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-id)
            BUILD_ID="${2:?missing value for --build-id}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --matrix)
            MATRIX_PATH="${2:?missing value for --matrix}"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="${2:?missing value for --source-dir}"
            shift 2
            ;;
        --keep-work)
            KEEP_WORK=1
            shift
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

[[ -n "${BUILD_ID}" ]] || { echo "--build-id is required" >&2; exit 64; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; exit 64; }

for command in git nvcc python3.12; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "Required build command is missing: ${command}" >&2
        exit 69
    }
done

MATRIX_PATH="$(python3.12 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${MATRIX_PATH}")"
OUTPUT_DIR="$(python3.12 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_DIR}"
if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Output directory must be empty: ${OUTPUT_DIR}" >&2
    exit 73
fi

WORK_PARENT="${SAGEATTN_WORK_ROOT:-/work/sageattention-wheel-builds}"
mkdir -p "${WORK_PARENT}"
WORK_PARENT="$(python3.12 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${WORK_PARENT}")"
if [[ "${WORK_PARENT}" == "/" ]]; then
    echo "Refusing to use the filesystem root as the build work directory" >&2
    exit 73
fi

mapfile -t MATRIX_VALUES < <(python3.12 - "${MATRIX_PATH}" "${BUILD_ID}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    matrix = json.load(handle)
matches = [item for item in matrix["builds"] if item["id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"expected one build named {sys.argv[2]!r}, found {len(matches)}")
build = matches[0]
package = matrix["package"]
platform = matrix["platform"]
policy = matrix["cuda_policy"]
resources = matrix["resources"]
build_frontend = matrix["build_frontend"]
values = [
    package["source_url"],
    package["source_commit"],
    str(package["source_date_epoch"]),
    package["upstream_version"],
    platform["python_version"],
    platform["python_tag"],
    policy["torch_cuda_arch_list"],
    build["cuda_version"],
    build["torch_version"],
    build["torch_cuda_version"],
    build["wheel_version"],
    build["wheel_filename"],
    build["builder_image"],
    str(resources["minimum_cpus"]),
    str(resources["minimum_memory_gib"]),
    str(resources["recommended_memory_gib"]),
    str(resources["minimum_free_disk_gib"]),
    str(resources["default_max_jobs"]),
    str(resources["default_extension_parallelism"]),
    str(resources["compiler_memory_per_job_mib"]),
    build_frontend["build"],
    build_frontend["packaging"],
    build_frontend["setuptools"],
    build_frontend["wheel"],
]
if any("\n" in value or "\r" in value for value in values):
    raise SystemExit("matrix values must be single-line strings")
print("\n".join(values))
PY
)

if [[ ${#MATRIX_VALUES[@]} -ne 24 ]]; then
    echo "Could not read a complete build definition from ${MATRIX_PATH}" >&2
    exit 65
fi

SOURCE_URL="${MATRIX_VALUES[0]}"
SOURCE_COMMIT="${MATRIX_VALUES[1]}"
SOURCE_DATE_EPOCH="${MATRIX_VALUES[2]}"
UPSTREAM_VERSION="${MATRIX_VALUES[3]}"
PYTHON_VERSION="${MATRIX_VALUES[4]}"
PYTHON_TAG="${MATRIX_VALUES[5]}"
TORCH_CUDA_ARCH_LIST="${MATRIX_VALUES[6]}"
CUDA_VERSION="${MATRIX_VALUES[7]}"
TORCH_VERSION="${MATRIX_VALUES[8]}"
TORCH_CUDA_VERSION="${MATRIX_VALUES[9]}"
WHEEL_VERSION="${MATRIX_VALUES[10]}"
WHEEL_FILENAME="${MATRIX_VALUES[11]}"
BUILDER_IMAGE_EXPECTED="${MATRIX_VALUES[12]}"
MIN_CPUS="${MATRIX_VALUES[13]}"
MIN_MEMORY_GIB="${MATRIX_VALUES[14]}"
RECOMMENDED_MEMORY_GIB="${MATRIX_VALUES[15]}"
MIN_DISK_GIB="${MATRIX_VALUES[16]}"
DEFAULT_MAX_JOBS="${MATRIX_VALUES[17]}"
DEFAULT_EXT_PARALLEL="${MATRIX_VALUES[18]}"
COMPILER_MEMORY_PER_JOB_MIB="${MATRIX_VALUES[19]}"
BUILD_FRONTEND_BUILD="${MATRIX_VALUES[20]}"
BUILD_FRONTEND_PACKAGING="${MATRIX_VALUES[21]}"
BUILD_FRONTEND_SETUPTOOLS="${MATRIX_VALUES[22]}"
BUILD_FRONTEND_WHEEL="${MATRIX_VALUES[23]}"

if [[ "${TORCH_CUDA_ARCH_LIST}" == *"+PTX"* || "${TORCH_CUDA_ARCH_LIST}" == *" "* ]]; then
    echo "Architecture list must be semicolon-delimited native SASS targets: ${TORCH_CUDA_ARCH_LIST}" >&2
    exit 65
fi

if [[ -f "${REPO_ROOT}/tools/pod_resources.py" ]]; then
    RESOURCE_HELPER=(python3.12 "${REPO_ROOT}/tools/pod_resources.py")
elif command -v pod-resources >/dev/null 2>&1; then
    RESOURCE_HELPER=(pod-resources)
else
    echo "pod-resources is required for cgroup-aware build preflight" >&2
    exit 69
fi

export POD_BUILD_MEMORY_PER_JOB_MIB="${COMPILER_MEMORY_PER_JOB_MIB}"
if [[ -n "${MAX_JOBS:-}" ]]; then
    [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
        echo "MAX_JOBS must be a positive integer" >&2
        exit 65
    }
    if (( MAX_JOBS > DEFAULT_MAX_JOBS )) && [[ "${ALLOW_UNSAFE_PARALLELISM:-0}" != "1" ]]; then
        echo "MAX_JOBS=${MAX_JOBS} exceeds the reviewed matrix cap ${DEFAULT_MAX_JOBS}; set ALLOW_UNSAFE_PARALLELISM=1 only on a larger Pod" >&2
        exit 65
    fi
    export POD_BUILD_MAX_JOBS="${MAX_JOBS}"
else
    export POD_BUILD_MAX_JOBS="${DEFAULT_MAX_JOBS}"
fi
RESOURCE_SNAPSHOT_JSON="$("${RESOURCE_HELPER[@]}" --json)" || {
    echo "pod-resources failed; refusing an unconstrained build" >&2
    exit 70
}

MIN_CPUS="${MIN_CPUS}" MIN_MEMORY_GIB="${MIN_MEMORY_GIB}" \
RECOMMENDED_MEMORY_GIB="${RECOMMENDED_MEMORY_GIB}" MIN_DISK_GIB="${MIN_DISK_GIB}" \
OUTPUT_DIR="${OUTPUT_DIR}" WORK_PARENT="${WORK_PARENT}" \
ALLOW_LOW_RESOURCES="${ALLOW_LOW_RESOURCES:-0}" \
RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 - <<'PY'
import json
import os
import shutil

snapshot = json.loads(os.environ["RESOURCE_SNAPSHOT_JSON"])
memory = snapshot["memory"]
cpu = snapshot["cpu"]
build = snapshot["build"]
capacity = memory.get("capacity_bytes")
capacity_source = memory.get("capacity_source")
capacity_is_hard_limit = memory.get("capacity_is_hard_limit")
assigned_capacity = memory.get("assigned_capacity_bytes")
usage_source = memory.get("usage_source")
usage_trustworthy = memory.get("usage_trustworthy")
usage_peak_eligible = memory.get("usage_peak_eligible")
usage_scope = memory.get("usage_scope")
available = memory.get("available_bytes")
minimum_cpus = int(os.environ["MIN_CPUS"])
minimum_limit = float(os.environ["MIN_MEMORY_GIB"]) * 1024 ** 3
recommended_limit = float(os.environ["RECOMMENDED_MEMORY_GIB"]) * 1024 ** 3
minimum_disk = float(os.environ["MIN_DISK_GIB"])
disk_free_gib = {
    "output": shutil.disk_usage(os.environ["OUTPUT_DIR"]).free / 1024 ** 3,
    "work": shutil.disk_usage(os.environ["WORK_PARENT"]).free / 1024 ** 3,
}
hard_failures = []
diagnostic_memory_failures = []

def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

if snapshot.get("schema_version") != 2:
    hard_failures.append("pod-resources schema version 2 is required")
if not positive_integer(capacity):
    hard_failures.append("memory capacity is missing or invalid")
elif capacity < minimum_limit:
    hard_failures.append(
        f"memory capacity {capacity / 1024 ** 3:.1f} GiB "
        f"< {minimum_limit / 1024 ** 3:.1f} GiB")
if not positive_integer(assigned_capacity):
    hard_failures.append(
        "a positive receipt-backed Runpod API memory assignment is required")

cgroup_version = snapshot.get("cgroup", {}).get("version")
is_cgroup_capacity = (
    cgroup_version in {1, 2}
    and isinstance(capacity_source, str)
    and capacity_source.startswith(f"cgroup-v{cgroup_version}:")
)
if capacity_source == "runpod-api-assignment":
    if (not positive_integer(assigned_capacity)
            or assigned_capacity != capacity
            or capacity_is_hard_limit is not False):
        hard_failures.append("Runpod assignment capacity evidence is inconsistent")
elif is_cgroup_capacity:
    if (capacity_is_hard_limit is not True
            or memory.get("limited") is not True
            or memory.get("limit_bytes") != capacity):
        hard_failures.append("cgroup hard-limit capacity evidence is inconsistent")
    if positive_integer(assigned_capacity) and capacity > assigned_capacity:
        hard_failures.append("cgroup capacity does not honor the Runpod assignment")
else:
    hard_failures.append(
        "neither a finite cgroup hard limit nor a verified Runpod API assignment "
        "establishes memory capacity")

if cpu["effective_count"] < minimum_cpus:
    hard_failures.append(f"effective CPU {cpu['effective_count']} < {minimum_cpus}")
if usage_peak_eligible is True and positive_integer(capacity):
    if not isinstance(available, int) or isinstance(available, bool):
        diagnostic_memory_failures.append("pod-scoped memory headroom is unknown")
    else:
        required_headroom = build["reserve_bytes"] + build["memory_per_job_bytes"]
        if available < required_headroom:
            diagnostic_memory_failures.append(
                f"memory headroom {available / 1024 ** 3:.1f} GiB cannot cover "
                f"reserve plus one compiler ({required_headroom / 1024 ** 3:.1f} GiB)")
elif not (build.get("forced_single_job") is True
          and build.get("suggested_jobs") == 1):
    hard_failures.append(
        "untrusted memory usage must force a one-job build recommendation")

if usage_peak_eligible is not True:
    hard_failures.append("no cgroup membership counter is eligible for peak evidence")
elif not isinstance(usage_source, str) or not usage_source.startswith("cgroup-v"):
    hard_failures.append("peak-eligible memory usage has no cgroup source")

for label, free_gib in disk_free_gib.items():
    if free_gib < minimum_disk:
        hard_failures.append(
            f"{label} filesystem free disk {free_gib:.1f} GiB "
            f"< {minimum_disk:.1f} GiB")

capacity_text = (
    "unknown" if not positive_integer(capacity)
    else f"{capacity / 1024 ** 3:.1f} GiB"
)
available_text = "unknown" if available is None else f"{available / 1024 ** 3:.1f} GiB"
print(
    f"CPU-bound preflight: cpus={cpu['effective_count']}, "
    f"memory_capacity={capacity_text}, capacity_source={capacity_source}, "
    f"hard_limit={capacity_is_hard_limit}, usage_trustworthy={usage_trustworthy}, "
    f"usage_scope={usage_scope}, "
    f"memory_headroom={available_text}, "
    f"output_free_disk={disk_free_gib['output']:.1f} GiB, "
    f"work_free_disk={disk_free_gib['work']:.1f} GiB, "
    f"recommended_limit={recommended_limit / 1024 ** 3:.1f} GiB"
)
if hard_failures:
    raise SystemExit("resource preflight failed: " + "; ".join(hard_failures))
if diagnostic_memory_failures:
    message = "resource preflight failed: " + "; ".join(diagnostic_memory_failures)
    if os.environ.get("ALLOW_LOW_RESOURCES") == "1":
        print("WARNING: " + message)
    else:
        raise SystemExit(message + "; set ALLOW_LOW_RESOURCES=1 only intentionally")
PY

FORCED_SINGLE_JOB="$(RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 -c \
    'import json,os; print("1" if json.loads(os.environ["RESOURCE_SNAPSHOT_JSON"])["build"].get("forced_single_job") is True else "0")')"
if [[ "${FORCED_SINGLE_JOB}" == "1" ]]; then
    if [[ "${MAX_JOBS:-1}" != "1" ]]; then
        echo "WARNING: ignoring MAX_JOBS=${MAX_JOBS}; untrusted Pod memory usage forces MAX_JOBS=1" >&2
    fi
    MAX_JOBS=1
elif [[ -z "${MAX_JOBS:-}" ]]; then
    MAX_JOBS="$(RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 -c \
        'import json,os; print(json.loads(os.environ["RESOURCE_SNAPSHOT_JSON"])["build"]["suggested_jobs"])')"
fi
EXT_PARALLEL="${EXT_PARALLEL:-${DEFAULT_EXT_PARALLEL}}"
if [[ "${FORCED_SINGLE_JOB}" == "1" && "${EXT_PARALLEL}" != "1" ]]; then
    echo "WARNING: ignoring EXT_PARALLEL=${EXT_PARALLEL}; untrusted Pod memory usage forces EXT_PARALLEL=1" >&2
    EXT_PARALLEL=1
fi
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_JOBS must be a positive integer" >&2; exit 65; }
[[ "${EXT_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo "EXT_PARALLEL must be a positive integer" >&2; exit 65; }
if (( EXT_PARALLEL > DEFAULT_EXT_PARALLEL )) && [[ "${ALLOW_UNSAFE_PARALLELISM:-0}" != "1" ]]; then
    echo "EXT_PARALLEL=${EXT_PARALLEL} exceeds the reviewed matrix cap ${DEFAULT_EXT_PARALLEL}; set ALLOW_UNSAFE_PARALLELISM=1 only on a larger Pod" >&2
    exit 65
fi

cgroup_peak_from_snapshot() {
    RESOURCE_SNAPSHOT_JSON="$1" python3.12 - <<'PY'
import json
import os
from pathlib import Path

snapshot = json.loads(os.environ["RESOURCE_SNAPSHOT_JSON"])
version = snapshot.get("cgroup", {}).get("version")
memory = snapshot.get("memory", {})
source = str(memory.get("usage_source") or "")
prefix = f"cgroup-v{version}:"
if (version not in {1, 2} or memory.get("usage_peak_eligible") is not True
        or not source.startswith(prefix)):
    print("")
    raise SystemExit(0)
directory = Path(source[len(prefix):])
peak_file = directory / ("memory.peak" if version == 2 else "memory.max_usage_in_bytes")
try:
    print(int(peak_file.read_text(encoding="ascii").strip()))
except (OSError, ValueError):
    print("")
PY
}

BUILD_STARTED_SECONDS="$(date +%s)"
CGROUP_PEAK_START="$(cgroup_peak_from_snapshot "${RESOURCE_SNAPSHOT_JSON}")"
MEMORY_CAPACITY_BYTES="$(RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 -c \
    'import json,os; print(json.loads(os.environ["RESOURCE_SNAPSHOT_JSON"])["memory"]["capacity_bytes"])')"
if [[ ! "${CGROUP_PEAK_START}" =~ ^[1-9][0-9]*$ ]]; then
    echo "A positive cgroup-membership memory peak is required before compilation" >&2
    exit 70
fi
if [[ ! "${MEMORY_CAPACITY_BYTES}" =~ ^[1-9][0-9]*$ ]] \
        || (( CGROUP_PEAK_START > MEMORY_CAPACITY_BYTES )); then
    echo "Initial cgroup memory peak exceeds the selected memory capacity" >&2
    exit 70
fi

if [[ -n "${BUILDER_CUDA_VERSION:-}" && "${BUILDER_CUDA_VERSION}" != "${CUDA_VERSION}" ]]; then
    echo "Builder CUDA mismatch: image=${BUILDER_CUDA_VERSION}, matrix=${CUDA_VERSION}" >&2
    exit 65
fi
if [[ -n "${BUILDER_TORCH_VERSION:-}" && "${BUILDER_TORCH_VERSION}" != "${TORCH_VERSION}" ]]; then
    echo "Builder torch mismatch: image=${BUILDER_TORCH_VERSION}, matrix=${TORCH_VERSION}" >&2
    exit 65
fi

EXPECTED_BUILD_FRONTEND_BUILD="${BUILD_FRONTEND_BUILD}" \
EXPECTED_BUILD_FRONTEND_PACKAGING="${BUILD_FRONTEND_PACKAGING}" \
EXPECTED_BUILD_FRONTEND_SETUPTOOLS="${BUILD_FRONTEND_SETUPTOOLS}" \
EXPECTED_BUILD_FRONTEND_WHEEL="${BUILD_FRONTEND_WHEEL}" \
EXPECTED_PYTHON="${PYTHON_VERSION}" EXPECTED_TORCH="${TORCH_VERSION}" \
EXPECTED_TORCH_CUDA="${TORCH_CUDA_VERSION}" EXPECTED_CUDA="${CUDA_VERSION}" python3.12 - <<'PY'
import importlib.metadata
import os
import re
import subprocess
import sys
import torch

expected_frontend = {
    "build": os.environ["EXPECTED_BUILD_FRONTEND_BUILD"],
    "packaging": os.environ["EXPECTED_BUILD_FRONTEND_PACKAGING"],
    "setuptools": os.environ["EXPECTED_BUILD_FRONTEND_SETUPTOOLS"],
    "wheel": os.environ["EXPECTED_BUILD_FRONTEND_WHEEL"],
}
for distribution, expected in expected_frontend.items():
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        actual = "<missing>"
    if actual != expected:
        raise SystemExit(
            f"build frontend mismatch for {distribution}: "
            f"expected {expected}, got {actual}"
        )

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if python_version != os.environ["EXPECTED_PYTHON"]:
    raise SystemExit(f"Python mismatch: expected {os.environ['EXPECTED_PYTHON']}, got {python_version}")
if str(torch.__version__) != os.environ["EXPECTED_TORCH"]:
    raise SystemExit(
        f"torch mismatch: expected {os.environ['EXPECTED_TORCH']}, got {torch.__version__}")
if str(torch.version.cuda) != os.environ["EXPECTED_TORCH_CUDA"]:
    raise SystemExit(
        f"torch CUDA mismatch: expected {os.environ['EXPECTED_TORCH_CUDA']}, got {torch.version.cuda}")
nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc)
if not match or match.group(1) != os.environ["EXPECTED_CUDA"]:
    raise SystemExit(
        f"nvcc mismatch: expected {os.environ['EXPECTED_CUDA']}, output was:\n{nvcc}")
PY

WORK_DIR="$(mktemp -d "${WORK_PARENT%/}/${BUILD_ID}.XXXXXX")"
if [[ "${WORK_DIR}" != "${WORK_PARENT}/"* ]]; then
    echo "Refusing unsafe temporary build path: ${WORK_DIR}" >&2
    exit 73
fi
cleanup() {
    if [[ "${KEEP_WORK}" == "1" ]]; then
        echo "Keeping build directory: ${WORK_DIR}"
    else
        rm -rf -- "${WORK_DIR}"
    fi
}
trap cleanup EXIT

CHECKOUT="${WORK_DIR}/source"
export GIT_TERMINAL_PROMPT=0
if [[ -n "${SOURCE_DIR}" ]]; then
    SOURCE_DIR="$(python3.12 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${SOURCE_DIR}")"
    actual_source_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
    if [[ "${actual_source_commit}" != "${SOURCE_COMMIT}" ]]; then
        echo "Source checkout mismatch: expected ${SOURCE_COMMIT}, got ${actual_source_commit}" >&2
        exit 65
    fi
    git clone --quiet --no-hardlinks --no-checkout "${SOURCE_DIR}" "${CHECKOUT}"
    git -C "${CHECKOUT}" checkout --quiet --detach "${SOURCE_COMMIT}"
else
    git init --quiet "${CHECKOUT}"
    git -C "${CHECKOUT}" remote add origin "${SOURCE_URL}"
    git -C "${CHECKOUT}" fetch --quiet --depth 1 origin "${SOURCE_COMMIT}"
    git -C "${CHECKOUT}" checkout --quiet --detach FETCH_HEAD
fi

actual_commit="$(git -C "${CHECKOUT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${SOURCE_COMMIT}" ]] || {
    echo "Fetched source mismatch: expected ${SOURCE_COMMIT}, got ${actual_commit}" >&2
    exit 65
}

PATCH_FILE="${REPO_ROOT}/patches/sageattention/${UPSTREAM_VERSION}/setup.py.patch"
git -C "${CHECKOUT}" apply --check "${PATCH_FILE}"
git -C "${CHECKOUT}" apply "${PATCH_FILE}"
git -C "${CHECKOUT}" diff --check

WHEEL_STAGE="${WORK_DIR}/release"
mkdir -p "${WHEEL_STAGE}"

export EXT_PARALLEL
export MAX_JOBS
export PYTHONHASHSEED=0
export SAGEATTN_TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION}"
export SAGEATTN_TORCH_VERSION="${TORCH_VERSION}"
export SAGEATTN_WHEEL_VERSION="${WHEEL_VERSION}"
export SOURCE_DATE_EPOCH
export TORCH_CUDA_ARCH_LIST
export CXX_APPEND_FLAGS="${CXX_APPEND_FLAGS:-} -ffile-prefix-map=${CHECKOUT}=/usr/src/sageattention -fdebug-prefix-map=${CHECKOUT}=/usr/src/sageattention"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -Xcompiler=-ffile-prefix-map=${CHECKOUT}=/usr/src/sageattention"

echo "Building ${WHEEL_FILENAME} with MAX_JOBS=${MAX_JOBS}, EXT_PARALLEL=${EXT_PARALLEL}"
python3.12 -m build --wheel --no-isolation --outdir "${WHEEL_STAGE}" "${CHECKOUT}"

mapfile -t BUILT_WHEELS < <(find "${WHEEL_STAGE}" -maxdepth 1 -type f -name '*.whl' -print)
if [[ ${#BUILT_WHEELS[@]} -ne 1 ]]; then
    echo "Expected one wheel, found ${#BUILT_WHEELS[@]}" >&2
    exit 65
fi
BUILT_WHEEL="${BUILT_WHEELS[0]}"
if [[ "$(basename -- "${BUILT_WHEEL}")" != "${WHEEL_FILENAME}" ]]; then
    echo "Wheel filename mismatch: expected ${WHEEL_FILENAME}, got $(basename -- "${BUILT_WHEEL}")" >&2
    exit 65
fi

RESOURCE_END_JSON="$("${RESOURCE_HELPER[@]}" --json)" || {
    echo "pod-resources failed after build; build evidence would be incomplete" >&2
    exit 70
}
RESOURCE_START_JSON="${RESOURCE_SNAPSHOT_JSON}" RESOURCE_END_JSON="${RESOURCE_END_JSON}" \
python3.12 - <<'PY'
import json
import os

start = json.loads(os.environ["RESOURCE_START_JSON"])
end = json.loads(os.environ["RESOURCE_END_JSON"])
memory_fields = (
    "assigned_capacity_bytes",
    "capacity_bytes",
    "capacity_source",
    "capacity_is_hard_limit",
    "usage_source",
    "usage_scope",
    "usage_peak_eligible",
    "usage_trustworthy",
)
for field in memory_fields:
    if end["memory"].get(field) != start["memory"].get(field):
        raise SystemExit(f"resource assignment changed during build: {field}")
if end["build"].get("forced_single_job") != start["build"].get("forced_single_job"):
    raise SystemExit("forced-single-job policy changed during build")
PY
CGROUP_PEAK_END="$(cgroup_peak_from_snapshot "${RESOURCE_END_JSON}")"
if [[ ! "${CGROUP_PEAK_END}" =~ ^[1-9][0-9]*$ ]]; then
    echo "A positive cgroup-membership memory peak is required after compilation" >&2
    exit 70
fi
if (( CGROUP_PEAK_END < CGROUP_PEAK_START \
        || CGROUP_PEAK_END > MEMORY_CAPACITY_BYTES )); then
    echo "Final cgroup memory peak is non-monotonic or exceeds selected capacity" >&2
    exit 70
fi
BUILD_FINISHED_SECONDS="$(date +%s)"
BUILD_ELAPSED_SECONDS="$((BUILD_FINISHED_SECONDS - BUILD_STARTED_SECONDS))"
EVIDENCE_FILE="${WHEEL_STAGE}/build-evidence.json"
RESOURCE_START_JSON="${RESOURCE_SNAPSHOT_JSON}" RESOURCE_END_JSON="${RESOURCE_END_JSON}" \
MATRIX_PATH="${MATRIX_PATH}" PATCH_FILE="${PATCH_FILE}" MAX_JOBS="${MAX_JOBS}" \
EXT_PARALLEL="${EXT_PARALLEL}" BUILD_ELAPSED_SECONDS="${BUILD_ELAPSED_SECONDS}" \
CGROUP_PEAK_START="${CGROUP_PEAK_START}" CGROUP_PEAK_END="${CGROUP_PEAK_END}" \
BUILDER_IMAGE_EXPECTED="${BUILDER_IMAGE_EXPECTED}" \
BUILDER_IMAGE_REF="${BUILDER_IMAGE_REF:-${BUILDER_IMAGE_EXPECTED}}" \
BUILDER_IMAGE_DIGEST="${BUILDER_IMAGE_DIGEST:-}" \
ALLOW_UNSAFE_PARALLELISM="${ALLOW_UNSAFE_PARALLELISM:-0}" \
python3.12 - "${EVIDENCE_FILE}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


nvcc_output = subprocess.check_output(["nvcc", "--version"], text=True)
nvcc_release = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_output)
peak_start = os.environ["CGROUP_PEAK_START"]
peak_end = os.environ["CGROUP_PEAK_END"]
builder_ref = os.environ["BUILDER_IMAGE_REF"] or None
builder_digest = os.environ["BUILDER_IMAGE_DIGEST"] or None
if builder_digest is None and builder_ref and "@sha256:" in builder_ref:
    builder_digest = builder_ref.split("@", 1)[1]
selected_gpu_id = os.environ.get("RUNPOD_SELECTED_GPU_ID") or None
if selected_gpu_id is not None and (
    selected_gpu_id != selected_gpu_id.strip()
    or "\n" in selected_gpu_id
    or "\r" in selected_gpu_id
    or "," in selected_gpu_id
):
    raise SystemExit("RUNPOD_SELECTED_GPU_ID must be one exact Runpod gpuId")
resource_start = json.loads(os.environ["RESOURCE_START_JSON"])
resource_end = json.loads(os.environ["RESOURCE_END_JSON"])
start_memory = resource_start["memory"]
evidence = {
    "builder_image": {
        "digest": builder_digest,
        "expected": os.environ["BUILDER_IMAGE_EXPECTED"],
        "ref": builder_ref,
    },
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cgroup_peak": {
        "end_bytes": int(peak_end) if peak_end else None,
        "monotonic": int(peak_end) >= int(peak_start),
        "scope": start_memory["usage_scope"],
        "source": start_memory["usage_source"],
        "start_bytes": int(peak_start) if peak_start else None,
        "usage_trustworthy": start_memory["usage_trustworthy"],
        "within_capacity": int(peak_end) <= start_memory["capacity_bytes"],
    },
    "elapsed_seconds": int(os.environ["BUILD_ELAPSED_SECONDS"]),
    "matrix_sha256": sha256(os.environ["MATRIX_PATH"]),
    "patch_sha256": sha256(os.environ["PATCH_FILE"]),
    "memory_policy": {
        "assigned_capacity_bytes": start_memory["assigned_capacity_bytes"],
        "capacity_bytes": start_memory["capacity_bytes"],
        "capacity_is_hard_limit": start_memory["capacity_is_hard_limit"],
        "capacity_source": start_memory["capacity_source"],
        "forced_single_job": resource_start["build"]["forced_single_job"],
        "usage_peak_eligible": start_memory["usage_peak_eligible"],
        "usage_scope": start_memory["usage_scope"],
        "usage_source": start_memory["usage_source"],
        "usage_trustworthy": start_memory["usage_trustworthy"],
    },
    "resource_end": resource_end,
    "resource_start": resource_start,
    "runpod_assignment": {
        "memory_bytes": start_memory["assigned_capacity_bytes"],
        "vcpu_count": resource_start["cpu"]["runpod_count"],
    },
    "selected_gpu_id": selected_gpu_id,
    "selected_parallelism": {
        "extension_parallelism": int(os.environ["EXT_PARALLEL"]),
        "max_jobs": int(os.environ["MAX_JOBS"]),
        "unsafe_override": os.environ["ALLOW_UNSAFE_PARALLELISM"] == "1",
    },
    "tool_versions": {
        "build": importlib.metadata.version("build"),
        "gcc": subprocess.check_output(["gcc", "-dumpfullversion"], text=True).strip(),
        "nvcc": nvcc_release.group(1) if nvcc_release else nvcc_output.strip(),
        "packaging": importlib.metadata.version("packaging"),
        "python": platform.python_version(),
        "setuptools": importlib.metadata.version("setuptools"),
        "torch": importlib.metadata.version("torch"),
        "wheel": importlib.metadata.version("wheel"),
    },
}
path = Path(sys.argv[1])
path.write_text(
    json.dumps(evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

python3.12 "${REPO_ROOT}/scripts/inspect-wheel.py" \
    --matrix "${MATRIX_PATH}" \
    --build-id "${BUILD_ID}" \
    --wheel "${BUILT_WHEEL}" \
    --evidence "${EVIDENCE_FILE}" \
    --manifest "${WHEEL_STAGE}/manifest.json" \
    --checksums "${WHEEL_STAGE}/SHA256SUMS"

python3.12 "${REPO_ROOT}/scripts/validate-wheel.py" \
    --matrix "${MATRIX_PATH}" \
    --build-id "${BUILD_ID}" \
    --wheel "${BUILT_WHEEL}" \
    --manifest "${WHEEL_STAGE}/manifest.json" \
    --checksums "${WHEEL_STAGE}/SHA256SUMS"

install -m 0644 "${BUILT_WHEEL}" "${OUTPUT_DIR}/${WHEEL_FILENAME}"
install -m 0644 "${WHEEL_STAGE}/manifest.json" "${OUTPUT_DIR}/manifest.json"
install -m 0644 "${WHEEL_STAGE}/SHA256SUMS" "${OUTPUT_DIR}/SHA256SUMS"

echo "Build complete: ${OUTPUT_DIR}"
