# Pod-aware resource reporting

Linux containers normally share the host's `/proc/meminfo`. As a result,
`free`, `top`, and some `htop` builds can display host RAM even when the pod has
a much smaller cgroup limit. The kernel still enforces the cgroup limit; the
misleading display does not grant the pod extra memory.

SageAttention builders are GPU-backed, but the build hides the accelerator with
`CUDA_VISIBLE_DEVICES=""`. Resource policy therefore concerns the Pod's
host-system vCPU, system RAM, and container disk—not GPU utilization or VRAM.
The attached GPU type is a scheduling choice and cannot substitute for a finite
system-memory cgroup limit.

This repository handles the problem in two layers:

1. `/usr/local/bin/pod-resources` is the authoritative source for automation.
   It resolves cgroups directly and reports machine-readable memory, CPU, and
   safe compiler parallelism data.
2. `/usr/local/bin/free`, `htop`, and `top` are optional presentation wrappers.
   They load a narrow compatibility library only for that one process so an
   operator sees the pod memory total. They must not drive build policy.

## Authoritative resolver

Install the dependency-free Python helper in the image:

```sh
install -m 0755 tools/pod_resources.py /usr/local/bin/pod-resources
```

Use JSON for programs, shell assignments for entrypoints, or request only the
job count:

```sh
pod-resources --pretty
eval "$(pod-resources --shell)"
jobs=$(pod-resources --suggested-jobs)
```

The JSON schema is versioned. Schema version 1 has this shape (values below are
illustrative):

```json
{
  "schema_version": 1,
  "cgroup": {
    "version": 2,
    "path": "/pod/container",
    "mount_point": "/sys/fs/cgroup"
  },
  "memory": {
    "limited": true,
    "host_total_bytes": 137438953472,
    "limit_bytes": 34359738368,
    "current_bytes": 7516192768,
    "working_set_bytes": 6442450944,
    "inactive_file_bytes": 1073741824,
    "free_bytes": 26843545600,
    "available_bytes": 27917287424,
    "high_bytes": null,
    "limit_source": "cgroup-v2:/sys/fs/cgroup/pod",
    "swap_limit_bytes": 0,
    "swap_current_bytes": 0
  },
  "cpu": {
    "host_count": 64,
    "affinity_count": 16,
    "cpuset_count": 8,
    "quota_cores": 4.0,
    "quota_job_count": 4,
    "runpod_count": 8,
    "effective_count": 4,
    "limiting_sources": ["quota"]
  },
  "build": {
    "suggested_jobs": 2,
    "jobs_by_cpu": 4,
    "jobs_by_memory": 2,
    "memory_per_job_bytes": 8589934592,
    "reserve_bytes": 5153960756,
    "usable_memory_bytes": 22763326668,
    "max_jobs_cap": 4
  },
  "warnings": []
}
```

Automation should require all of the following before starting a SageAttention
build:

- `schema_version == 1`
- `cgroup.version` is `1` or `2`
- `memory.limited == true`
- `memory.limit_bytes > 0`
- `build.suggested_jobs >= 1`

If that validation fails, stop the remote build unless a deliberate operator
override selects one job. `limit_bytes` falls back to the host total when
`memory.limited` is false, so checking only that number is not sufficient.
Warnings are diagnostic; a missing current-usage counter is handled
conservatively as zero available headroom.

Default GPU-backed release policy requires an assignment with at least 4
effective vCPUs, 32 GB system RAM, and an 80 GB container disk before source
upload; 16 vCPUs and 64 GB system RAM are recommended. The backend-neutral
helper and build script then independently require a finite 32 GiB cgroup
limit, at least one safe compiler job, and 20 GiB currently free on both work
and output filesystems. Disk is checked by `build-wheel.sh`, not by the
resource-reporting helper.

The shell form exports `POD_MEMORY_LIMIT_BYTES`,
`POD_MEMORY_CURRENT_BYTES`, `POD_MEMORY_WORKING_SET_BYTES`,
`POD_MEMORY_AVAILABLE_BYTES`, `POD_CPU_COUNT`, `POD_BUILD_JOBS`, and the
assumptions used for the recommendation, among other `POD_*` values. Values are
shell-quoted and no input-derived variable names are emitted.

## How limits are resolved

The resolver parses both `/proc/self/cgroup` and `/proc/self/mountinfo`; it does
not assume that cgroups are mounted at a fixed path. It maps the process's
membership through the mount root, including the `/` membership commonly seen
inside a private cgroup namespace.

For cgroup v2 it reads `memory.max`, `memory.current`, `memory.stat`,
`memory.high`, `memory.swap.max`, `memory.swap.current`, `cpu.max`, and
`cpuset.cpus.effective`. For cgroup v1 it uses the corresponding memory,
memsw, CPU CFS, and cpuset controller files. The v1
`hierarchical_memory_limit` statistic is considered when an inherited parent
is hidden by a cgroup namespace.

The smallest finite hard limit in every visible ancestor up to the controller
mount is authoritative. When an ancestor supplies that limit, current usage
and memory statistics are read from the same ancestor so sibling usage inside
the constrained subtree is not missed. `max`, `-1`, and the conventional huge
v1 sentinel are unlimited; zero remains a real finite value.

Memory values use these definitions:

```text
working_set = max(0, current - inactive_file)
free        = max(0, hard_limit - current)
available   = max(0, hard_limit - working_set)
```

The working set is a practical container metric, not a promise that every byte
of inactive file cache can be reclaimed immediately. Build sizing therefore
keeps an additional reserve.

Effective CPU count is the minimum positive constraint among host online CPU
count, process affinity, cpuset, floored CFS quota, and `RUNPOD_CPU_COUNT`.
Fractional CFS quotas are floored for concurrent compiler processes (with a
minimum of one); this is intentionally more conservative than a runtime thread
hint.

## SageAttention build parallelism

The cu128 and cu130 builders execute sequentially; GPU is the default backend
and sized CPU is an explicit capacity fallback. Within one builder, compiler
parallelism depends only on effective host-system CPU and RAM. GPU count,
compute capability, and VRAM never increase the suggested job count.

The defaults reflect observed SageAttention NVCC builds: a two-by-two parallel
build peaked near 30 GiB in a container with roughly a 7 GiB baseline, while a
serialized build peaked near 9.2 GiB total. The helper therefore assumes 8 GiB
per concurrent compiler job, reserves the larger of 4 GiB or 15% of the hard
limit, and caps the recommendation at four jobs:

```text
reserve        = max(4 GiB, ceil(hard_limit * 0.15))
usable         = max(0, available - reserve)
jobs_by_memory = max(1, floor(usable / 8 GiB))
suggested_jobs = max(1, min(jobs_by_memory, effective_cpus, 4))
```

With otherwise idle memory and ample CPU, the defaults produce:

| Hard limit | Available now | Suggested jobs |
| ---: | ---: | ---: |
| 16 GiB | 16 GiB | 1 |
| 32 GiB | 32 GiB | 3 |
| 32 GiB | 25 GiB (about 7 GiB occupied) | 2 |
| 48 GiB | 48 GiB | 4 |
| 64 GiB | 64 GiB | 4 |

Those are standalone helper calculations, not accepted release-builder sizes;
the release hard minimum remains 32 GiB. The reviewed release matrix currently
sets `default_max_jobs` to 2, and `scripts/build-wheel.sh` exports that value as
`POD_BUILD_MAX_JOBS` before taking its snapshot. Consequently, normal release
builds and the helper agree on a maximum of two jobs, with
`EXT_PARALLEL=1`; reaching three or four requires explicit review and new peak
evidence on a larger Pod.

Operators can tune the assumptions with `POD_BUILD_MEMORY_PER_JOB_MIB`,
`POD_BUILD_RESERVE_MIB`, and `POD_BUILD_MAX_JOBS`. An existing positive
`MAX_JOBS` is also accepted as the cap when `POD_BUILD_MAX_JOBS` is unset.
The explicit CLI flags `--memory-per-job-mib` and `--reserve-mib` take
precedence over environment values. Lowering these safeguards should be backed
by peak-RSS evidence from the same source, CUDA, compiler, and architecture
matrix.

## Scoped display shim

Build and install the shim with a C17 compiler, libc development headers,
linker, `libdl`, and standard `install`/`mktemp` utilities:

```sh
sh docker/resource-shim/install.sh
```

The installer compiles with `-Wall -Wextra -Werror` and installs:

```text
/usr/local/lib/libpodproc.so
/usr/local/libexec/podproc-wrapper
/usr/local/bin/free -> ../libexec/podproc-wrapper
/usr/local/bin/htop -> ../libexec/podproc-wrapper
/usr/local/bin/top  -> ../libexec/podproc-wrapper
```

A link is created only when the matching distro executable exists at
`/usr/bin` or `/bin`. `/usr/local/bin` must precede those directories in
`PATH`, as it does in the intended image. The original tool remains directly
available as `/usr/bin/free`, `/usr/bin/htop`, or `/usr/bin/top`.

The wrapper prepends the library to `LD_PRELOAD` for only the selected tool.
The library intercepts read-only opens of the exact path `/proc/meminfo`,
creates an in-memory replacement, and adjusts the fields used by procps and
htop: `MemTotal`, `MemFree`, `MemAvailable`, `Buffers`, `Cached`,
`SReclaimable`, `Shmem`, and swap totals. If cgroup membership, mount mapping,
hard limit, or usage cannot be resolved confidently, it opens the original
file unchanged.

Set `PODPROC_DISABLE=1` to escape the shim for a command:

```sh
PODPROC_DISABLE=1 free --bytes
```

For temporary testing of a nonstandard library path, set `PODPROC_LIBRARY` on
the wrapper. Never add this library to `/etc/ld.so.preload`, never export it
globally from the image entrypoint, and never preload it into Python, CUDA,
NVCC, ComfyUI, or arbitrary user commands.

### Shim limitations

- It is a display compatibility layer, not resource isolation. The kernel
  cgroup remains the only enforcer.
- It targets dynamically linked glibc tools that open the absolute
  `/proc/meminfo` path. Static binaries, secure-execution/setuid programs,
  direct syscalls, relative `openat` calls, and programs that read cgroups on
  their own can bypass it.
- It changes memory presentation only. Load average, uptime, process lists,
  per-process percentages, CPU topology, and similar host-visible data are not
  virtualized.
- Different `htop` versions have different native container-awareness. The
  wrapper is still safe because the replacement is derived from the same
  cgroup, but the authoritative JSON should be retained in build evidence.
- The shim cannot make arbitrary monitoring libraries such as psutil or a
  Jupyter server cgroup-aware. Feed the helper's values into those systems'
  supported settings instead.
- LXCFS can virtualize `/proc` more comprehensively, but it requires runtime
  mounts and privileges controlled by the pod platform. It cannot be enabled
  reliably by this image alone.

## Tests

The Python suite uses synthetic `/proc` and cgroup trees, so v1/v2, private
namespaces, inherited limits, CPU constraints, missing counters, and build
sizing are testable without Docker or root:

```sh
python -m unittest discover -s tests -p 'test_pod_resources.py' -v
```

Linux CI must also compile and smoke-test the shared library independently of
the large CUDA builder image:

```sh
sh docker/resource-shim/smoke-test.sh
```

For an already built image on a Docker host, an optional one-GiB end-to-end
check compares the helper's hard limit with `free --bytes`:

```sh
sh docker/resource-shim/docker-memory-test.sh IMAGE_TAG
```
