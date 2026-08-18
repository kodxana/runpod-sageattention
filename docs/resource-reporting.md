# Pod-aware resource reporting

Linux containers normally share the host's `/proc/meminfo`. As a result,
`free`, `top`, and some `htop` builds can display host RAM even when the pod has
a much smaller cgroup limit. The kernel still enforces the cgroup limit; the
misleading display does not grant the pod extra memory.

SageAttention builders are GPU-backed, but the build hides the accelerator with
`CUDA_VISIBLE_DEVICES=""`. Resource policy therefore concerns the Pod's
host-system vCPU, system RAM, and container disk—not GPU utilization or VRAM.
The attached GPU type is a scheduling choice and cannot substitute for assigned
system RAM. Ordered builder fallbacks may therefore span GPU architectures, but
every selected candidate is still subject to the same API-assignment, cgroup,
CPU, and disk checks; candidate ordering must never be based on VRAM as a proxy
for system RAM.

This repository handles the problem in two layers:

1. `tools/pod_resources.py` is the authoritative source for build automation.
   The build prefers this uploaded repository copy so the policy can be fixed
   without rebuilding an older builder image; `/usr/local/bin/pod-resources`
   remains its installed fallback. It resolves cgroups directly and reports
   machine-readable memory, CPU, and safe compiler parallelism data.
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

The JSON schema is versioned. Schema version 2 has this shape (values below are
illustrative):

```json
{
  "schema_version": 2,
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
    "swap_current_bytes": 0,
    "capacity_bytes": 34359738368,
    "capacity_source": "cgroup-v2:/sys/fs/cgroup/pod",
    "capacity_is_hard_limit": true,
    "assigned_capacity_bytes": 68719476736,
    "usage_source": "cgroup-v2:/sys/fs/cgroup/pod",
    "usage_current_bytes": 7516192768,
    "usage_trustworthy": true,
    "usage_peak_eligible": true,
    "usage_scope": "cgroup-capacity"
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
    "max_jobs_cap": 4,
    "forced_single_job": false
  },
  "warnings": []
}
```

Automation should require all of the following before starting a SageAttention
build:

- `schema_version == 2`
- `memory.capacity_bytes > 0`
- `memory.assigned_capacity_bytes > 0`, authorized by the exact receipt and
  command-scoped environment pair for every Runpod release build
- capacity comes either from a consistent finite cgroup hard limit or from
  `memory.capacity_source == "runpod-api-assignment"`
- either `memory.peak_evidence_mode == "cgroup"`, with cgroup version 1 or 2
  and a peak-eligible resolved membership source, or the restricted
  `process-group-rss` fallback described below
- `build.suggested_jobs >= 1`

`RUNPOD_ASSIGNED_MEMORY_BYTES` is reserved for the orchestrator: it must be
copied from the verified Runpod Pod API response, never entered as an operator
guess. The helper rejects non-canonical, non-positive, and over-host values. It
records the assignment separately and never labels it a kernel-enforced hard
limit. If a real cgroup hard limit is no larger than the assignment, that
smaller limit wins. Raw `limit_bytes`, `limit_source`, and `limited` fields are
preserved so the distinction remains auditable.

For assignment-backed capacity, memory usage comes from the process's resolved
cgroup membership. An explicitly scoped leaf is usable for headroom sizing. A
`/` membership on a `/` mount can also be a private cgroup namespace, as on
some Runpod Pods, but it is not distinguishable from a host controller root
from inside the container. When its readable counter does not exceed the
verified assignment it is eligible for conservative headroom and peak checks;
inactive cache receives no reclaimability credit. The helper marks the scope
`ambiguous-cgroup-root` and forces both `MAX_JOBS=1` and `EXT_PARALLEL=1`. A
malformed or over-assignment counter is not accepted, and the build stops
before compilation.

When the cgroup membership or its current counter is genuinely absent, the
helper can instead select `peak_evidence_mode=process-group-rss`. This is not a
general escape hatch: the selected capacity must be the exact receipt-backed
Runpod assignment, it must meet the matrix's recommended 64 GiB capacity, both
compiler parallelism values are forced to one, and neither low-resource nor
unsafe-parallelism overrides may be active. A readable counter that conflicts
with the assignment never qualifies for this fallback.

In this mode a Python supervisor launches the exact wheel command with
`start_new_session=True` and samples only that dedicated Linux process group
every 100 ms. It checkpoints incomplete evidence atomically every five seconds,
then records the PGID/leader binding, positive endpoints and peak, lifecycle
timestamps, and an observed native compiler executable. It requires at least
two samples and checks that the peak plus the normal build reserve fits the
verified assignment. HUP, INT, and TERM received by either the build shell or
supervisor are forwarded to the isolated group; cleanup escalates to KILL after
ten seconds and verifies that no descendant remains.

The resulting evidence is labeled `process-group-rss`, not cgroup peak
accounting: it excludes unrelated Pod processes and most filesystem cache, may
conservatively count shared resident pages more than once, and is not
kernel-enforced or whole-Pod enforcement. Promotion validates those limitations,
the supervisor lifecycle, and the serialized policy explicitly.

Default GPU-backed release policy requires an assignment with at least 4
assigned vCPUs, 32 GB system RAM, and an 80 GB container disk before source
upload; 16 vCPUs and 64 GB system RAM are recommended. The backend-neutral
helper and build script then independently require at least 4 effective vCPUs,
a verified API assignment for every release build, and at least 32 GiB
established by the smaller of that assignment and a finite cgroup limit. They
also require one safe compiler job, acceptable typed peak evidence, and 20 GiB
currently free on both work and output filesystems. CPU, minimum capacity, and
actual disk checks cannot be bypassed with `ALLOW_LOW_RESOURCES`; disk is
checked by `build-wheel.sh`, not by the resource-reporting helper.

The shell form exports `POD_MEMORY_LIMIT_BYTES`, `POD_MEMORY_CAPACITY_BYTES`,
`POD_MEMORY_CAPACITY_SOURCE`, `POD_MEMORY_USAGE_SOURCE`,
`POD_MEMORY_USAGE_PEAK_ELIGIBLE`,
`POD_MEMORY_PEAK_EVIDENCE_MODE`,
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

Memory values use these definitions, where capacity is the smaller applicable
cgroup hard limit or verified Runpod assignment:

```text
working_set = max(0, current - inactive_file)
free        = max(0, capacity - current)
available   = max(0, capacity - working_set)
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
per concurrent compiler job, reserves the larger of 4 GiB or 15% of effective
capacity, and caps the recommendation at four jobs:

```text
reserve        = max(4 GiB, ceil(capacity * 0.15))
usable         = max(0, available - reserve)
jobs_by_memory = max(1, floor(usable / 8 GiB))
suggested_jobs = max(1, min(jobs_by_memory, effective_cpus, 4))
```

With otherwise idle memory and ample CPU, the defaults produce:

| Capacity | Available now | Suggested jobs |
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

Even when a local diagnostic environment has a finite cgroup limit, invoking
the release build entrypoint requires both `RUNPOD_ASSIGNED_MEMORY_BYTES` and a
matching fixed-path receipt. This prevents a locally acceptable cgroup-only
build from compiling for hours and then failing promotion's resource-evidence
gate.

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

When Runpod reports an assigned memory capacity but exposes an unlimited
memory cgroup, the orchestrator also writes the verified decimal-byte value to
`/run/sageattention/verified-memory-bytes-v1`. The receipt is an exact positive
decimal followed by one newline, owned by `root:root`, mode `0444`, and written
atomically only after the REST assignment has passed the image, GPU/CPU, disk,
and memory checks. The shim rejects symlinks or multiply-linked receipts, any
other owner or mode, malformed content, missing cgroup usage, and a current
usage greater than the selected capacity. In those cases it exposes the real
`/proc/meminfo` unchanged.

`RUNPOD_ASSIGNED_MEMORY_BYTES` supplies the same presentation ceiling to
commands descended from the orchestrated build shell. It is command-scoped, so
the receipt is what makes later `free`, `top`, and `htop` sessions consistent.
When more than one ceiling is available, the shim displays the smallest of the
host total, a real finite cgroup limit, the verified receipt, and the
command-scoped value. Assignment-only presentation reports no swap because the
Runpod RAM assignment does not establish a swap allowance.

Neither input turns a scheduler assignment into kernel isolation. The shim uses
them only for presentation. For build authorization, the uploaded
`pod-resources` helper requires the command-scoped value and the fixed receipt
to be present, independently well formed, and exactly equal; an environment
value alone is never authoritative. It then applies cgroup/assignment minimum,
usage, peak, CPU, disk, and parallelism policy, and the result is retained in
build evidence. Images built before the receipt consumer was added must be
rebuilt and pinned to a new digest before their interactive wrappers can use
it; the uploaded helper allows existing builder images to use the build policy
without an image rebuild.

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
