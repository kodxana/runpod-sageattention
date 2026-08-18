# Runpod orchestration

GitHub coordinates this factory, but NVCC compilation and CUDA execution happen
only inside short-lived Runpod Pods. Pull requests never receive Runpod
credentials and cannot create a paid Pod.

## Trust and approval model

Configure these GitHub environments before enabling the workflows:

- `runpod-paid` protects every GPU-backed build and GPU-test job. Add required
  reviewers and restrict deployment branches/tags as appropriate.
- `sageattention-release` is a second approval boundary around creation of the
  GitHub Release.

Store the following repository or `runpod-paid` environment secrets:

| Secret | Purpose |
|---|---|
| `RUNPOD_API_KEY` | Authenticates the Runpod REST API and checksum-pinned client. |
| `RUNPOD_SSH_PRIVATE_KEY` | Ephemeral workflow-side SSH identity. |

The private key is written with mode `0600`; its public half is derived with
`ssh-keygen -y` and passed to the Pod as `PUBLIC_KEY`, eliminating a mismatched
key-pair secret. It is used only for that job and removed with the hosted
runner. `tools/runpod_job.py` gives each invocation a
fresh `known_hosts` file and uses `StrictHostKeyChecking=accept-new`; it never
disables host-key checking globally.

The dispatch form also requires `confirm_paid_pods=true`. The orchestration
executable has its own mandatory `--allow-paid-pod` switch. Only `build.yml`
passes it, and those jobs are protected by `runpod-paid`.
`ci.yml` has no Runpod secrets, no `runpodctl`, and no paid-resource switch.

The default builders and all validators use GPU Pods created by the
checksum-pinned `runpodctl` v2.3.0 command. Every GPU request carries an exact
`gpuId`, an immutable image, an 80 GB-or-larger container disk for builders,
and an RFC3339 `--terminate-after` deadline. The same checked client handles
readiness and deletion.

`build_backend=CPU` remains an explicit capacity fallback. Because pinned
`runpodctl` does not expose sized CPU creation, that path uses the official
`POST /v1/pods` REST API with `cpuFlavorIds`, `vcpuCount`, and no persistent Pod
volume. It is not the release default and retains the in-Pod self-delete
watchdog because its platform termination field is not verified.

## Builder image bootstrap

Builder images contain the full CUDA toolkit, pinned PyTorch stack, SSH server,
the checksum-pinned Runpod client, a cgroup-aware resource helper, and an
optional entrypoint self-delete watchdog. They do not contain a prebuilt
SageAttention wheel. Building these Docker images on GitHub or a local Buildx
host is allowed; the costly NVCC wheel compilation happens on short-lived
Runpod GPU-backed builder Pods.

Authenticate to the chosen registry, then bootstrap both images:

```bash
export IMAGE_REPOSITORY=madiatorlabs/sageattention-wheel-builder
export IMAGE_TAG=bootstrap-$(git rev-parse --short=12 HEAD)

docker buildx bake -f docker/docker-bake.hcl \
  builder-cu128 builder-cu130 --push
```

The resulting tags are `${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu128` and
`${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu130`. Inspect each registry manifest and
record its digest:

```bash
docker buildx imagetools inspect \
  "${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu128"
docker buildx imagetools inspect \
  "${IMAGE_REPOSITORY}:${IMAGE_TAG}-cu130"
```

Pass `repository@sha256:<64 hex>` references to release workflows. Manual
development builds also default to digest enforcement, but an operator may
explicitly turn it off while iterating. Releases cannot turn it off. Runtime
ComfyUI image inputs are subject to the same release-time digest requirement.

## Ordered GPU candidates

The factory accepts ordered, comma-separated exact Runpod `gpuId` candidates.
The defaults below were conservatively reviewed against the
[official Runpod GPU type table](https://docs.runpod.io/references/gpu-types),
[NVIDIA's compute-capability table](https://developer.nvidia.com/cuda/gpus),
and the user's available-only catalog snapshot from 2026-08-18. They are not
fuzzy display names and availability at dispatch time is not guaranteed:

| Role | Required capability | Ordered exact `gpuId` candidates |
|---|---|---|
| Sequential builders | GPU hidden; capability unrestricted | `NVIDIA A100 80GB PCIe` → `NVIDIA A100-SXM4-80GB` → `NVIDIA H100 PCIe` → `NVIDIA H100 80GB HBM3` → `NVIDIA H100 NVL` → `NVIDIA H200` → `NVIDIA GeForce RTX 5090` → `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| Runtime representative | SM 8.0 | `NVIDIA A100 80GB PCIe` → `NVIDIA A100-SXM4-80GB` |
| Runtime representative | SM 8.6 | `NVIDIA A40` → `NVIDIA RTX A6000` → `NVIDIA GeForce RTX 3090` |
| Runtime representative | SM 8.9 | `NVIDIA L40S` → `NVIDIA RTX 6000 Ada Generation` → `NVIDIA GeForce RTX 4090` → `NVIDIA L4` |
| Runtime representative | SM 9.0 | `NVIDIA H100 PCIe` → `NVIDIA H100 80GB HBM3` → `NVIDIA H100 NVL` → `NVIDIA H200` |
| Runtime representative | SM 12.0 | `NVIDIA GeForce RTX 5090` → `NVIDIA RTX PRO 6000 Blackwell Server Edition` → `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` → `NVIDIA RTX PRO 4500 Blackwell Server Edition` → `NVIDIA RTX PRO 4500 Blackwell` |

Builder candidates may span architectures only because `build-wheel.sh` hides
the accelerator before compilation. Runtime fallback is stricter: every entry
in one validation list must have the exact same compute capability, and the job
must verify both the assigned `gpuId` and the actual CUDA capability before it
accepts evidence. Candidate exhaustion fails the job; it must not silently
substitute another architecture. MIG profiles and the RTX PRO 6000 Blackwell
Max-Q Workstation Edition are intentionally absent from these defaults.

Fallback is deliberately narrow. The orchestrator makes at most two ordered
placement rounds, with a five-second backoff before round two only when every
create in round one returned Runpod's explicit no-capacity response. A
pre-upload assignment mismatch is deleted before the next candidate is tried
and prevents another full round. Authentication, configuration, SSH, upload,
build, and test failures are never treated as capacity and are not retried
against another GPU. If deletion of a rejected Pod fails, fallback stops
immediately instead of risking a second billed Pod.

The SM120 server ID `NVIDIA RTX PRO 4500 Blackwell Server Edition` appears in
NVIDIA's compute-capability table and in the user's live Runpod catalog. The
static Runpod table currently lists the workstation/base ID
`NVIDIA RTX PRO 4500 Blackwell` but not that server ID. Keep the live catalog
spelling exact and recheck it before dispatch. `NVIDIA RTX A4500` is also left
out of the prefilled SM86 list because NVIDIA's current capability table omits
that SKU; add it only after a separate authoritative 8.6 confirmation.

The attached build GPU is deliberately hidden from the compiler, so its model
does not guarantee build resources. Every candidate still needs at least 4
effective vCPUs, 32 GiB system RAM, and an 80 GB container disk; 16 vCPUs and
64 GiB system RAM are recommended.

For a GPU-backed build, the selected exact candidate is preserved as
`artifact.build_evidence.selected_gpu_id` in `manifest.json`, and release
provenance checks it against the configured builder list. CPU-backed fallback
builds do not export a selected GPU and record this field as JSON `null`.

The same ordered values are recorded in `.env.example`. Confirm exact IDs and
current capacity in the live available-only catalog before a paid dispatch;
use `--include-unavailable` only to diagnose a missing type, not to claim it has
capacity. This behavior is defined by the
[official `runpodctl gpu` reference](https://docs.runpod.io/runpodctl/reference/runpodctl-gpu):

```bash
runpodctl gpu list -o json \
  | jq -r '.[] | [.displayName, .gpuId] | @tsv'
```

The workflow does not enumerate every commercial GPU SKU. B200 (SM100) and
B300 (SM103) are intentionally absent; NVIDIA lists them as 10.0 and 10.3
respectively, while SageAttention 2.2.0 has no SM10x
source/cubin or dispatcher path. The SM120 RTX 5090 result does not imply SM100
or SM103 compatibility. Adding either data-center GPU requires upstream kernel
and dispatcher support plus its own build target and representative validation.
Either could technically be allocated as the deliberately unused build
accelerator, but doing so would provide no runtime-compatibility evidence.

The two CUDA wheel variants build sequentially so only one billed builder and
one high-memory NVCC workload run at a time. Runtime validation remains a
reviewed ten-job matrix: two wheel variants multiplied by five capabilities.
`gpu_max_parallel` limits simultaneously billed validation Pods; the default is
two.

## Build and validation flow

`.github/workflows/build.yml` is both manually dispatchable and callable by the
release workflow.

1. A local planning job reads `matrix.json`. It refuses missing GPU ids,
   unexpected build/CUDA variants, or non-digest images when digest enforcement
   is enabled.
2. The two build variants run sequentially. Each starts a one-GPU Pod from its
   exact builder image using the ordered builder candidates, port 22, registry
   authentication when needed, at least 80 GB
   container disk, and an absolute `--terminate-after` deadline. The checked-out
   repository is archived without Git metadata or links and uploaded below
   `/work` on that explicitly sized ephemeral container filesystem; compilation
   does not require a persistent/network volume.
3. Runpod GPU placement attaches host-system CPU and RAM independently of GPU
   VRAM. Before upload, orchestration re-reads the assignment and requires the
   exact GPU type, at least 4 effective vCPUs, 32 GB system RAM, and the
   requested 80 GB container disk. `build-wheel.sh` then reads the assigned
   cgroup and requires 20 GiB currently free on both work and output
   filesystems. A 16-vCPU/64-GB system assignment is recommended. Passing the
   selected `gpuId` alone does not prove these host resources.
4. `build-wheel.sh` hides the attached accelerator with
   `CUDA_VISIBLE_DEVICES=""`, derives safe compiler concurrency without
   exceeding the matrix cap, and executes the pinned source build. The output,
   compiler work root, transfer archives, and `TMPDIR` all stay below `/work`,
   so disk preflight and compilation consume the same 80 GB-or-larger container
   filesystem. The workflow invokes it as:

   ```bash
   TMPDIR=/work/tmp \
   SAGEATTN_WORK_ROOT=/work/sageattention-wheel-builds \
   bash scripts/build-wheel.sh \
     --build-id BUILD_ID \
     --output-dir dist/BUILD_ID
   ```

   When `build_backend=CPU` is deliberately selected, the same command and
   preflight run on the sized CPU fallback. GPU remains the default backend.

5. `dist/BUILD_ID/` is downloaded and its `SHA256SUMS` is checked before GitHub
   stores `wheel-BUILD_ID` as a workflow artifact.
6. Each representative GPU job downloads that exact artifact, uploads it with
   the same source checkout into the matching immutable ComfyUI-compatible
   runtime image, verifies that Runpod reports the exact selected image digest,
   installs the wheel without dependencies, and runs static plus numerical
   validation:

   ```bash
   python3.12 scripts/validate-wheel.py \
     --matrix matrix.json \
     --build-id BUILD_ID \
     --wheel dist/BUILD_ID/WHEEL.whl \
     --manifest dist/BUILD_ID/manifest.json \
     --checksums dist/BUILD_ID/SHA256SUMS \
     --runtime \
     --expected-capability 8.9 \
     --runtime-report runtime-results/BUILD_ID/sm89.json
   ```

7. Each GPU produces a separate `runtime-BUILD_ID-smNN` evidence artifact.
   Missing, skipped, wrong-capability, non-finite, or numerically failing tests
   fail the job.

CUDA scheduler floors come from the matching build entry in `matrix.json`:
CUDA 12.8 for cu128 and CUDA 13.0 for cu130. GPU-backed builders and runtime
validators both forward them as `runpodctl pod create --min-cuda-version`,
preventing placement on a host whose driver cannot start the matching CUDA
container. The build still hides the accelerator after container startup.

## Release ordering

`.github/workflows/release.yml` is manual and accepts an existing Git tag. A
free preflight job fetches that exact tag, peels annotated tags to their commit,
and emits the full 40-character commit SHA before any paid Pod can start. The
complete build workflow receives that SHA rather than the tag name, and its
plan, GPU-builder, and GPU-test checkouts remain pinned to the resolved commit.
Only after both sequential GPU-backed builds and all ten representative-GPU
jobs pass does the
`publish` job become eligible for the separate `sageattention-release`
approval.

The publish job then:

1. downloads both wheel payloads and all GPU reports from the same workflow
   run;
2. rechecks every `SHA256SUMS` file;
3. binds every report to its build id, wheel SHA-256, expected and actual
   capability, exact PyTorch/CUDA tuple, runtime image digest, and passing
   numerical thresholds, requires the exact canonical output shape and dtype,
   and requires the exact dispatch/kernel implementation and causal/non-causal
   result cross-product declared for that capability;
4. verifies each per-build manifest recorded the selected builder image digest;
5. uses `scripts/merge-manifests.py` to create the selector-compatible merged
   `manifest.json` and combined `SHA256SUMS`;
6. stages uniquely named per-variant manifests, `release-inputs.json` binding
   the resolved source commit plus the selected builder/runtime refs and GPU
   ids, and a compressed validation evidence bundle;
7. queries the remote tag again immediately before publication and refuses to
   publish if its current peeled commit differs from the preflight SHA; and
8. creates the GitHub Release with `gh release create --verify-tag`.

There is no pre-test upload to a public release, mutable package index, or
stable wheel URL. Failure in any build, transfer, or GPU test prevents the
release job from running.

## Pod lifetime and cleanup

Every default GPU builder and validator Pod combines independent lifetime
controls:

- the Python orchestrator shares one monotonic hard deadline across Pod
  creation, SSH readiness, upload, commands, and artifact download;
- Pod creation includes an absolute RFC3339 `--terminate-after` datetime set to
  the hard timeout plus cleanup grace; and
- after a Pod id exists, deletion always runs from a `finally` path and is
  retried three times.

A deletion failure is a workflow failure rather than a warning. The absolute
platform deadline remains the final protection against runner loss,
cancellation, an image-start failure, or a workflow-side network partition.

The explicit CPU fallback has the same orchestrator deadline and mandatory
`finally` deletion, but its REST provisioning path has no verified platform
deadline. It therefore must pass `RUNPOD_SELF_TERMINATE_SECONDS` to the builder
entrypoint, which arms the reviewed in-Pod deletion watchdog. That watchdog is
specific to the fallback and is never a replacement for orchestrator cleanup.

Cancellation handling must still verify deletion in the Runpod console. Never
assume an in-Pod watchdog ran: it cannot arm before the builder entrypoint
starts.

When a remote command fails, the orchestrator attempts to retrieve any requested
diagnostic artifact before deletion, provided the hard deadline still has time.
The original command error, artifact error, and cleanup error are reported
together.

## Direct operator use

The same tool can run a reviewed command outside GitHub. This example uses the
first two ordered GPU-backed builder candidates and retrieves one build
directory; repeat `--gpu-id` once per exact fallback in priority order:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_SSH_PUBLIC_KEY='ssh-ed25519 AAAA... operator'

python3.12 tools/runpod_job.py \
  --allow-paid-pod \
  --mode gpu \
  --gpu-workload build \
  --gpu-id 'NVIDIA A100 80GB PCIe' \
  --gpu-id 'NVIDIA A100-SXM4-80GB' \
  --gpu-min-vcpu-count 16 \
  --gpu-min-memory-gb 32 \
  --min-cuda-version 12.8 \
  --image 'registry/repository@sha256:...' \
  --name sageattention-manual-build \
  --container-disk-gb 80 \
  --ssh-key /secure/path/id_ed25519 \
  --repo . \
  --remote-dir /work/sageattention-factory \
  --timeout-seconds 14400 \
  --terminate-grace-seconds 900 \
  --command 'install -d -m 0700 /work/tmp /work/sageattention-wheel-builds; TMPDIR=/work/tmp SAGEATTN_WORK_ROOT=/work/sageattention-wheel-builds bash scripts/build-wheel.sh --build-id cp312-torch2.10.0-cu128 --output-dir dist/cp312-torch2.10.0-cu128' \
  --artifact dist/cp312-torch2.10.0-cu128 \
  --artifact-output ./artifacts
```

Run the cu130 command only after cu128 finishes, using its matching builder
image and `--min-cuda-version 13.0`. The tool tries only the explicitly ordered
exact IDs and accepts only the candidate actually reported by Runpod. Builders
and runtime validators both pass the CUDA scheduler floor from their matrix
entry; runtime candidates must additionally stay within that job's one exact
compute capability.

## Operational cautions

- The minimum/default 80 GB container disk is intentional: CUDA toolkits, PyTorch,
  source trees, native objects, and fat wheels need substantial temporary
  space. Builder work stays below `/work`; `/workspace` is retained for GPU
  runtime images whose ComfyUI layout expects it.
- The accelerator is attached but unused during compilation. Select a GPU offer
  with adequate host-system resources: 4 effective vCPUs and 32 GiB system RAM
  are hard minimums, while 16 vCPUs and 64 GiB system RAM are recommended.
- Every builder must expose finite cgroup limits and satisfy the resource
  preflight. Host-wide `/proc/meminfo`, GPU VRAM, and GPU utilization are not
  used to choose compilation parallelism.
- Do not add `LD_PRELOAD` to workflows. Resource-reporting wrappers are scoped
  inside the builder image.
- Keep CUDA 12.8 and CUDA 13.0 wheel assets distinguishable by their downstream
  version and manifest. Standard Python wheel tags do not encode CUDA or
  PyTorch compatibility.
- If a run is cancelled, verify Pod deletion in the Runpod console. The
  platform deadline on GPU Pods and in-Pod watchdog on the CPU fallback are
  backstops, not reasons to ignore a failed cleanup step.
