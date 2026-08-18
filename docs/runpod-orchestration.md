# Runpod orchestration

GitHub coordinates this factory, but NVCC compilation and CUDA execution happen
only inside short-lived Runpod Pods. Pull requests never receive Runpod
credentials and cannot create a paid Pod.

## Trust and approval model

Configure these GitHub environments before enabling the workflows:

- `runpod-paid` protects every CPU-build and GPU-test job. Add required
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

The orchestration intentionally uses two provisioning APIs. GPU Pods use the
checksum-pinned `runpodctl` v2.3.0 command. That version's CPU creation path
does not expose `cpuFlavorIds` or `vcpuCount` and silently omits registry auth
and `--terminate-after`, so CPU build Pods are created through the official
`POST /v1/pods` REST endpoint instead. The same `RUNPOD_API_KEY` authenticates
both paths; `runpodctl` remains the checked lifecycle client for readiness and
deletion.

## Builder image bootstrap

Builder images contain the full CUDA toolkit, pinned PyTorch stack, SSH server,
the checksum-pinned Runpod client, a cgroup-aware resource helper, and an
entrypoint self-delete watchdog. They do not contain a prebuilt SageAttention
wheel. Building these Docker images on GitHub or a local Buildx host is allowed;
the costly NVCC wheel compilation still happens on Runpod CPU Pods.

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

## Exact GPU representatives

The factory tests one explicitly configured Runpod GPU type for every required
compute capability in `matrix.json`:

| Capability | Example representative |
|---|---|
| SM 8.0 | A100 |
| SM 8.6 | A40 or RTX 3090 |
| SM 8.9 | L40S or RTX 4090 |
| SM 9.0 | H100 or H200 |
| SM 12.0 | RTX 5090 |

The workflow inputs are exact `gpuId` values, not fuzzy names. Resolve them at
dispatch time from the live Runpod catalog, for example:

```bash
runpodctl gpu list --include-unavailable -o json \
  | jq -r '.[] | [.displayName, .gpuId] | @tsv'
```

The workflow does not enumerate every commercial GPU SKU. It forms a reviewed
ten-job matrix: two wheel variants multiplied by the five required compute
capabilities. `gpu_max_parallel` limits simultaneously billed validation Pods;
the default is two.

## Build and validation flow

`.github/workflows/build.yml` is both manually dispatchable and callable by the
release workflow.

1. A local planning job reads `matrix.json`. It refuses missing GPU ids,
   unexpected build/CUDA variants, or non-digest images when digest enforcement
   is enabled.
2. Each build variant starts a CPU Pod from its exact builder image using the
   official REST API. The request explicitly sets `cpuFlavorIds`, `vcpuCount`,
   registry auth, port 22, and `supportPublicIp=true` on Community Cloud. The
   defaults are `cpu3g` with 16 vCPUs, which targets the recommended 64 GB
   configuration; assignment must still report at least 32 GB. Before upload,
   the orchestrator re-reads the Pod and rejects any image, CPU-flavor, vCPU,
   memory, container-disk, or zero-volume mismatch. The checked-out repository
   is then archived without Git
   metadata or links and uploaded to `/work/sageattention-factory`, on the
   explicitly sized ephemeral container disk. No paid persistent/network
   volume is provisioned for compilation: the CPU REST request explicitly sets
   `volumeInGb: 0`, and the orchestrator rejects a CPU checkout outside a
   directory below `/work`.
3. `build-wheel.sh` performs the authoritative resource preflight and derives
   safe concurrency without exceeding the cap in `matrix.json`, then executes
   the pinned source build. The output, compiler work root, transfer archives,
   and `TMPDIR` are all under `/work`, so the disk preflight and actual build
   consume the same 80 GB-or-larger container filesystem. The workflow invokes
   it as:

   ```bash
   TMPDIR=/work/tmp \
   SAGEATTN_WORK_ROOT=/work/sageattention-wheel-builds \
   bash scripts/build-wheel.sh \
     --build-id BUILD_ID \
     --output-dir dist/BUILD_ID
   ```

4. `dist/BUILD_ID/` is downloaded and its `SHA256SUMS` is checked before GitHub
   stores `wheel-BUILD_ID` as a workflow artifact.
5. Each representative GPU job downloads that exact artifact, uploads it with
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

6. Each GPU produces a separate `runtime-BUILD_ID-smNN` evidence artifact.
   Missing, skipped, wrong-capability, non-finite, or numerically failing tests
   fail the job.

CUDA scheduler floors come from the matching build entry in `matrix.json`:
CUDA 12.8 for the cu128 runtime and CUDA 13.0 for cu130. They are forwarded as
`runpodctl pod create --min-cuda-version`, preventing placement on an
incompatible host driver.

## Release ordering

`.github/workflows/release.yml` is manual and accepts an existing Git tag. A
free preflight job fetches that exact tag, peels annotated tags to their commit,
and emits the full 40-character commit SHA before any paid Pod can start. The
complete build workflow receives that SHA rather than the tag name, and its
plan, CPU-build, and GPU-test checkouts remain pinned to the resolved commit.
Only after both CPU builds and all ten representative-GPU jobs pass does the
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

Every Pod combines the orchestrator bound with a mode-specific backstop:

- the Python orchestrator shares one monotonic hard deadline across Pod
  creation, SSH readiness, upload, commands, and artifact download;
- GPU creation includes the RFC3339 `--terminate-after` datetime required by
  the pinned `runpodctl` v2.3.0 binary; and
- CPU creation passes `RUNPOD_SELF_TERMINATE_SECONDS` equal to the hard timeout
  plus cleanup grace. The builder entrypoint validates the Pod-scoped
  `RUNPOD_API_KEY` and `RUNPOD_POD_ID`, then starts an independent watchdog that
  retries self-deletion through Runpod after that interval. This compensates
  for v2.3.0 dropping its termination flag on the CPU REST path.

After a Pod id exists, deletion runs from a `finally` path and is retried three
times. A deletion failure is a workflow failure rather than a warning. The GPU
platform deadline or CPU in-Pod watchdog remains the final protection against
runner loss, cancellation, or a workflow-side network partition.

The CPU watchdog can arm only after the builder entrypoint starts. An image-pull
or pre-entrypoint startup failure therefore still requires checking and deleting
the Pod in the Runpod console; the current CPU REST path has no verified
server-side termination field.

When a remote command fails, the orchestrator attempts to retrieve any requested
diagnostic artifact before deletion, provided the hard deadline still has time.
The original command error, artifact error, and cleanup error are reported
together.

## Direct operator use

The same tool can run a reviewed command outside GitHub. This example uses a CPU
builder and retrieves one build directory:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_SSH_PUBLIC_KEY='ssh-ed25519 AAAA... operator'

python3.12 tools/runpod_job.py \
  --allow-paid-pod \
  --mode cpu \
  --image 'registry/repository@sha256:...' \
  --name sageattention-manual-build \
  --container-disk-gb 80 \
  --cpu-flavor-ids cpu3g \
  --cpu-vcpu-count 16 \
  --cpu-min-memory-gb 32 \
  --ssh-key /secure/path/id_ed25519 \
  --repo . \
  --remote-dir /work/sageattention-factory \
  --timeout-seconds 14400 \
  --command 'install -d -m 0700 /work/tmp /work/sageattention-wheel-builds; TMPDIR=/work/tmp SAGEATTN_WORK_ROOT=/work/sageattention-wheel-builds bash scripts/build-wheel.sh --build-id cp312-torch2.10.0-cu128 --output-dir dist/cp312-torch2.10.0-cu128' \
  --artifact dist/cp312-torch2.10.0-cu128 \
  --artifact-output ./artifacts
```

GPU mode additionally requires `--gpu-id` and should specify the matrix-derived
`--min-cuda-version`. The tool never discovers or substitutes a cheaper GPU;
the exact requested id is sent to Runpod.

## Operational cautions

- The minimum/default 80 GB container disk is intentional: CUDA toolkits, PyTorch,
  source trees, native objects, and fat wheels need substantial temporary
  space. CPU factory work stays below `/work`; `/workspace` is retained only
  for GPU runtime images whose ComfyUI layout expects it.
- `cpu3g` provides 4 GB RAM per vCPU. The 16-vCPU default therefore targets
  64 GB; 8 vCPUs is only the 32 GB hard minimum and leaves little headroom over
  the measured 29.65 GB peak.
- A CPU Pod must expose finite cgroup limits and satisfy the resource preflight.
  Host-wide `/proc/meminfo` is not used to choose compilation parallelism.
- Do not add `LD_PRELOAD` to workflows. Resource-reporting wrappers are scoped
  inside the builder image.
- Keep CUDA 12.8 and CUDA 13.0 wheel assets distinguishable by their downstream
  version and manifest. Standard Python wheel tags do not encode CUDA or
  PyTorch compatibility.
- If a run is cancelled, verify Pod deletion in the Runpod console. The GPU
  platform deadline or CPU watchdog is a backstop, not a reason to ignore a
  failed cleanup step.
