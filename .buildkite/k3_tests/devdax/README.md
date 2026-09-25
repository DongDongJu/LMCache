# DevDAX integration in QEMU

CPU-only tests exercise production MP L1/L2 against three independent 256 MiB
volatile CXL devices. The guest builds the tested source with `NO_GPU_EXT=1`
and verifies the compiled native extension. Host DAX devices are never used.

## Selection

| Variable | Default | Values |
| --- | --- | --- |
| `LMCACHE_DEVDAX_QEMU` | `auto` | `on`, `off`, `auto` |
| `LMCACHE_DEVDAX_SUITE` | `both` | `l1`, `l2`, `both` |
| `LMCACHE_DEVDAX_QEMU_ACCEL` | `kvm` | `kvm`, `tcg`; no automatic fallback |

Invalid values fail. `off` disables only QEMU; `on` forces it. In `auto`, scheduled
builds, `force-ci`, relevant changes and unknown diffs run coverage. README-only
PRs skip. `ci_selection.py` contains the conservative dependency allowlist,
including the CPU installer under `.github/`. PRs compare their full merge-base
against the Buildkite base branch (default `dev`); pushes need the full webhook
before SHA in `LMCACHE_DEVDAX_BEFORE_SHA`, otherwise they run. Deleted/renamed paths
and shallow histories are covered. The independent unit bootstrap command avoids
the generic trivial-file filter; there is no second `if_changed` gate.

## Nightly CI image

The existing `nightly_build.yml` schedule calls `build_devdax_image.yml`, which
builds the environment, runs both real-device suites inside its container, and
then publishes `ghcr.io/lmcache/lmcache-devdax-ci:nightly`. Every successful build
also has a `build-<run-id>-<attempt>` tag; only builds from `dev` update `nightly`.
The workflow supports manual dispatch and uses `GITHUB_TOKEN` with package-write
permission. Set the GHCR package visibility to public for anonymous CI pulls, or
configure Docker registry authentication on the Buildkite agent.

Buildkite jobs pull this image automatically. The agent needs Docker and KVM;
it no longer needs a locally built QEMU, kernel or guest disk. To use it locally:

```bash
LMCACHE_DEVDAX_CONTAINER_IMAGE=ghcr.io/lmcache/lmcache-devdax-ci:nightly \
bash .buildkite/k3_tests/devdax/run.sh
```

Override that variable with `ghcr.io/...@sha256:...` for repeatable runs. The image
contains the environment only: the current checkout is mounted, copied into a
fresh guest overlay, and built there. The container gets `/dev/kvm` explicitly,
without privileged mode; output files retain the calling user's ownership.
`container-image.json` records the pulled digest alongside the VM reports.

## Building the environment locally

Use an x86-64 Linux host with 4 cores, 16 GiB free RAM, 30 GiB free disk, Python
3.12+, OpenSSH, `cloud-localds`, and working KVM permissions. VM hosts also need
nested virtualization. Build prerequisites: `build-essential`, `flex`, `bison`,
`bc`, `libelf-dev`, `libssl-dev`, `libglib2.0-dev`, `libpixman-1-dev`, `libslirp-dev`,
`ninja-build`, and Python venv support.

Download the QEMU, kernel and cloud-image URLs from `image/manifest.json` into
`/var/cache/lmcache/`. The recipes verify their pinned source checksums:

```bash
bash .buildkite/k3_tests/devdax/image/build-qemu.sh \
  /var/cache/lmcache/qemu-11.1.1.tar.xz /var/cache/lmcache/qemu
export PATH="/var/cache/lmcache/qemu/bin:$PATH"
bash .buildkite/k3_tests/devdax/image/build-kernel.sh \
  /var/cache/lmcache/linux-6.12.50.tar.xz /var/cache/lmcache/kernel
bash .buildkite/k3_tests/devdax/run.sh --prepare-image \
  --base /var/cache/lmcache/noble.img --image /var/cache/lmcache/base.qcow2

export LMCACHE_DEVDAX_IMAGE=/var/cache/lmcache/base.qcow2
export LMCACHE_DEVDAX_KERNEL=/var/cache/lmcache/kernel/bzImage
LMCACHE_DEVDAX_SUITE=both bash .buildkite/k3_tests/devdax/run.sh
```

QEMU 11.1.1 fixes noninterleaved CXL mappings under KVM; older versions can fault
on vectorized copies. The test kernel enables `CXL_REGION_INVALIDATION_TEST`
for virtual CPUs and `FS_DAX` for the inode DAX flag, and disables `DEV_DAX_KMEM`
to prevent conversion to system RAM. This is an emulation test kernel.

Image preparation installs CPU PyTorch 2.10.0 and the repository's existing
build/common/test requirements. It records `pip freeze` and the OS package
inventory. Dependency resolution may change on rebuild: retain the prepared
image and its `.sha256` file to reproduce an environment. The builder refuses
to overwrite images. Every run verifies the image/kernel checksums, creates a
private overlay/key/socket and a QEMU-assigned SSH port, and copies the tested
checkout after the PR-base merge. Commit/tree and source-archive hashes are saved.
KVM has a 30-minute overall deadline; explicit TCG has 120 minutes. Override the
300/1200-second boot deadline with `LMCACHE_DEVDAX_BOOT_TIMEOUT`.

## Tests and reports

`guest_test.py` declares exact node IDs, builds LMCache and runs L1/L2 serially.
Both selects four L1 and five L2 cases, including a combined test on separate
devices; each individual suite selects four. An L1 failure still produces an L2
result. Empty suites, unexpected collection, skips/xfails, missing reports and
failures fail the job. Payloads and layouts are checked in full.

The guest discovers CXL regions and validates character-device identity, driver,
capacity/alignment and an mmap byte round trip. It exports strict mode, both
opt-in flags, discovered L1/L2 paths and `LMCACHE_TEST_DEVDAX_MANIFEST`. Normal
opt-in runs use isolated temporary files; real-device fixtures reject aliases
and parallel execution. Never supply devices containing useful data.

Artifacts under `artifacts/devdax-qemu/run-<id>/` include source/image identity,
console/kernel logs, topology, device manifest, package inventory, build/pytest
logs, collection JSON, JUnit and `summary.json`. Summary states distinguish
`not selected`, `passed`, `failed` and `infrastructure failure`. The uploader
saves `selection.json` and labels skipped coverage. Cleanup stops only this VM.

```bash
python -m pytest --confcutdir=tests/ci tests/ci/test_devdax_ci_selection.py
RUN_DEVDAX_L1_INTEGRATION=1 RUN_DAX_L2_INTEGRATION=1 OMP_NUM_THREADS=2 \
python -m pytest -q tests/v1/distributed/test_devdax_l1_reconfigure_integration.py \
  tests/v1/distributed/test_dax_l2_integration.py
RUN_DEVDAX_QEMU_HOST_TESTS=1 python -m pytest --confcutdir=tests/ci \
  tests/ci/test_devdax_qemu_runner.py
```

The last command requires the prepared image/kernel environment above. It checks
real VM payload corruption, missing devices, timeouts and cancellation isolation.

## Buildkite activation

A project administrator must provision CPU queue `devdax-qemu` with Docker and
working KVM access. Publish the first GHCR image before enabling the queue. Paste
`../unit/buildkite-pipeline.yml` into the active K3 unit Steps editor; Git changes
alone do not update it. Preserve fork/runner access policy and allow eligible PR
webhooks through, including docs-only explicit `on` builds.

Use `buildkite-pipeline.yml` for manual Buildkite runs against branch `dev`.
The GitHub nightly image workflow already runs both suites before publishing. Verify L1-only, L2-only, combined, unrelated-path and explicit-off builds;
retain their URLs. Do not add this uploader to other test pipelines.

These functional checks do not establish GPU registration/DMA, physical CXL
performance/coherence, persistence, or physical hot removal.
