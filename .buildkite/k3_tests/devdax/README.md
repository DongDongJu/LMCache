# DevDAX CPU integration in QEMU

This gate runs the production MP L1 allocator and L2 adapter against three
independent, volatile, 256 MiB CXL Type-3 devices inside a disposable Linux guest.
It builds the tested checkout with CPU PyTorch and `NO_GPU_EXT=1`, including the
compiled common native extension. No model, vLLM, GPU, or host DAX device is used.

## Controls

| Environment variable | Default | Accepted values / purpose |
| --- | --- | --- |
| `LMCACHE_DEVDAX_QEMU` | `auto` | `on` forces selection, `off` disables only this job |
| `LMCACHE_DEVDAX_SUITE` | `both` | `l1`, `l2`, `both`; both includes the combined case |
| `LMCACHE_DEVDAX_QEMU_ACCEL` | `kvm` | `kvm`, or explicit slower `tcg`; no fallback |
| `LMCACHE_DEVDAX_KERNEL` | `/opt/lmcache-devdax/kernel/bzImage` | Built test kernel |
| `LMCACHE_DEVDAX_IMAGE` | `/opt/lmcache-devdax/base.qcow2` | Prepared guest image |
| `LMCACHE_DEVDAX_BEFORE_SHA` | unset | Full push webhook before SHA, when available |
| `LMCACHE_DEVDAX_BOOT_TIMEOUT` | 300 / 1200 | SSH boot deadline in seconds for KVM / TCG |

Enums are validated before selection. `off` wins over all triggers. In `auto`,
scheduled builds, `force-ci`, and relevant changes select the job. Unknown diffs
also select it. A root README-only PR skips; `on` overrides that skip.

The allowlist is `PREFIXES` and `FILES` in `ci_selection.py`: allocator, DAX, MP,
distributed, platform, native, build/dependency, fixture and CI infrastructure
paths. It intentionally includes the CPU installer under `.github/`. PRs compare
their full merge-base against the Buildkite base branch (default `dev`). Pushes
require a reliable full before SHA; without one they run. NUL-delimited diffs
include deletions and both sides of renames. Unresolved shallow history runs.
There is no second `if_changed` gate.

## Runner and image

Use a self-hosted x86-64 Linux agent with 16 GiB free RAM, 30 GiB free disk,
4 CPU cores, functional KVM, OpenSSH client, Python 3.12+, `cloud-localds`, and
QEMU 11.1.1 from `image/manifest.json`. The service account must be
able to open `/dev/kvm`. A VM host also needs nested virtualization enabled.
The proposed `devdax-qemu` queue is CPU-only and must be provisioned by an admin.

QEMU 11.1.1 includes the upstream noninterleaved CXL memory mapping fix;
older QEMU versions can fault on vectorized CPU accesses under KVM. Install
`libglib2.0-dev`, `libpixman-1-dev`, `libslirp-dev`, `ninja-build` and Python venv
support, then build the signed release pinned in the manifest:

```bash
bash .buildkite/k3_tests/devdax/image/build-qemu.sh \
  /var/cache/lmcache/qemu-11.1.1.tar.xz /var/cache/lmcache/qemu
export PATH="/var/cache/lmcache/qemu/bin:$PATH"
```

The source checksum was verified against the release signature with fingerprint
`CEACC9E15534EBABB82D3FA03353C9CEF108B584`. The guest uses `-cpu host` with KVM,
so ordinary PyTorch/libc CPU dispatch and vectorized copies remain enabled.

Build the test kernel with `gcc`, `make`, `flex`, `bison`, `bc`, `libelf-dev`
and `libssl-dev` installed. Download the kernel source URL from the manifest:

```bash
bash .buildkite/k3_tests/devdax/image/build-kernel.sh \
  /var/cache/lmcache/linux-6.12.50.tar.xz /var/cache/lmcache/kernel
```

The recipe checks the kernel tarball checksum, starts from its pinned x86-64
config and builds the boot/CXL/DAX drivers in. It enables
`CONFIG_CXL_REGION_INVALIDATION_TEST` because a virtual CPU cannot provide the
bare-metal cache invalidation required during region creation. This documented
[kernel test option](https://code.googlesource.com/linux/torvalds/linux/+/0bdf0621f89f87858ca26344378188eff194eddd/drivers/cxl/core/region.c)
is for emulation only; the production DAX mappings and copies remain real.
The system-RAM (`DEV_DAX_KMEM`) driver is disabled so new CXL regions
bind directly to Device-DAX. `FS_DAX` is also required: this kernel uses it
to enable the inode DAX flag checked by the character-device mmap path. The effective running config is exported from `/proc/config.gz`. The cloud
image's stock kernel is used only to prepare the root filesystem.

Download the root filesystem URL from `image/manifest.json` to a path outside
this checkout; the builder verifies its SHA-256. Prepare an image once:

```bash
bash .buildkite/k3_tests/devdax/image/build.sh \
  --base /var/cache/lmcache/noble.img \
  --image /var/cache/lmcache/base.qcow2

LMCACHE_DEVDAX_IMAGE=/var/cache/lmcache/base.qcow2 \
LMCACHE_DEVDAX_KERNEL=/var/cache/lmcache/kernel/bzImage \
LMCACHE_DEVDAX_QEMU_ACCEL=kvm LMCACHE_DEVDAX_SUITE=both \
bash .buildkite/k3_tests/devdax/run.sh
```

The builder refuses to overwrite an existing image. It runs image provisioning
inside QEMU, caches CPU dependencies from the hash-locked Python requirements and version-locked OS packages,
retains the kernel config and package inventory, and removes the source checkout
before saving the image. Rebuild it when the recipe or dependency lock changes.
Update the Python lock with:

```bash
uv pip compile --python 3.12 --generate-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  .buildkite/k3_tests/devdax/image/requirements.in \
  -o .buildkite/k3_tests/devdax/image/requirements.lock
```

Each run creates a private overlay, key, QMP socket, serial log and ephemeral
localhost SSH port assigned by QEMU. It packages tracked and nonignored local
files after the repository PR-base merge, records commit/tree plus archive
SHA-256, and verifies that archive in the guest. Local uncommitted edits are
included. The installed module must resolve inside the extracted checkout.
Only that run's QEMU PID is terminated on exit or cancellation.

Topology follows the [QEMU volatile CXL example](https://www.qemu.org/docs/master/system/devices/cxl.html).
Only the guest discovers memdevs/decoders, creates three noninterleaved RAM
regions and validates sysfs, CXL parent identity, driver, capacity and alignment.
The guest provisioning command refuses to run outside the runner's DMI profile.
Automatic daxctl conversion to system RAM is disabled in the image.

KVM jobs have a 30-minute Buildkite deadline; explicit TCG jobs have 120 minutes.
Selected jobs fail when prerequisites, devices, reports or tests are missing.
TCG is a development option, not evidence of KVM readiness.

## Test and artifact contracts

`suites.json` declares exact node IDs. L1 and L2 run serially; an L1 failure
still allows L2 to report. Both requires four L1 and five L2 cases; individual
suites require four each. Collection must match the manifest, and JUnit must
contain every case with no skips, xfails or failures. Full deterministic payloads
and layouts are checked through the public allocator, task and StorageManager
interfaces. L2 StorageManager coverage first observes an empty resident L1.
Mapping resize stays within a fixed device; it is not CXL capacity hotplug.

The guest exports `LMCACHE_TEST_REQUIRE_REAL_DEVDAX=1`, both relevant opt-in
flags, dynamically discovered `LMCACHE_TEST_DEVDAX_L1_PATHS` and
`LMCACHE_TEST_DEVDAX_L2_PATHS`, and `LMCACHE_TEST_DEVDAX_MANIFEST`.
Ordinary opt-in tests create private temporary files. Supplied device paths
always undergo character-device and sysfs checks. Strict mode additionally
requires an exact guest manifest match. Symlink aliases count as duplicates.
Never point these destructive test fixtures at a device containing useful data.
The runner itself never supplies host device paths.

Artifacts are under `artifacts/devdax-qemu/run-<unique-id>/`. Each run retains
source/image identity, the QEMU command and serial console, guest topology and
device manifest, effective kernel config and package inventory, build logs,
per-selected-suite pytest output, collection JSON, JUnit XML and `summary.json`.
Summary states are `not selected`, `passed`, `failed`, and `infrastructure failure`.
The uploader separately saves `selection.json`; a skipped selection is labeled
SKIPPED, not passed coverage. Start with `infrastructure-error.txt` or the guest
build log for startup failures, and `pytest-l1.log` / `pytest-l2.log` for failures.

Real-VM failure checks (using the same image/kernel environment above):

```bash
RUN_DEVDAX_QEMU_HOST_TESTS=1 python -m pytest --confcutdir=tests/ci -s \
  tests/ci/test_devdax_qemu_runner.py
```

These deliberately mismatch a payload assertion, omit emulated devices, expire
the boot deadline, and time out a guest command; all must fail and clean up.

Fast local checks:

```bash
python -m pytest --confcutdir=tests/ci tests/ci/test_devdax_ci_selection.py
RUN_DEVDAX_L1_INTEGRATION=1 RUN_DAX_L2_INTEGRATION=1 OMP_NUM_THREADS=2 \
python -m pytest -q \
  tests/v1/distributed/test_devdax_l1_reconfigure_integration.py \
  tests/v1/distributed/test_dax_l2_integration.py
python -m pytest -q tests/v1/distributed/test_devdax_l1_allocator.py \
  tests/v1/distributed/test_dax_l2_adapter.py
```

## Buildkite activation (requires project administration)

Repository changes do **not** update the active Buildkite Steps editor.
No Buildkite API credentials or agent are available in the local development
environment, so activation and Buildkite run URLs remain outstanding:

1. Provision/map queue `devdax-qemu` to the host profile above, install the prepared
   image and kernel, and configure `LMCACHE_DEVDAX_IMAGE` and
   `LMCACHE_DEVDAX_KERNEL` in the agent environment.
2. Replace the K3 unit pipeline's Steps text with
   `../unit/buildkite-pipeline.yml`. It retains the existing unit uploader and
   adds an independent DAX uploader. Preserve existing fork/runner access policy.
3. Ensure GitHub PR webhooks and branch/path filters permit this bootstrap for
   all eligible PRs, including docs-only explicit `on` builds. Do not add the DAX
   uploader to the other test pipelines.
4. Create a dedicated manual/nightly pipeline using `buildkite-pipeline.yml`,
   repository `LMCache/LMCache`, default branch `dev`, and an enabled schedule
   `0 2 * * *` in UTC on branch `dev`. Schedule environment:
   `LMCACHE_DEVDAX_QEMU=auto`, `LMCACHE_DEVDAX_SUITE=both`,
   `LMCACHE_DEVDAX_QEMU_ACCEL=kvm`.
5. Run explicit `on` builds for `l1`, `l2`, and `both`; retain their URLs. Confirm
   an unrelated PR skips in `auto`, a DAX PR selects, and `off` affects only DAX.

This functional gate does not measure GPU registration/DMA, real CXL bandwidth,
coherence, physical hot removal, or persistence after volatile-index restart.

## Local validation — 2026-09-25

Tested checkout: `05fc77a0a7ababd9a7f2e343a771bc4bbc5b65cb` plus this uncommitted
CI/test change. Each run's `source.json` records the actual source-archive hash.
No production LMCache modules were changed.

| Check | Result |
| --- | --- |
| Selector, rendered upload/retry, fixtures and report validation | 44 passed |
| File-backed L1/L2 integration | 9 passed |
| Existing L1 allocator and MP DAX L2 unit suites | 56 passed |
| Existing DAX storage backend suite | 29 passed |
| KVM L1-only, rebuilt image | 4 passed, L2 not selected |
| KVM L2-only, rebuilt image | 4 passed, L1 not selected |
| KVM combined, rebuilt image | 4 L1 + 5 L2 passed; zero skips |
| Real-VM corruption/missing-device/timeout/cancellation checks | 5 passed |
| `SKIP=rust-fmt,rust-clippy pre-commit run --all-files` | Passed |
| ShellCheck and YAML parsing | Passed |
| `cd docs && make clean && make html` | Succeeded with 30 warnings in unchanged Sphinx sources |

The combined run used QEMU 11.1.1, Linux 6.12.50, Python 3.12.3, CPU PyTorch
2.10.0, ndctl/daxctl/cxl 77, four vCPUs and 8 GiB guest RAM. It discovered
three 256 MiB character devices with 2 MiB alignment, CXL parents and the
`device_dax` driver. The compiled `lmcache_native` extension was built and
loaded inside the guest. The suites took approximately 6 seconds (L1) and
7 seconds (L2), excluding boot and source compilation. These are functional
checks, not CXL performance measurements.

Local artifacts: `artifacts/devdax-qemu/run-adc6c3879610/` (combined),
`run-88d272da8a31/` (L1), `run-e22555437685/` (L2), and `local-validation/`.
The L1/L2 jobs overlapped in time using different overlays/sockets/SSH ports.
Cancellation testing kept another live VM responsive after killing the victim.

For this host, which grants passwordless sudo but lacks KVM group membership:

```bash
sudo -n -u dongjoo -g kvm env \
  PATH="/tmp/lmcache-devdax-image/qemu/bin:$PATH" \
  LMCACHE_DEVDAX_SUITE=both \
  bash .buildkite/k3_tests/devdax/run.sh \
  --image /tmp/lmcache-devdax-image/base-pinned.qcow2 \
  --kernel /tmp/lmcache-devdax-image/kernel/bzImage
```

The rebuilt image SHA-256 is
`ae83b732f59615b3a603afe97f34d3957476bbd7edafdba98a3e311ae89438a0`.
Base image, QEMU and kernel build commands above were executed locally. TCG
execution and actual Buildkite builds were not run. The exact queue/bootstrap/
nightly activation steps above remain with the project's Buildkite administrator.
Host package installation also encountered pre-existing pending NVIDIA DKMS
kernel-build errors; no host DAX devices were opened or reconfigured.
