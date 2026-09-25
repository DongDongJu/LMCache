#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../.."
# shellcheck source=.buildkite/k3_tests/common_scripts/helpers.sh
source .buildkite/k3_tests/common_scripts/helpers.sh
merge_pr_base_branch origin
image="${LMCACHE_DEVDAX_CONTAINER_IMAGE:-}"
if [[ -z "$image" && "${BUILDKITE:-}" == true ]]; then
    image=ghcr.io/lmcache/lmcache-devdax-ci:nightly
fi
if [[ -n "$image" ]]; then
    # Local image IDs support pre-publication testing; registry references refresh.
    if [[ "$image" != sha256:* ]]; then
        docker pull -- "$image"
    fi
    image=$(docker image inspect --format '{{.Id}}' -- "$image")
    mkdir -p artifacts/devdax-qemu
    docker image inspect --format '{"id":{{json .Id}},"digests":{{json .RepoDigests}}}' -- "$image" \
        > artifacts/devdax-qemu/container-image.json
    exec docker run --rm --init --device /dev/kvm \
        --group-add "$(stat -c %g /dev/kvm)" \
        -e "LMCACHE_CI_UID=$(id -u)" -e "LMCACHE_CI_GID=$(id -g)" \
        --volume "$PWD:$PWD" --workdir "$PWD" \
        -e HOME -e LMCACHE_DEVDAX_SUITE \
        -e LMCACHE_DEVDAX_QEMU -e LMCACHE_DEVDAX_BOOT_TIMEOUT \
        -- "$image" python3 .buildkite/k3_tests/devdax/runner.py "$@"
fi
exec python3 .buildkite/k3_tests/devdax/runner.py "$@"
