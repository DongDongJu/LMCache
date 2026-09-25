#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode="${LMCACHE_DEVDAX_QEMU-on}"
case "$mode" in
    on) enabled=true ;;
    off) enabled=false ;;
    *) echo 'LMCACHE_DEVDAX_QEMU must be on or off' >&2; exit 2 ;;
esac
report="${LMCACHE_DEVDAX_ARTIFACT_DIR:-artifacts/devdax-qemu}"
mkdir -p "$report"
printf '{"LMCACHE_DEVDAX_QEMU":"%s","run":%s}\n' "$mode" "$enabled" > "$report/selection.json"
buildkite-agent artifact upload "$report/selection.json"
if [[ "$mode" == off ]]; then
    buildkite-agent annotate --context devdax-selection --style info "DevDAX QEMU coverage SKIPPED: LMCACHE_DEVDAX_QEMU=off."
    exit 0
fi
# A retried bootstrap must not upload the same step key twice.
if buildkite-agent step get key --step devdax-qemu >/dev/null 2>&1; then
    echo 'DevDAX step already uploaded'
    exit 0
fi
buildkite-agent pipeline upload .buildkite/k3_tests/devdax/pipeline.yml
