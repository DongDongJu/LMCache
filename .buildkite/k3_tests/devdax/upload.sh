#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../.."
python3 .buildkite/k3_tests/devdax/ci_selection.py
report="${LMCACHE_DEVDAX_ARTIFACT_DIR:-artifacts/devdax-qemu}"
buildkite-agent artifact upload "$report/selection.json"
if ! python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["run"])' "$report/selection.json"; then
    buildkite-agent annotate --context devdax-selection --style info "DevDAX QEMU coverage SKIPPED; see selection.json."
    exit 0
fi
# A retried bootstrap must not upload the same step key twice.
if buildkite-agent step get key --step devdax-qemu >/dev/null 2>&1; then
    echo 'DevDAX step already uploaded'
    exit 0
fi
buildkite-agent pipeline upload "$report/pipeline.json"
