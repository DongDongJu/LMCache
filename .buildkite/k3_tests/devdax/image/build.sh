#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../../.."
exec python3 .buildkite/k3_tests/devdax/runner.py --prepare-image "$@"
