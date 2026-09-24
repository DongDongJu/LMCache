#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../.."
# shellcheck source=.buildkite/k3_tests/common_scripts/helpers.sh
source .buildkite/k3_tests/common_scripts/helpers.sh
merge_pr_base_branch origin
exec python3 .buildkite/k3_tests/devdax/runner.py "$@"
