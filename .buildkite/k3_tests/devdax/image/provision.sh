#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs only in the disposable image-building guest.
set -euo pipefail
[[ "$(cat /sys/class/dmi/id/product_name)" == LMCache-DevDAX-QEMU ]]
cd /root/source
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential pkg-config libnuma-dev \
    ndctl=77-2ubuntu2 daxctl=77-2ubuntu2 cxl=77-2ubuntu2
mkdir -p /etc/udev/rules.d
ln -sf /dev/null /etc/udev/rules.d/90-daxctl-device.rules
python3 -m venv /opt/lmcache-test
# shellcheck source=/dev/null
source /opt/lmcache-test/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
# ponytail: dependency resolution can change on rebuild; retain the checksummed image.
python -m pip install -r requirements/build.txt -r requirements/common.txt \
    -r requirements/test.txt
mkdir -p /opt/lmcache-image
dpkg-query -W > /opt/lmcache-image/packages.txt
python -m pip freeze > /opt/lmcache-image/python-packages.txt
cloud-init clean --logs
