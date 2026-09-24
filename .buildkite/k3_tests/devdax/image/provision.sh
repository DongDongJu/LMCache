#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Executed only in the disposable guest by runner.py --prepare-image.
set -euo pipefail
[[ "$(cat /sys/class/dmi/id/product_name)" == LMCache-DevDAX-QEMU ]]
export DEBIAN_FRONTEND=noninteractive
apt-get update
mapfile -t packages < /root/source/.buildkite/k3_tests/devdax/image/apt-packages.lock
apt-get install -y --no-install-recommends "${packages[@]}"
# Prevent automatic conversion to system-ram; provisioning must observe devdax.
mkdir -p /etc/udev/rules.d
ln -sf /dev/null /etc/udev/rules.d/90-daxctl-device.rules
python3 -m venv /opt/lmcache-test
/opt/lmcache-test/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    --require-hashes -r /root/source/.buildkite/k3_tests/devdax/image/requirements.lock
mkdir -p /opt/lmcache-image
cp /boot/config-6.8.0-139-generic /opt/lmcache-image/kernel.config
dpkg-query -W > /opt/lmcache-image/packages.txt
/opt/lmcache-test/bin/pip freeze > /opt/lmcache-image/python-packages.txt
cxl --version > /opt/lmcache-image/cxl-version.txt
# Each test overlay supplies a new key and instance ID through NoCloud.
cloud-init clean --logs
