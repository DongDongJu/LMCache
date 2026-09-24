#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: build-kernel.sh <linux-6.12.50.tar.xz> <output-directory>
set -euo pipefail
archive="$(realpath "${1:?kernel source tarball required}")"
output="$(realpath -m "${2:?output directory required}")"
echo "c435bd74d1c21fc5a950781a50d78bae2b93944144694843359948ad3afc72a5  $archive" | sha256sum -c -
mkdir -p "$output"
build="$(mktemp -d "$output/build.XXXXXX")"
trap 'rm -rf "$build"' EXIT
tar -xf "$archive" -C "$build" --strip-components=1
cd "$build"
export KBUILD_BUILD_TIMESTAMP='2025-10-06 00:00:00 UTC'
export KBUILD_BUILD_USER=lmcache KBUILD_BUILD_HOST=devdax-ci
make x86_64_defconfig
# All boot and test drivers are built in: no guest module package dependency.
for config in FS_DAX DAX TRANSPARENT_HUGEPAGE NUMA ACPI_NUMA EFI_SOFT_RESERVE IKCONFIG IKCONFIG_PROC DEVTMPFS DEVTMPFS_MOUNT VIRTIO_PCI VIRTIO_BLK VIRTIO_NET EXT4_FS \
    CXL_BUS CXL_PCI CXL_ACPI CXL_MEM CXL_PORT CXL_REGION \
    CXL_REGION_INVALIDATION_TEST DEV_DAX DEV_DAX_CXL DEV_DAX_HMEM \
    DEV_DAX_HMEM_DEVICES MEMORY_HOTPLUG MEMORY_HOTREMOVE ZONE_DEVICE; do
    scripts/config --enable "$config"
done
scripts/config --disable DEV_DAX_KMEM --disable DEBUG_INFO --disable DEBUG_INFO_BTF --disable DRM \
    --disable SOUND --set-str SYSTEM_TRUSTED_KEYS '' --set-str SYSTEM_REVOCATION_KEYS ''
make olddefconfig
cp .config "$output/kernel.config"
for config in FS_DAX CXL_REGION_INVALIDATION_TEST DEV_DAX_CXL VIRTIO_BLK EXT4_FS; do
    grep -qx "CONFIG_${config}=y" .config
done
make -j"${LMCACHE_DEVDAX_KERNEL_JOBS:-4}" bzImage
cp arch/x86/boot/bzImage "$output/bzImage"
cp .config "$output/kernel.config"
sha256sum "$output/bzImage" > "$output/bzImage.sha256"
gcc --version > "$output/compiler.txt"
