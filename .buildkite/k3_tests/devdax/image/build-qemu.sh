#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: build-qemu.sh <qemu-11.1.1.tar.xz> <install-prefix>
set -euo pipefail
archive="$(realpath "${1:?QEMU source tarball required}")"
prefix="$(realpath -m "${2:?install prefix required}")"
echo "079ffbff8a7111bbc89022107cbabf3bbfd614d5fc9d7cc675991196aca12482  $archive" | sha256sum -c -
mkdir -p "$prefix"
build="$(mktemp -d "$prefix/build.XXXXXX")"
trap 'rm -rf "$build"' EXIT
tar -xf "$archive" -C "$build" --strip-components=1
cd "$build"
./configure --prefix="$prefix" --target-list=x86_64-softmmu \
    --without-default-features --enable-system --enable-tools \
    --enable-kvm --disable-tcg --enable-slirp --enable-pixman --enable-download \
    --disable-werror
make -j"${LMCACHE_DEVDAX_QEMU_JOBS:-4}"
make install
