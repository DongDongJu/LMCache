#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /root/source
# shellcheck source=/dev/null
source /opt/lmcache-test/bin/activate
export NO_GPU_EXT=1 MAX_JOBS=4 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0
export LMCACHE_TRACK_USAGE=false
unset LMCACHE_DEVICE_BACKEND
unset NO_NATIVE_EXT
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
python lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py
python - <<'PY'
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import json
import platform
import subprocess
import lmcache
import lmcache.lmcache_native as native
import torch
assert Path(lmcache.__file__).is_relative_to('/root/source')
assert any(native.__file__.endswith(s) for s in EXTENSION_SUFFIXES)
assert torch.version.cuda is None
assert lmcache.torch_device_type == "cpu"
result = dict(kernel=platform.release(), python=platform.python_version(),
              torch=torch.__version__, lmcache=lmcache.__version__, native=native.__file__,
              cxl=subprocess.check_output(['cxl', '--version'], text=True).strip(),
              daxctl=subprocess.check_output(['daxctl', '--version'], text=True).strip())
Path('artifacts/devdax-qemu/versions.json').write_text(json.dumps(result, indent=2) + '\n')
PY
