Testing Companion L2 Plugins
============================

The Python filesystem adapter lives in the separate ``lmcache-l2-plugins``
package. LMCache keeps its ``fs`` registration and compatibility imports, while
``type: "plugin"`` loads the same implementation through the external adapter
loader. Both paths must be tested against the LMCache version being changed.

Local setup
-----------

Install LMCache, its common native extensions, and test dependencies into one
environment. For a CPU build, use ``NO_GPU_EXT=1`` when installing LMCache; the
``lmcache_native`` extension is still required. From the LMCache checkout:

.. code-block:: bash

    python -m pip install -r requirements/test.txt
    python tools/run_l2_plugin_tests.py
    LMCACHE_TRACK_USAGE=false python -m pytest -q \
        tests/v1/distributed/test_fs_plugin.py

The helper clones the repository and immutable revision recorded in
``tools/run_l2_plugin_tests.py`` into a temporary directory. It installs the
plugin into the current interpreter and runs the companion's complete test
suite. Installation uses ``--no-deps`` so it cannot replace the LMCache or
device runtime being validated. It uses ``uv`` when available, otherwise
``python -m pip``. Any checkout, installation, or test failure stops the command.

To validate an unpublished change across both repositories, provide a local
companion checkout:

.. code-block:: bash

    LMCACHE_L2_PLUGINS_PATH=../lmcache_l2_plugins \
        python tools/run_l2_plugin_tests.py
    LMCACHE_TRACK_USAGE=false python -m pytest -q \
        tests/v1/distributed/test_fs_plugin.py \
        tests/v1/distributed/test_l2_adapter_factory.py \
        tests/test_usage_telemetry.py

The helper installs a wheel built from that checkout. Run it again after editing
plugin source. For interactive development without reinstalling, use
``python -m pip install --no-deps -e ../lmcache_l2_plugins`` instead.

Automatic CI integration
------------------------

The main GitHub test job runs the helper after building LMCache and before the
CPU unit suite. Buildkite's LMCache-only, vLLM, and SGLang setup scripts also run
it, covering the shared unit and MP pipelines. The standalone AMD, MUSA, Metax,
RBLN, and legacy unit entrypoints invoke the same helper.

The companion tests run in their own checkout. Core's
``tests/v1/distributed/test_fs_plugin.py`` runs under the normal LMCache pytest
fixtures and requires the plugin at collection time. This prevents a missing
installation from silently skipping the integration contract.

When updating the companion pin:

1. Validate both checkouts with ``LMCACHE_L2_PLUGINS_PATH``.
2. Publish the tested companion commit to the repository named by
   ``PLUGIN_REPOSITORY`` in the helper.
3. Update ``PLUGIN_REVISION`` to its full commit hash and run the helper without
   the local override to check the published installation path.

The initial migration pin uses the integration fork. That commit must be
published before remote CI can fetch it. Switch the repository and revision
to upstream once upstream contains the filesystem implementation. A local
override validates the working tree; it does not prove the remote pin exists.

Coverage and limits
-------------------

.. list-table:: Filesystem migration checks
   :header-rows: 1
   :widths: 30 70

   * - Contract
     - Coverage
   * - Configuration and imports
     - Legacy ``fs`` and explicit ``plugin`` loading, typed config discovery,
       validation, L1 descriptor forwarding, and compatibility class identity.
   * - Files and isolation
     - Existing filename encoding, model paths, object groups, salted keys,
       raw file contents, temporary writes, missing keys, and restart reads.
   * - I/O
     - Real tensor buffers, buffered reads, read-ahead, aligned direct I/O,
       unaligned-length fallback, and truncated files reported as misses.
   * - Controllers and wrappers
     - Real L1 write and L2 store, L1 clear, L2 prefetch, exact loaded values,
       FP8 serialize/deserialize, and a pass-through fault-injection wrapper.
   * - Lifecycle and accounting
     - Store/load/delete notifications, per-salt byte counts, runtime adapter
       removal and re-addition, preserved files, and released L1 memory.
   * - Core regression tests
     - Factory registration, lazy loading, telemetry, and native filesystem
       configuration tests remain in LMCache's suite.

These checks run without a GPU. Direct-I/O cases first probe the test filesystem
using an aligned buffer and skip only when the OS or filesystem does not support
that operation. Use the target filesystem for hardware-specific validation.
GPU serving-engine integration and GPU-only serde tests still require their
normal device jobs. A successful CPU run does not establish storage throughput
or full vLLM/SGLang inference correctness.

The ``fs_native`` implementation remains in core. Its migration and native
connector performance are outside this filesystem plugin change.

For a CLI data-integrity smoke test on a fresh directory:

.. code-block:: bash

    lmcache bench l2 \
        --l2-adapter '{"type":"plugin","module_path":"lmcache_l2_plugins.fs","class_name":"FSL2Adapter","adapter_params":{"base_path":"/tmp/lmcache-plugin-check"}}' \
        --num-keys 4 --data-size-kb 4 --rounds 2 --warmup-rounds 0 \
        --no-skip-verify

For direct I/O, add ``"use_odirect": true`` to ``adapter_params`` and pass
``--l1-align-bytes 4096`` on a filesystem with 4096-byte blocks. Adjust the buffer
size and alignment to the filesystem's block size when it differs.
