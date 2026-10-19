FileSystem
==========

A pure file-system L2 adapter using async I/O (``aiofiles``).  Each KV cache
object is stored as a raw ``.data`` file whose name encodes the full
``ObjectKey``.  Does **not** require NIXL -- works on any POSIX file system.

The implementation is provided by the separate ``lmcache-l2-plugins`` package.
Install its checkout into the same environment as LMCache:

.. code-block:: bash

    python -m pip install -e /path/to/lmcache_l2_plugins

Existing ``type: "fs"`` configurations and Python imports remain supported
through a compatibility module in core. Existing cache files require no
conversion. The adapter can also be loaded through the standard plugin loader:

.. code-block:: bash

    --l2-adapter '{"type":"plugin","module_path":"lmcache_l2_plugins.fs","class_name":"FSL2Adapter","adapter_params":{"base_path":"/data/lmcache/l2"}}'

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``relative_tmp_dir``: Relative sub-directory for temporary files during
  writes (atomic rename on completion).
- ``read_ahead_size``: Trigger file-system read-ahead by reading this many
  bytes first (positive integer, optional).
- ``use_odirect``: ``true`` or ``false`` (default ``false``) -- bypass the
  page cache via ``O_DIRECT``.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'
