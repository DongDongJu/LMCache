DAX
===

An L2 adapter that maps Device-DAX paths, such as ``/dev/daxX.X`` and
``/dev/daxY.Y``, and stores KV cache objects in fixed-size slots. This adapter
is intended for byte-addressable memory devices such as persistent memory or
CXL memory.

The MP ``dax`` adapter is volatile in this release.  It keeps the key index in
server memory and rebuilds an empty index on restart.  Old bytes may remain on
the DAX device, but they are unreachable after the LMCache server restarts.

**Required fields for the legacy single-device form:**

- ``device_path``: Path to the mmap-able DAX device or test file.
- ``max_dax_size_gb``: Number of GiB to map from ``device_path``.
- ``slot_bytes``: Fixed slot size in bytes. This must be large enough for one
  full LMCache chunk because MP memory descriptors do not expose the
  non-MP full-chunk size.

**Required fields for the multi-device form:**

- ``devices``: List of objects with ``device_path`` and ``max_dax_size_gb``.
  The list may be empty only when ``hotplug_enabled`` is ``true``.
- ``slot_bytes``: Fixed slot size in bytes shared by every DAX device in the
  adapter facade.

**Optional fields:**

- ``hotplug_enabled`` (bool, default ``false``): Enables runtime
  ``/reconfigure/dax/status``, ``/reconfigure/dax/add``,
  ``/reconfigure/dax/remove``, and ``/reconfigure/dax/resize``.
- ``num_store_workers`` (int, default ``1``): Store worker threads.
- ``num_lookup_workers`` (int, default ``1``): Lookup worker threads.
- ``num_load_workers`` (int, default ``min(4, os.cpu_count())``): Load worker
  threads.
- ``persist_enabled`` (bool): Accepted by common L2 config parsing but has no
  effect for ``dax`` because restart recovery is not implemented.
- ``watch_directory`` (str, default ``""``): Absolute directory that the
  adapter scans for present Device-DAX nodes (for example
  ``/dev/dax-cxl/<NAMESPACE>_<POD_NAME>``). Empty disables the presence
  watcher. The watcher only *reports* what it sees; it never attaches a
  device. See :ref:`the presence watcher section <dax-presence-watcher>`.
- ``watch_interval_seconds`` (float, default ``1.0``): Seconds between two
  watcher scans. Must be ``> 0``.

**Configuration examples:**

.. code-block:: bash

    # Backward-compatible single-device form.
    --l2-adapter '{
      "type": "dax",
      "device_path": "/dev/dax1.0",
      "max_dax_size_gb": 100,
      "slot_bytes": 268435456,
      "num_store_workers": 1,
      "num_lookup_workers": 1,
      "num_load_workers": 4,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

.. code-block:: bash

    # Multi-device hotplug-ready form.
    --l2-adapter '{
      "type": "dax",
      "devices": [
        {"device_path": "/dev/daxX.X", "max_dax_size_gb": 100},
        {"device_path": "/dev/daxY.Y", "max_dax_size_gb": 100}
      ],
      "slot_bytes": 268435456,
      "hotplug_enabled": true,
      "num_store_workers": 1,
      "num_lookup_workers": 1,
      "num_load_workers": 4
    }'

.. code-block:: bash

    # Hotplug-only form: start with no Device-DAX device at all.
    # The adapter is created and healthy with zero capacity, and devices are
    # attached later through /reconfigure/dax/add (by an operator or by the
    # MP Memory Coordinator).
    --l2-adapter '{
      "type": "dax",
      "devices": [],
      "slot_bytes": 268435456,
      "hotplug_enabled": true
    }'

A server started in the hotplug-only form declares an ``l2/dax`` compartment
with ``capacity_bytes: 0`` to the MP Coordinator, so its ``usage_ratio`` is
``null`` until the first device is added. The MP Memory Coordinator therefore
does not rank it as a donor or a receiver until then.

Runtime management uses JSON bodies because DAX paths contain slashes. See the
:doc:`Device-DAX backend guide </kv_cache/storage_backends/dax>` for complete
examples. These routes use StorageManager's generic L2 adapter reconfiguration
API; the HTTP path selects the backend and operation, the DAX adapter
interprets the operation payload, and the same interface can be reused by
future adapters such as P2P.

.. code-block:: bash

    curl http://127.0.0.1:9000/reconfigure/dax/status
    curl -X POST http://127.0.0.1:9000/reconfigure/dax/add \
      -H 'Content-Type: application/json' \
      -d '{"device_path": "/dev/daxX.X", "size": "100GiB"}'

.. _dax-presence-watcher:

**Presence watcher and physical inspection:**

Every mapped device carries a ``physical`` block in
``/reconfigure/dax/status`` and in ``/status`` (under
``l2_adapters[].devices[]``). It is the result of a *read-only* probe taken
when the device was attached: ``stat`` on the path plus plain reads under
``/sys/dev/char/<major>:<minor>`` (``subsystem``, ``driver``, ``size``,
``align``). The probe never ``open()``\ s the candidate device, because
opening an unbound dax node forks ``modprobe`` on the host and can stall.

``physical.mode`` is one of:

- ``devdax``: bound to ``device_dax``; usable by LMCache.
- ``system-ram``: bound to ``kmem``; the range is System RAM and is never
  mapped.
- ``unbound``: a dax device with no driver bound (for example a freshly
  created, size-0 device).
- ``not-a-device``: a regular file (test arenas) or a non-dax character
  device such as ``/dev/null``.
- ``absent``: the path does not exist.
- ``unknown``: sysfs could not be read; ``detail`` says why.

When ``watch_directory`` is set, the adapter also runs a daemon thread that
scans that directory every ``watch_interval_seconds`` and publishes the
result as ``watcher`` in ``/reconfigure/dax/status``. While the watcher is
enabled, the ``physical`` block of every attached device whose path lies
directly under ``watch_directory`` (spelled exactly as
``<watch_directory>/<name>``) is refreshed from the latest scan, so a
device that becomes unbound after mapping is visible in status; devices
attached under other paths keep their attach-time probe. A scan is adopted
only when it is at least as fresh as the probe already held, so a scan
that started before an attach never overwrites the attach-time probe. With
the watcher disabled, ``watcher`` is ``{"enabled": false}``.

.. code-block:: json

    {
      "hotplug_enabled": true,
      "slot_bytes": 268435456,
      "total_capacity_bytes": 274877906944,
      "total_used_bytes": 0,
      "devices": [
        {
          "index": 0,
          "device_path": "/dev/dax-cxl/ns_pod/dax0.1",
          "state": "active",
          "max_dax_size_bytes": 274877906944,
          "physical": {
            "device_path": "/dev/dax-cxl/ns_pod/dax0.1",
            "mode": "devdax",
            "present": true,
            "major": 249,
            "minor": 2,
            "kernel_name": "dax2.1",
            "driver": "device_dax",
            "size_bytes": 274877906944,
            "align_bytes": 2097152,
            "probed_at": 1756800000.0,
            "detail": ""
          }
        }
      ],
      "watcher": {
        "enabled": true,
        "directory": "/dev/dax-cxl/ns_pod",
        "interval_seconds": 1.0,
        "last_scan_at": 1756800001.0,
        "present_devices": [
          {"device_path": "/dev/dax-cxl/ns_pod/dax0.1", "mode": "devdax", "present": true, "major": 249, "minor": 2, "kernel_name": "dax2.1", "driver": "device_dax", "size_bytes": 274877906944, "align_bytes": 2097152, "probed_at": 1756800001.0, "detail": ""},
          {"device_path": "/dev/dax-cxl/ns_pod/dax0.2", "mode": "unbound", "present": true, "major": 249, "minor": 0, "kernel_name": "dax2.0", "driver": "", "size_bytes": 0, "align_bytes": 2097152, "probed_at": 1756800001.0, "detail": ""}
        ]
      }
    }

The MP server never attaches a present device on its own. Presence proves
that a node is usable, not that this server owns it: the same ``major:minor``
can be exposed under any number of directories, and a device that the
outside allocator has already handed to another node still looks identical
in ``/dev``. Only a caller that can prove allocator assignment should
consume ``watcher.present_devices`` and issue ``/reconfigure/dax/add``.

**Add gate:** ``POST /reconfigure/dax/add`` probes the path before mapping
and rejects the request with ``400`` (creating no entry) only on a positive
bad signal:

- ``physical.mode == "system-ram"``: ``DAX device is bound to kmem (system-ram)``
- ``physical.mode == "unbound"``: ``DAX device has no device_dax driver bound``
- ``physical.mode == "devdax"`` with a known sysfs ``size`` smaller than the
  requested size: ``DAX device size <size> < requested <size>`` (the kernel
  would accept the ``mmap`` and raise ``SIGBUS`` on first touch).

Inconclusive probes (``not-a-device``, ``absent``, ``unknown``) proceed
exactly as before, so regular-file test arenas keep working and a missing
path still fails in ``mmap``. The gate runs before the already-active
check: a same-size re-add of an attached path stays idempotent (``200``
with the existing entry) as long as the probe is not a positive bad signal,
but a re-add of a path that has since been bound to ``kmem`` or unbound is
rejected too, and the existing entry is left untouched.

**Current limits:**

- Runtime hotplug changes only LMCache mappings and metadata. It does not
  create, destroy, or reconfigure kernel CXL or DAX devices.
- Per-TP partitions and on-device restart metadata are not implemented.
- Only single-buffer objects are supported. Multi-tensor objects are rejected.
- Capacity is slot-based, not payload-byte-based. L2 eviction and usage
  metrics count occupied slots.
- Lookups acquire DAX-side external locks. ``submit_unlock`` releases those
  locks after load/retrieve completes, making entries evictable again.
- Remove ``mode="evict"`` is destructive for the DAX tier. Remove
  ``mode="migrate"`` requires enough capacity on another active DAX device.
