Shared L1
=========

.. note::

   Experimental and opt-in. The interface may change.

Several MP servers can share one memory region as an L1. A chunk stored by
one server is retrieved by every other server from its own mapping of the
region, without a network copy. A separate process, the memory orchestrator
(``lmcache memory``), owns the region's allocation, key index and object
state; KV bytes never pass through it. The region is backed by a *medium*
each server maps on its host:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - L1 type
     - Medium
     - Use
   * - ``DEVDAX``
     - a Device-DAX character device (``path``)
     - CXL memory attached to several hosts; needs ``software_fenced``
   * - ``DRAM``
     - a regular file (``path``), created or extended on first use
     - several servers on one host over tmpfs (``/dev/shm``) or hugetlbfs;
       ``coherent``

Start the orchestrator
----------------------

One orchestrator per region, on any host the servers can reach:

.. code-block:: bash

   lmcache memory --region-id pool-a --capacity-gb 512 --alignment 2M \
       --listen 0.0.0.0:7700 --state-dir /var/lib/lmcache/memory/pool-a

``--region-id`` names the region; ``--capacity-gb`` (or ``--capacity-bytes``)
is what every server maps from offset zero of its medium; ``--alignment``
sets the extent alignment (default ``2M``); ``--visibility-mode`` is
``software_fenced`` (default; servers flush CPU caches around every copy, for
memory shared by several hosts) or ``coherent`` (no cache maintenance, for a
region one host maps); ``--state-dir`` holds the startup marker that keeps a
second orchestrator off the same region; ``--max-batch-entries`` caps one
RPC (default 4096).

Configure the servers
---------------------

A shared L1 is a ``--l1-manager`` with a ``shared`` object:

.. code-block:: bash

   # CXL region shared across hosts, private DRAM as overflow
   lmcache server --chunk-size 256 --eviction-policy LRU \
       --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DEVDAX","tag":"cxl","size_gb":512,"path":"/dev/dax0.0",
                      "shared":{"orchestrator":"10.0.0.5:7700","region_id":"pool-a"}}' \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":64}'

   # tmpfs region shared by several servers on one host
   lmcache server --chunk-size 256 --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DRAM","tag":"pool","size_gb":64,"path":"/dev/shm/pool-a",
                      "shared":{"orchestrator":"127.0.0.1:7700","region_id":"pool-a"}}'

List the shared L1 first: writes follow ``--l1-manager`` order and only
out-of-memory writes move on, so stores land where every server can read
them and a following DRAM L1 takes them once the region is full. Lookups
always ask local L1s first and the shared L1 only for their misses.

``shared`` fields: ``orchestrator`` (``host:port``, required), ``region_id``
(must equal ``--region-id``, required), ``client_id`` (this server's identity
across restarts; default ``<hostname>-<first 8 characters of
/etc/machine-id>:<port>``; set it where that does not tell hosts apart) and
``rpc_timeout_seconds`` (default 5). ``size_gb`` must be at least the region
capacity. The eviction policy defaults to, and must be, ``noop``.

The operator establishes that every server's ``path`` names the same physical
memory; LMCache cannot infer it.

Limits
------

- ``lmcache_driven`` transfer and TP 1 only. CacheBlend, P2P, experimental
  modules (``--enable``) and ``lmcache trace replay`` are rejected.
- One shared L1 per server; L2 adapters cannot use it as ``affinity_tag``.
  GDS L1s cannot be shared.
- Shared objects are never evicted or deleted and extents are never reused,
  so once the region is full new stores overflow to the next L1 until the
  region is reset. Shared placements are not reported as fleet cache events.

Failures and reset
------------------

A shared-L1 failure never crashes the server: reads miss and stores overflow.

- **Orchestrator unreachable**: the shared L1 reports unhealthy and serves
  again once the orchestrator answers.
- **A server dies mid-store**: its chunks stay unreadable (others recompute
  them) until it restarts under the same ``client_id``.
- **Orchestrator restarts**: its state is in memory, so it comes back
  requiring a reset and every server's shared L1 stops serving. Reset
  offline: stop every server using the region, stop the orchestrator, delete
  ``<state-dir>/<region-id>.marker``, start the orchestrator, start the
  servers.
