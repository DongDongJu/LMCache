Shared Device-DAX L1
====================

Several MP servers can share one Device-DAX region as an L1: a chunk stored by
one server is retrieved by every other server from its own mapping of the
region, without a network copy. Sharing is an option of the ordinary
``DEVDAX`` L1. A separate process, the memory orchestrator
(``lmcache memory``), owns the region's allocation, key index and object
state; KV bytes never pass through it. Without a ``shared`` object, a
``DEVDAX`` L1 stays private and never contacts an orchestrator. Since the first
revision of #5899, the standalone ``SharedL1Manager`` is gone, ``owns_range``
replaces ``owns_device``, ``resolve_l1_affinity`` replaces ``affinity_tag``, and
lookups no longer ask private L1s first.

.. code-block:: text

   store:  ReserveWrite -> view of the granted extent -> D2H copies complete
           -> publish (cache write-back) -> FinishWrite      (readable everywhere)
   read:   ReserveRead (lease) -> acquire (cache invalidate) -> H2D copies
           complete -> FinishRead
   failed store: queued copies drain -> AbortWrite (never published)

Start the orchestrator
----------------------

One orchestrator per region, on any host the servers can reach:

.. code-block:: bash

   lmcache memory --region-id pool-a --capacity-gb 512 --alignment 2M \
       --listen 0.0.0.0:7700 --state-dir /var/lib/lmcache/memory/pool-a

``--capacity-gb`` (or ``--capacity-bytes``) is the size every server maps;
``--alignment`` is the extent alignment (default ``2M``);
``--visibility-mode`` is ``software_fenced`` (default; servers write back and
invalidate CPU caches around every copy, for memory several hosts map without
hardware coherence) or ``coherent`` (no cache maintenance);
``--state-dir`` holds the startup marker that keeps a second orchestrator off
the region; ``--max-batch-entries`` caps one RPC (default 4096).
``software_fenced`` requires an extent alignment of at least one host page so
independent writers never share a CPU cache line.

Configure the servers
---------------------

A shared L1 is a ``DEVDAX`` ``--l1-manager`` with a ``shared`` object. List it
first: writes follow ``--l1-manager`` order and only out-of-memory writes move
on, so stores land where every server can read them and a following DRAM L1
takes them once the region is full.

.. code-block:: bash

   # CXL region shared across hosts, private DRAM as overflow
   lmcache server --chunk-size 256 --eviction-policy LRU \
       --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DEVDAX","tag":"cxl","size_gb":512,"path":"/dev/dax0.0",
                      "offset_gb":0,
                      "shared":{"orchestrator":"10.0.0.5:7700","region_id":"pool-a",
                                "client_id":"host-a:5555"}}' \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":64}'

``shared`` takes ``orchestrator`` (``host:port``), ``region_id`` (matching
``--region-id``), ``client_id`` and the optional ``rpc_timeout_seconds``
(default 5). The region is mapped from the L1's own ``path`` and
``offset_gb``; extents are relative to that offset. ``size_gb`` must cover the
region capacity. Shared eviction defaults to, and requires, ``noop``. GPU
registration of the mapping must succeed; startup fails rather than falling
back to pageable copies.

Give each MP server a unique, stable ``client_id``. Peers map the same
physical bytes through their own paths: paths may differ between hosts. The
operator asserts that every server's ``path`` and ``offset_gb`` name the same
physical memory; LMCache cannot verify it, and a mismatch serves wrong KV
data without an error.

The legacy single-manager DRAM-plus-DAX hybrid (``--l1-devdax-size-gb``)
cannot be shared; configure private DRAM as a separate ``--l1-manager``.

Testing with a file
~~~~~~~~~~~~~~~~~~~

The ``DEVDAX`` option also maps a regular file, so CPU tests share a region by
pointing every server at one file (for example on tmpfs) with a ``coherent``
orchestrator. This is a test path only: host-local sharing is not a supported
production topology, and there is no shared ``DRAM`` L1 (a ``DRAM`` entry
rejects a ``shared`` object).

Supported topology and limits
-----------------------------

- Cross-host Device-DAX, ``lmcache_driven`` transfer, equal TP1/2/4/8 peers.
  The shared L1 reports memory tags ``CXL``/``REMOTE``.
- CacheBlend, P2P, engine-driven or ``auto`` transfer, and runtime plugins are
  rejected at startup.
- One shared L1 per server. L2 adapters cannot bind to it; Device-DAX
  hot-plug of a shared L1 returns 409; GDS and DRAM L1s cannot be shared.
- Shared objects are never evicted. ``delete`` makes objects unreadable for
  every server, and ``clear`` deletes the objects this server's
  ``client_id`` wrote, leaving other servers' objects. Neither frees space:
  deleted, aborted and expired extents all stay consumed, so once the region
  is full new stores overflow to the next L1 until the region is reset.
  Leases taken before a delete keep their bytes readable until released.
- There is no authentication or tenant isolation: every attached server can
  read every object of the region. Attach servers of one trust domain only.

Failures and reset
------------------

- **Orchestrator unreachable**: reads miss, stores overflow to the next L1,
  ``delete`` returns ``KEY_IS_LOCKED``, ``clear`` logs that nothing was
  deleted, and the shared L1 reports ``state: unavailable`` until an RPC
  succeeds.
- **Ambiguous commit** (FinishWrite reply lost): the store reports failure
  and the extent is not reused; a later writer may find the key committed.
- **Malformed grant** or a peer layout that differs from the reader's: the
  grant is released and logged, never reported as capacity pressure.
- **Region changed** (epoch change or ``RESET_REQUIRED``): the shared L1
  reports ``state: disabled`` and serves nothing until the server restarts
  after the region is reset. It never re-registers on its own.
- **Server restart**: the server registers under its ``client_id`` again,
  which retires its previous incarnation's writes and leases. Extents are
  never reused, so copies the previous incarnation left in flight cannot
  overwrite an object other servers read.

An orchestrator restart loses its in-memory state, so it comes back requiring
a reset. Reset offline: stop every server using the region, stop the
orchestrator, delete ``<state-dir>/<region-id>.marker``, start the
orchestrator, start the servers.

Tensor parallel serving
-----------------------

Peers must use the same model, serving engine and version, TP size, KV dtype,
chunk size and hash algorithm. Lookup serves only the prefix every rank and
object group (including hybrid state) has stored; a peer object whose shapes
or dtypes differ from the reader's is rejected. KV is never resharded.

Status and metrics
------------------

``GET /status`` lists each L1 under ``storage_manager.l1_managers``. The
shared entry counts this server's write grants and read leases, and its
``shared`` section reports ``state`` and ``disabled_reason``.
``memory_used_bytes`` is this server's uncommitted write payload;
``memory_total_bytes`` is the whole region's capacity.

The region's capacity appears on every attached server, in ``/status`` and in
``lmcache_mp_l1_*`` gauges: summing those reports counts the same physical
region once per server. Placements in a shared L1 are not reported as fleet
cache events.
