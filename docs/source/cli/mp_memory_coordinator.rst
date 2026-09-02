lmcache mp-memory-coordinator
=============================

The ``lmcache mp-memory-coordinator`` command runs the standalone **MP Memory
Coordinator**, a process that rebalances Device-DAX capacity between MP
servers: it reads the fleet's occupancy from the MP Coordinator, picks one LOW
donor and one HIGH receiver after three stable samples, and moves one managed
allocation (donor drain -> donor remove -> outside deallocate -> outside
allocate -> receiver add) through a crash-safe journal.

.. code-block:: bash

   lmcache mp-memory-coordinator --config /etc/lmcache/mp-memory-coordinator.yaml

What it does
------------

Every ``poll_interval_seconds`` it takes a *sandwich* read of the fleet
(``/instances`` -> ``/instances/usage`` -> ``/instances``) and accepts an
instance only when its identity (registration epoch, address, and
``metadata.worker_ip``) is unchanged across the reads and its private
``l2/dax`` compartment has a declared capacity. An instance whose
``used/capacity`` ratio stays **HIGH** (``>= high_ratio``) for
``stable_samples`` consecutive samples is a receiver candidate; one that stays
**LOW** (``<= low_ratio``) is a donor candidate. After a live preflight of both
servers (``/status`` and ``/reconfigure/dax/status``), the least-used managed
runtime device of the donor is moved:

.. code-block:: text

   donor drain -> donor evict -> outside deallocate -> outside allocate -> receiver add

Only devices the coordinator manages take part: the bootstrap device at DAX
index 0, shared pools, and unlisted paths are never touched. With the default
``actuation_enabled: false`` the process observes, logs its proposals and
rejection reasons, and mutates nothing.

Node identity
-------------

The outside allocation service addresses a worker by its host IP. MP servers
register it as ``metadata.worker_ip`` from the ``LMCACHE_WORKER_NODE_IP``
environment variable (or ``--coordinator-worker-node-ip``); the operator's
DaemonSet injects it from ``status.hostIP``. It is metadata only: the direct
MP address stays the Pod IP (``LMCACHE_COORDINATOR_ADVERTISE_IP``).

Safety
------

* Every side effect is journaled before it is issued (intent, then
  ``dispatched``, then result), with fsync and atomic replacement, so a
  restart resumes from durable evidence and current status.
* An outside POST is issued at most once. If its outcome cannot be proven
  after a crash, the move enters ``BLOCKED`` and nothing further is mutated
  until an operator reconciles it.
* Before every POST the process re-renews its Lease and re-reads the fleet;
  a lost Lease, an unreachable MP Coordinator, or a changed identity defers
  the effect.
* Rollbacks restore the donor (re-add the old path, or allocate the same
  size back and attach the returned path) or end safely in ``BLOCKED``.
* A corrupt, truncated, or unknown-version journal makes the process unready
  and inert.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--config PATH``
     - Required. YAML configuration file (keys below).
   * - ``--adopt PATH``
     - Adopt the allocations listed in this allowlist into the journal once,
       then exit. The coordinator never discovers devices on its own.
   * - ``--check``
     - Validate the configuration and exit.

Exit status ``2`` means a configuration error (unknown key, wrong type, or a
failed validation such as ``low_ratio >= high_ratio``).

Configuration
-------------

Unknown keys and wrongly typed values are rejected. Every key has a safe
default; ``actuation_enabled`` is ``false`` unless set.

.. code-block:: yaml

   mp_coordinator_url: http://lmcache-mp-coordinator:9300
   memory_allocation_url: http://memory-allocation-service:8080
   poll_interval_seconds: 10
   stable_samples: 3
   high_ratio: 0.75
   low_ratio: 0.40
   minimum_ratio_gap: 0.25
   projected_donor_max_ratio: 0.70
   cooldown_seconds: 300
   adapter_index: 0
   min_devices_per_instance: 1
   allowed_device_path_prefix: /dev/dax-cxl/
   drain_timeout_seconds: 300
   state_directory: /var/lib/lmcache-memory-coordinator
   actuation_enabled: false
   http_host: 0.0.0.0
   http_port: 9400
   leader_election: none          # or kubernetes (pre-created Lease)
   lease_name: lmcache-mp-memory-coordinator
   lease_duration_seconds: 15
   lease_renew_interval_seconds: 5
   adoption_file: ""              # applied once when the journal is uninitialized

Adoption allowlist
------------------

.. code-block:: yaml

   allocations:
     - worker_ip: 192.168.0.40
       device_path: /dev/dax-cxl/NAMESPACE_POD_NAME/dax0.x
       allocation_size_gib: 64
       device_map_size_bytes: 68719476736

An entry is adopted only when the path is active at DAX index ``> 0`` on the
MP server registered for that worker IP, is listed under the same worker by
the outside allocation service, and matches the approved size.

Endpoints
---------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Path
     - Meaning
   * - ``/healthz``
     - Process alive and journal readable (503 on a corrupt journal).
   * - ``/readyz``
     - Leader, MP Coordinator reachable, inventory reconciled, no BLOCKED move.
   * - ``/status``
     - Inventory, cooldowns, pressure history, active move, counters, last cycle.
   * - ``/journal``
     - The durable journal document (read-only).
   * - ``/metrics``
     - Prometheus counters: proposed, succeeded, rolled back, blocked moves.

Rollout
-------

1. Deploy with ``actuation_enabled: false`` and an explicit adoption
   allowlist; observe proposals and rejection reasons for at least 24 hours.
2. Confirm ``high_ratio`` is below any local eviction trigger.
3. Enable one donor/receiver canary with a five-minute cooldown and alert on
   ``BLOCKED`` (``/readyz`` turns 503, ``lmcache_memcoord_moves_blocked_total``).
4. Never scale down, downgrade, or detach the volume during an active or
   blocked move; disable by setting observation-only mode first and waiting
   for ``COMPLETE``.
