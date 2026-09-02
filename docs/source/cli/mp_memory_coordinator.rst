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
       then exit without starting the control loop. Stop a running
       coordinator first: the journal has no cross-process lock.
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

Which devices the coordinator manages
-------------------------------------

The coordinator only moves a device in its managed inventory, because the
donor step ends in ``POST /deallocations`` and must never hand back memory it
does not own. Every cycle it re-derives ownership from the outside Memory
Allocation service: a live device is adopted when its DAX index is ``> 0``,
its state is ``active`` and it is healthy and not closing, its path starts
with ``allowed_device_path_prefix``, the outside service lists that exact
path under exactly the one worker IP the MP instance registered, and its DAX
map size is a positive whole number of GiB. No allowlist is needed, and a
path that changes with its Pod name is re-derived instead of going stale.
Devices that are declined are reported per path in
``/status.last_cycle.discovery.skipped``, so an empty inventory is always
explained.

Adoption allowlist
------------------

Optional: discovery finds the same devices on its own. Use it when a path
must be approved explicitly before the coordinator may manage it.

.. code-block:: yaml

   allocations:
     - worker_ip: 192.168.0.40
       device_path: /dev/dax-cxl/NAMESPACE_POD_NAME/dax0.x
       allocation_size_gib: 64
       device_map_size_bytes: 68719476736

An entry is adopted only when the path is active at DAX index ``> 0`` on the
MP server registered for that worker IP, is listed under the same worker by
the outside allocation service, and matches the approved size. ``adoption_file``
is read only while the journal carries no ``initialized`` marker; ``--adopt``
applies an allowlist at any time.

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

Kubernetes deployment example
-----------------------------

A production Kustomize example is available in
`examples/mp_memory_coordinator/README.md`_. It creates namespace
``lmcache-system``, a single-replica ``Recreate`` Deployment, narrow Lease
RBAC, a pre-created Lease, a ``ReadWriteOncePod`` journal PVC, probes, and a
Service on port 9400.

Before applying it, edit
``examples/mp_memory_coordinator/kubernetes/config/mp-memory-coordinator.yaml``
with the MP Coordinator base URL, set the pinned LMCache image in the
Kustomization, and keep ``actuation_enabled: false``. No allowlist is
required: the coordinator discovers the devices the outside service assigns
to each worker. Fill in ``adoption.yaml`` and point ``adoption_file`` at
``/etc/lmcache/adoption.yaml`` only if a path must be approved explicitly.
The
``memory_allocation_url`` value is the allocator **base URL**; do not include
an API path. The client calls ``<base>/api/v2/apps/lmcache`` for status and
``<base>/api/v2/apps/lmcache/allocations`` or
``<base>/api/v2/apps/lmcache/deallocations`` for mutations.

From the repository root, render and deploy the example, then inspect its
read-only status API:

.. code-block:: bash

   kubectl kustomize examples/mp_memory_coordinator/kubernetes
   kubectl apply -k examples/mp_memory_coordinator/kubernetes
   kubectl -n lmcache-system rollout status \
     deployment/lmcache-mp-memory-coordinator --timeout=5m
   kubectl -n lmcache-system port-forward \
     service/lmcache-mp-memory-coordinator 9400:9400

In another terminal, require ``leader: true``, the expected inventory, and
``actuation_enabled: false`` before considering actuation. If the inventory is
empty, ``last_cycle.discovery.skipped`` names the reason for every live device
that was declined:

.. code-block:: bash

   curl -fsS http://127.0.0.1:9400/healthz
   curl -sS http://127.0.0.1:9400/readyz
   curl -fsS http://127.0.0.1:9400/status | jq \
     '{leader, actuation_enabled, inventory, active_move, last_cycle}'
   curl -fsS http://127.0.0.1:9400/status | jq .last_cycle.discovery

Observe an eligible dry-run proposal and confirm that the allocator received
no POST before setting ``actuation_enabled: true`` and re-applying the
Kustomization. Never scale above one replica, change the ``Recreate`` strategy,
or delete/detach the journal PVC. Observation mode prevents new moves but does
not stop recovery of an already durable move; preserve a ``BLOCKED`` journal
for manual reconciliation.

.. _examples/mp_memory_coordinator/README.md: https://github.com/LMCache/LMCache/blob/dev/examples/mp_memory_coordinator/README.md

Rollout
-------

1. Deploy with ``actuation_enabled: false``; confirm the discovered (or
   adopted) inventory is exactly what you expect, then observe proposals and
   rejection reasons for at least 24 hours.
2. Confirm ``high_ratio`` is below any local eviction trigger.
3. Enable one donor/receiver canary with a five-minute cooldown and alert on
   ``BLOCKED`` (``/readyz`` turns 503, ``lmcache_memcoord_moves_blocked_total``).
4. Never scale down, downgrade, or detach the volume during an active or
   blocked move; disable by setting observation-only mode first and waiting
   for ``COMPLETE``.
