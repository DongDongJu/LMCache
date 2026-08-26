# SPDX-License-Identifier: Apache-2.0
"""Standalone MP Memory Coordinator.

Rebalances LMCache-visible Device-DAX capacity between MP servers: it reads
fleet membership and occupancy from the MP Coordinator's ``/instances`` and
``/instances/usage`` endpoints, picks one LOW donor and one HIGH receiver
after three stable samples, and moves one managed allocation through a
crash-safe saga (donor drain -> donor remove -> outside deallocate -> outside
allocate -> receiver add).

This package is a separate process with its own configuration, journal,
inventory, Lease, and lifecycle. It talks to the MP Coordinator over HTTP only
and never imports :mod:`lmcache.v1.mp_coordinator` (enforced by
``tests/v1/mp_memory_coordinator/test_architecture_boundary.py``).

See ``docs/design/v1/mp_memory_coordinator/`` for the design.
"""
