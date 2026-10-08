# SPDX-License-Identifier: Apache-2.0
"""Memory orchestrator: allocation and readiness authority for one shared
memory region (Device-DAX/CXL across hosts, or a file on one host).

One orchestrator process per region owns keys, extents, the region epoch,
write tokens and read leases; KV bytes never cross its API. MP servers talk to
it through ``client.MemoryOrchestratorClient``; ``server`` runs the process
(``lmcache memory``). The modules of this package depend only on gRPC,
protobuf and ``lmcache.logging``; none of them imports torch.
"""
