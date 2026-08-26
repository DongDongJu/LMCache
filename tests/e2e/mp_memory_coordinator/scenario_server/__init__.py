# SPDX-License-Identifier: Apache-2.0
"""Deterministic scenario server: fake MP Coordinator plus two fake MP servers.

Test double for the MP Memory Coordinator E2E suite. It is intentionally
independent of production code so that the server under test cannot tell it
is talking to a fake; see ``README.md`` in this directory.
"""
