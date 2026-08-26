# SPDX-License-Identifier: Apache-2.0
"""Durable state of the MP Memory Coordinator (journal, inventory, cooldowns).

Independent of the MP Coordinator's persistence package by design: the two
processes share no file, format, or code.
"""
