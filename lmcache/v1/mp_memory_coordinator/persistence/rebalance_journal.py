# SPDX-License-Identifier: Apache-2.0
"""Atomic, checksummed journal of moves, inventory, and cooldowns.

One JSON document (:class:`JournalDocument`) is the whole durable state.
Every save writes a temporary file beside the target, fsyncs it, renames it
over the target, and fsyncs the directory, so a reader sees either the
previous document or the new one. The file carries a schema version and a
SHA-256 checksum of its payload; a missing, corrupt, truncated,
checksum-invalid, or unknown-version file fails closed with
:class:`JournalCorruptError` -- the process must not guess.

File layout::

    {"schema_version": 1, "checksum": "sha256:<hex>", "payload": {...}}
"""

# Standard
from pathlib import Path
import hashlib
import json
import os

# Third Party
from pydantic import ValidationError

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.models import (
    JOURNAL_SCHEMA_VERSION,
    JournalDocument,
)

logger = init_logger(__name__)

JOURNAL_FILE_NAME = "journal.json"
_TMP_SUFFIX = ".tmp"
_CHECKSUM_PREFIX = "sha256:"


class JournalError(Exception):
    """Base of journal failures."""


class JournalCorruptError(JournalError):
    """The journal exists but cannot be trusted (fail closed)."""


class JournalVersionError(JournalCorruptError):
    """The journal was written by an unknown schema version."""


def compute_checksum(payload: dict[str, object]) -> str:
    """Return the ``sha256:<hex>`` checksum of a canonical JSON payload.

    Args:
        payload: The document payload (JSON-serializable).

    Returns:
        The checksum string.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _CHECKSUM_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RebalanceJournal:
    """The journal file inside the state directory."""

    def __init__(self, state_directory: Path) -> None:
        """Args:
        state_directory: Directory holding ``journal.json``; created on
            the first save.
        """
        self._directory = state_directory
        self._path = state_directory / JOURNAL_FILE_NAME

    @property
    def path(self) -> Path:
        """The journal file path."""
        return self._path

    def exists(self) -> bool:
        """Whether a journal file is present."""
        return self._path.exists()

    def load(self) -> JournalDocument:
        """Read and verify the journal.

        Returns:
            The document. A missing file yields a fresh, uninitialized
            document (first start).

        Raises:
            JournalVersionError: If ``schema_version`` is unknown.
            JournalCorruptError: If the file is not JSON, lacks the
                envelope, fails its checksum, or fails validation.
        """
        if not self._path.exists():
            return JournalDocument()
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            raise JournalCorruptError(f"{self._path}: unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise JournalCorruptError(f"{self._path}: envelope is not an object")
        version = raw.get("schema_version")
        if version != JOURNAL_SCHEMA_VERSION:
            raise JournalVersionError(
                f"{self._path}: schema_version {version!r} is not "
                f"{JOURNAL_SCHEMA_VERSION}; refusing to mutate"
            )
        payload = raw.get("payload")
        checksum = raw.get("checksum")
        if not isinstance(payload, dict) or not isinstance(checksum, str):
            raise JournalCorruptError(f"{self._path}: missing payload or checksum")
        if compute_checksum(payload) != checksum:
            raise JournalCorruptError(f"{self._path}: checksum mismatch")
        try:
            document = JournalDocument.model_validate(payload)
        except ValidationError as exc:
            raise JournalCorruptError(f"{self._path}: invalid payload: {exc}") from exc
        if document.schema_version != JOURNAL_SCHEMA_VERSION:
            raise JournalVersionError(f"{self._path}: payload schema mismatch")
        return document

    def save(self, document: JournalDocument) -> None:
        """Atomically replace the journal with ``document``.

        Writes ``journal.json.tmp``, fsyncs it, renames it over
        ``journal.json``, then fsyncs the directory.

        Args:
            document: The document to persist.

        Raises:
            OSError: If the directory or file cannot be written.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        payload = document.model_dump(mode="json")
        envelope = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "checksum": compute_checksum(payload),
            "payload": payload,
        }
        tmp = self._path.with_name(self._path.name + _TMP_SUFFIX)
        data = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._path)
        dir_fd = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
