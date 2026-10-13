# SPDX-License-Identifier: Apache-2.0
"""Reject corrupted or out-of-scope runtime wheels before staging an image."""

# Standard
from pathlib import Path
import base64
import csv
import hashlib
import io
import tempfile
import unittest
import zipfile

# Third Party
from stage_dynamo_overlay import wheel_files


class RuntimeWheelTest(unittest.TestCase):
    """Exercise wheel integrity using isolated synthetic packaging fixtures."""

    def make_wheel(
        self, folder: Path, *, tamper: bool = False, extra: str | None = None
    ) -> Path:
        """Construct packaging bytes only; this fixture is not a native runtime."""
        files = {
            "dynamo/_core.abi3.so": b"packaging-test-fixture",
            "ai_dynamo_runtime-1.5.1.dist-info/METADATA": (
                b"Name: ai-dynamo-runtime\nVersion: 1.5.1\n"
                b"Requires-Dist: typing-extensions>=4\n"
            ),
        }
        record = "ai_dynamo_runtime-1.5.1.dist-info/RECORD"
        stream = io.StringIO()
        writer = csv.writer(stream)
        for name, data in files.items():
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            writer.writerow([name, "sha256=" + encoded.decode().rstrip("="), len(data)])
        writer.writerow([record, "", ""])
        files[record] = stream.getvalue().encode()
        if tamper:
            files["dynamo/_core.abi3.so"] += b"unexpected"
        if extra:
            files[extra] = b"unlisted"
        target = folder / "runtime.whl"
        with zipfile.ZipFile(target, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return target

    def test_valid_record(self):
        """A complete exact RECORD exposes metadata and byte-identical payload."""
        with tempfile.TemporaryDirectory() as temp:
            files, metadata = wheel_files(self.make_wheel(Path(temp)))
        self.assertEqual(files["dynamo/_core.abi3.so"], b"packaging-test-fixture")
        self.assertEqual(metadata["version"], "1.5.1")
        self.assertEqual(metadata["requires_dist"], ["typing-extensions>=4"])

    def test_corrupt_native_payload(self):
        """A changed runtime binary is rejected before any staging write."""
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "RECORD mismatch"):
                wheel_files(self.make_wheel(Path(temp), tamper=True))

    def test_unlisted_file(self):
        """An unlisted namespace file cannot bypass RECORD verification."""
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "exact file set"):
                wheel_files(self.make_wheel(Path(temp), extra="dynamo/extra.py"))

    def test_forbidden_paths(self):
        """Traversal and unrelated-package writes cannot escape the native patch."""
        for name in ("../escape", "/absolute", "lmcache/foreign.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    wheel_files(self.make_wheel(Path(temp), extra=name))


if __name__ == "__main__":
    unittest.main()
