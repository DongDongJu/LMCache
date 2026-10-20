# SPDX-License-Identifier: Apache-2.0
"""Media a shared L1 maps its region through: a Device-DAX device or a file."""

# Standard
from typing import Literal
import os
import stat

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_device_identity, get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    L1ManagerConfig,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import open_devdax_mapping
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)

DataPath = Literal["direct-pinned", "pageable-staging"]


def _char_device_size(device: os.stat_result) -> int | None:
    """Return the size sysfs reports for a character device, if it reports one."""
    path = f"/sys/dev/char/{os.major(device.st_rdev)}:{os.minor(device.st_rdev)}/size"
    try:
        with open(path) as size_file:
            return int(size_file.read().strip())
    except (OSError, ValueError):
        return None


def open_medium(
    config: L1ManagerConfig, capacity_bytes: int, alignment_bytes: int
) -> "RegionMedium":
    """Map the medium the L1 type names: a device for DEVDAX, else a file.

    Args:
        config: An L1 configuration with a ``shared`` section.
        capacity_bytes: Region capacity from the orchestrator contract.
        alignment_bytes: Extent alignment from the orchestrator contract.

    Returns:
        The mapped medium.

    Raises:
        ValueError: No ``shared`` section, or the medium rejects the path.
        OSError: The path cannot be opened or mapped.
    """
    if config.shared is None:
        raise ValueError("open_medium needs an L1 configuration with a shared section")
    if isinstance(config, DevDaxL1ManagerConfig):
        return DevDaxMedium(config.shared.path, capacity_bytes, alignment_bytes)
    if isinstance(config, DRAML1ManagerConfig):
        return FileMedium(config.shared.path, capacity_bytes, alignment_bytes)
    raise ValueError("A shared L1 requires a DRAM or DEVDAX configuration")


class RegionMedium:
    """One mapping of the region on this host: mmap, DMA registration, views.

    Args:
        path: The device or file to map.
        capacity_bytes: Region capacity from the orchestrator contract.
        alignment_bytes: Extent alignment from the orchestrator contract.

    Raises:
        OSError: The path cannot be opened or mapped.
    """

    backend_type: L1BackendType  # set by each subclass

    def __init__(self, path: str, capacity_bytes: int, alignment_bytes: int) -> None:
        self._path = path
        self._capacity_bytes = capacity_bytes
        self._alignment_bytes = alignment_bytes
        self._fd, self._mmap, self._mmap_buffer, self._buffer = open_devdax_mapping(
            path, capacity_bytes
        )
        self._base_address = self._buffer.data_ptr()
        self._pinned = current_device_spec.is_pin_supported and (
            current_device_spec.pin_memory(self._base_address, capacity_bytes)
        )
        self._data_path: DataPath = "direct-pinned"
        if not self._pinned:
            self._data_path = "pageable-staging"
            logger.warning(
                "Shared L1 %s is not registered for GPU DMA; copies stage "
                "through pageable memory",
                path,
            )

    @property
    def path(self) -> str:
        """The mapped device or file."""
        return self._path

    @property
    def capacity_bytes(self) -> int:
        """Mapped bytes, equal to the region capacity."""
        return self._capacity_bytes

    @property
    def data_path(self) -> DataPath:
        """How GPU copies reach the region: registered memory or staging."""
        return self._data_path

    def address(self, offset: int) -> int:
        """Return this process's address of a region offset.

        Args:
            offset: Byte offset inside the region.

        Returns:
            The virtual address of that byte in this process.
        """
        return self._base_address + offset

    def view(
        self, offset: int, length: int, layout: MemoryLayoutDesc
    ) -> TensorMemoryObj:
        """Build a memory object over a granted extent.

        Args:
            offset: Extent offset chosen by the orchestrator.
            length: Aligned extent length chosen by the orchestrator.
            layout: Shapes and dtypes of the object's tensors.

        Returns:
            A view with no parent allocator: dropping it never frees region bytes.

        Raises:
            ValueError: The extent is misaligned, too small or outside the region.
        """
        payload = get_size_bytes(layout.shapes, layout.dtypes)
        if (
            offset % self._alignment_bytes
            or length < payload
            or offset < 0
            or offset + length > self._capacity_bytes
        ):
            raise ValueError(
                f"extent [{offset}, {offset + length}) for a {payload}-byte "
                f"object does not fit the {self._capacity_bytes}-byte region"
            )
        return TensorMemoryObj(
            raw_data=self._buffer[offset : offset + length],
            metadata=MemoryObjMetadata(
                layout.shapes[0],
                layout.dtypes[0],
                offset,
                length,
                1,
                0,
                MemoryFormat.KV_2LTD,
                shapes=list(layout.shapes),
                dtypes=list(layout.dtypes),
            ),
            parent_allocator=None,
        )

    def owns_device(self, device_path: str) -> bool:
        """Whether ``device_path`` names the mapped device or file.

        Args:
            device_path: Candidate path or alias.

        Returns:
            True when both resolve to the same device or file.
        """
        identity = get_device_identity(device_path)
        return identity is not None and identity == get_device_identity(self._fd)

    def close(self) -> None:
        """Unregister the medium and unmap it unless a view is alive; idempotent."""
        if self._fd < 0:
            return
        if self._pinned:
            current_device_spec.unpin_memory(self._base_address)
            self._pinned = False
        self._buffer = torch.empty(0, dtype=torch.uint8)
        self._mmap_buffer = None
        try:
            self._mmap.close()
        except BufferError:
            logger.warning(
                "Shared L1 %s still has live views; leaving it mapped", self._path
            )
        os.close(self._fd)
        self._fd = -1


class DevDaxMedium(RegionMedium):
    """A Device-DAX character device: CXL memory that several hosts attach.

    Args:
        path: The ``/dev/daxX.Y`` device.
        capacity_bytes: Region capacity from the orchestrator contract.
        alignment_bytes: Extent alignment from the orchestrator contract.

    Raises:
        ValueError: The path is not a character device or holds less than the region.
        OSError: The device cannot be opened or mapped.
    """

    backend_type = L1BackendType.DEVDAX

    def __init__(self, path: str, capacity_bytes: int, alignment_bytes: int) -> None:
        device = os.stat(path)
        if not stat.S_ISCHR(device.st_mode):
            raise ValueError(
                f"Device-DAX medium {path!r} must be a character device; a "
                "shared DRAM L1 maps a file"
            )
        size = _char_device_size(device)
        if size is not None and size < capacity_bytes:
            raise ValueError(
                f"{path} holds {size} bytes, less than the region capacity of "
                f"{capacity_bytes} bytes"
            )
        super().__init__(path, capacity_bytes, alignment_bytes)


class FileMedium(RegionMedium):
    """A regular file, created or extended to the region capacity as needed.

    Args:
        path: The file to map, e.g. on tmpfs (``/dev/shm``) or hugetlbfs.
        capacity_bytes: Region capacity from the orchestrator contract.
        alignment_bytes: Extent alignment from the orchestrator contract.

    Raises:
        ValueError: The path exists and is not a regular file.
        OSError: The file cannot be created, opened or mapped.
    """

    backend_type = L1BackendType.DRAM

    def __init__(self, path: str, capacity_bytes: int, alignment_bytes: int) -> None:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"file medium {path!r} must be a regular file")
            if info.st_size < capacity_bytes:
                os.ftruncate(fd, capacity_bytes)
        finally:
            os.close(fd)
        super().__init__(path, capacity_bytes, alignment_bytes)
