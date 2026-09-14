# UECC Handling in RawBlock L2

## Overview

Uncorrectable ECC (UECC) errors indicate physical media corruption on NVMe
devices. When detected, LMCache's raw-block backend transitions the device
to a blocked state, quarantines affected slots, and prevents further I/O
to protect data integrity.

## Configuration

### Enabling NVMe passthrough

```python
RawBlockL2AdapterConfig(
    device_path="/dev/ng0n1",
    use_uring_cmd=True,
    io_engine="io_uring",
    # ... other config ...
)
```

### Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| device_path | (required) | NVMe namespace character device |
| use_uring_cmd | False | Enable NVMe io_uring_cmd passthrough |
| io_engine | "posix" | I/O engine: "posix" or "io_uring" |
| meta_checkpoint_interval_sec | 60 | Periodic metadata checkpoint interval |
| meta_enable_periodic | True | Enable periodic checkpoint thread |

## UECC Detection

### Status code matching

UECC is detected when the NVMe completion status word satisfies:

```
(status & 0x07ff) == 0x0281
```

The Rust raw-block device exposes this as `OSError.errno`.

### Status field decoding

| Field | Bits | Description |
|-------|------|-------------|
| status_code | 0-10 | Raw status code (0x0281 = Unrecoverable ECC) |
| DNR | 11 | Do Not Retry flag |
| SCT | 11-13 | Status Code Type |
| SC | 0-7 | Status Code within SCT |

### Python API

```python
from lmcache.v1.storage_backend.raw_block.uecc import (
    classify_oserror,
    is_uecc_errno,
    decode_uecc_status,
)

# Quick check
if is_uecc_errno(exc.errno):
    # This is a UECC error

# Full classification
result = classify_oserror(exc)
# result["is_uecc"], result["sct"], result["sc"], result["dnr"]

# Decode raw status word
fields = decode_uecc_status(0x4281)
# fields["is_uecc"] = True, fields["dnr"] = True
```

## Safety Gates

### Device lifecycle states

```
[ACTIVE] --> [BLOCKED]  (on first UECC)
```

Once blocked, the device remains blocked for its lifetime. It is NOT
automatically recovered.### Behavior in each state

| Operation | ACTIVE | BLOCKED |
|-----------|--------|---------|
| put_many | Write to device | Return all False (no I/O) |
| load_many_into | Read from device | Return all False (no I/O) |
| get_metadata_prefix | Read metadata | Return [] |
| exists_many | Check index | Return all False |
| delete_many | Modify index | Return all False |
| Checkpoint (periodic) | Write checkpoint | Skip |
| Checkpoint (force) | Write checkpoint | Skip (even force=True) |
| Close | Final checkpoint + close | Skip checkpoint, close device |

### Quarantine

- Poisoned slots are added to `_quarantined_slots`
- Quarantined slots are excluded from free lists (global and FDP)
- Quarantined slots are never re-allocated

## Expected Output

### Critical log on UECC

```
CRITICAL lmcache.v1.storage_backend.raw_block.core: UECC detected on 
device /dev/ng0n1 (status=0x0281, uecc_count=1). Device blocked.
```

### report_status output (blocked device)

```python
{
    "is_healthy": False,
    "device_active": False,
    "device_blocked": True,
    "critical_error": True,
    "uecc_count": 1,
    "quarantined_slot_count": 3,
    "replacement_recommended": True,
    # ... other fields ...
}
```

## Monitoring

### Checking device health

```python
adapter = storage_manager.get_l2_adapter("raw_block")
status = adapter.report_status()

if not status["is_healthy"]:
    print(f"Device blocked. UECC count: {status['uecc_count']}")
    print(f"Quarantined slots: {status['quarantined_slot_count']}")
    print("Replacement recommended: True")
```

### Log patterns to watch for

1. `CRITICAL ... UECC detected` - first UECC event, device blocked
2. `WARNING ... Periodic raw-block metadata checkpoint failed` - may indicate issues
3. `ERROR ... RawBlockCore load failed` - non-UECC read errors## Recovery Procedures

### Immediate response

1. Check `report_status()` to confirm device is blocked
2. Review logs for the critical UECC message and status code
3. If `replacement_recommended=True`, plan hardware replacement

### Short-term mitigation

1. Stop all cache operations on the affected device
2. Do NOT reinitialize the device - the metadata may still be valid
3. If the cache is a hot spare, the system should failover to another node/device

### Hardware replacement

1. Physically replace the NVMe device
2. Update `device_path` in configuration
3. Initialize a new `RawBlockCore` with the new device path
4. The cache engine will start with an empty index on the new device
5. Existing clients will need to rebuild their cache entries

### Do NOT

- Do NOT ignore UECC errors and continue using the device
- Do NOT call `close()` and reopen the same device (state is not recoverable)
- Do NOT attempt to re-read poisoned slots (data is corrupted)
- Do NOT use `force=True` on checkpoint after UECC

## Testing

### Software validation (no hardware)

```bash
pytest -xvs tests/v1/storage_backend/test_raw_block_uecc.py \
          tests/v1/storage_backend/test_raw_block_uecc_integration.py
```

### Hardware validation (opt-in)

```bash
LMCACHE_TEST_HW_URING_CMD=1 \
LMCACHE_TEST_URING_CMD_PATH=/dev/ng0n1 \
    pytest -xvs tests/v1/storage_backend/test_raw_block_uecc_harness.py
```

## Architecture reference

```
RawBlockL2Adapter (L2 interface)
    |
    +-- ThreadPoolExecutor (store/lookup/load pools)
    |       |
    |       +-- RawBlockCore (storage engine)
    |               |
    |               +-- RawBlockDevice (Rust, via lmcache_rust_raw_block_io)
    |                       |
    |                       +-- NVMe io_uring_cmd (hardware)
    |
    +-- EventFd notifiers (store/lookup/load)
```